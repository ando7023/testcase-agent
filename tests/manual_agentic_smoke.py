"""Opt-in live smoke test. Never discovered by unittest's test_*.py pattern."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from app.llm import LLMError
from app.agents import CaseReviewAgent
from app.review_policy import restore_review_history
from app.orchestrator import TestCaseOrchestrator
from app.store import JsonStore
from app.supervisor import AgenticSupervisor


REQUIREMENT = """优惠券领取规则：已登录用户可领取活动优惠券，每人每个活动最多领取1张。
领取成功时库存减1并保存领取记录；库存为0时拒绝领取且不产生领取记录。
同一用户重复或并发领取只允许一次成功，库存不得为负数。
未登录用户不能领取；库存写入失败时回滚领取记录和库存，允许用户重试。
所有失败都返回明确的失败原因。仅涉及领券，不涉及支付、核销和工单审核。"""
GOAL = "仅根据当前优惠券需求生成并评审测试用例，不引入 设备借用/工单知识；模块尽量控制在3个、用例约10条，覆盖正常、边界、异常、权限、并发幂等。根据评审反馈决定修复或补充，最后重新评审。"


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


class LoggingStore(JsonStore):
    def __init__(self, root):
        super().__init__(root)
        self.last = None

    def save_agent_run(self, run):
        super().save_agent_run(run)
        if not run.steps:
            return
        step = run.steps[-1]
        signature = (run.id, step.index, step.status, run.status)
        if signature != self.last:
            self.last = signature
            emit(event="step", run_id=run.id, index=step.index, status=step.status,
                 capability=step.decision.capability or step.decision.action if step.decision else None,
                 skills=step.decision.skills if step.decision else [],
                 reason=step.decision.reason if step.decision else "",
                 observation=step.observation, run_status=run.status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--continue-run", action="store_true")
    operation.add_argument("--restart-run", action="store_true", help="Start a new run on existing artifacts after failure")
    operation.add_argument("--review-only", action="store_true", help="Diagnose one review request without changing project or run")
    parser.add_argument("--stream", action="store_true", help="Receive SSE events for JSON calls and log progress counts")
    parser.add_argument("--model", default="", help="Override model for this test without editing .env")
    parser.add_argument("--reasoning-effort", choices=["low", "high", "max"], default=None,
                        help="Override GLM thinking effort for this test without editing .env")
    parser.add_argument("--confirm-test-modules", action="store_true")
    parser.add_argument("--answer", default="")
    parser.add_argument("--additional-steps", type=int, default=0, choices=range(21))
    parser.add_argument("--max-steps", type=int, default=12, choices=range(1, 21))
    parser.add_argument("--timeout-seconds", type=int, default=None,
                        help="Override this test's model timeout without editing .env")
    args = parser.parse_args()
    if args.timeout_seconds is not None and not 1 <= args.timeout_seconds <= 900:
        parser.error("--timeout-seconds must be between 1 and 900")
    existing = args.continue_run or args.restart_run or args.review_only
    if existing and not args.root:
        parser.error("--root is required when continuing or restarting")
    if args.additional_steps and not args.continue_run:
        parser.error("--additional-steps requires --continue-run; use --max-steps for a new run")
    base = (ROOT / "data" / "live_smoke").resolve()
    folder = Path(args.root).resolve() if args.root else base / datetime.now().strftime("glm-%Y%m%d-%H%M%S")
    if base not in folder.parents:
        raise ValueError("Live smoke artifacts must be below data/live_smoke")
    if existing and not (folder / "manifest.json").exists():
        raise ValueError("Existing smoke manifest required")
    if not existing and folder.exists():
        raise ValueError("Use a fresh directory for a new smoke test")
    os.environ["RAG_EMBEDDING_PROVIDER"] = "hashing"
    os.environ["MEMORY_EMBEDDING_PROVIDER"] = "hashing"
    store = LoggingStore(folder)
    worker = TestCaseOrchestrator(store)
    if args.model.strip():
        worker.llm.model = args.model.strip()
    if args.reasoning_effort is not None:
        worker.llm.reasoning_effort = args.reasoning_effort
    if args.timeout_seconds is not None:
        worker.llm.timeout = args.timeout_seconds
    if args.stream:
        worker.llm.stream_json = True
    if not worker.llm.enabled:
        raise ValueError("A live model key is required for this opt-in test")
    calls = []
    original = worker.llm.generate_json
    last_progress = [0.0]

    def progress(info):
        now = time.monotonic()
        if info["phase"] != "receiving" or now - last_progress[0] >= 15:
            emit(event="llm_progress", call=len(calls), **info)
            last_progress[0] = now

    worker.llm.on_stream_progress = progress

    def save_response(content):
        # Keep final answer only, never reasoning deltas, headers or credentials.
        path = folder / ("response-" + str(time.time_ns()) + ".txt")
        path.write_text(content.replace(worker.llm.api_key, "[REDACTED]"), encoding="utf-8")
        if calls:
            calls[-1]["response_file"] = str(path)

    worker.llm.on_json_response = save_response

    def measured(*positional, **kwargs):
        if len(calls) >= 40:
            raise LLMError("Smoke test reached its 40-call budget")
        entry = {"index": len(calls) + 1, "status": "running"}
        entry["input_sha256"] = hashlib.sha256(json.dumps(
            {"args": positional, "kwargs": kwargs}, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        calls.append(entry)
        emit(event="llm_start", call=entry["index"])
        started = time.perf_counter()
        try:
            result = original(*positional, **kwargs)
            entry["status"] = "success"
            return result
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc).replace(worker.llm.api_key, "[REDACTED]")[:1000]
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - started, 2)
            emit(event="llm_end", **entry)

    worker.llm.generate_json = measured
    if existing:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        project = store.get_project(manifest["project_id"])
        if args.restart_run:
            previous_run = store.get_agent_run(project.id, manifest["run_id"])
            if previous_run.status not in {"failed", "budget_exhausted", "needs_attention"}:
                raise ValueError("Restart requires a failed, exhausted or needs-attention run; resume paused runs")
            manifest.setdefault("previous_run_ids", []).append(previous_run.id)
    else:
        project = store.create_project("GLM Agentic 实测：优惠券领取", REQUIREMENT,
                                       "这是独立的合成测试需求，信息完整；不要增加其他业务流程。")
        manifest = {"project_id": project.id, "model": worker.llm.model, "root": str(folder)}
        (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    settings = {"model": worker.llm.model, "reasoning_effort": worker.llm.reasoning_effort,
                "temperature": worker.llm.temperature, "top_p": worker.llm.top_p,
                "max_tokens": worker.llm.max_tokens, "base_url": worker.llm.base_url}
    emit(event="begin", timeout_seconds=worker.llm.timeout, stream=worker.llm.stream_json, **{**manifest, **settings})
    if args.review_only:
        if not project.analysis or not project.module_tree or not project.cases:
            raise ValueError("Review diagnosis requires existing analysis, modules and cases")
        source_run = store.get_agent_run(project.id, manifest["run_id"])
        restore_review_history(source_run)
        with worker.tracer.run("live_smoke.review_diagnostic", project_id=project.id) as trace:
            result = CaseReviewAgent(worker.llm).run({
                "requirement": project.requirement,
                "project_context": project.context,
                "review_constraints": "\n".join([source_run.goal] + source_run.responses + ([args.answer] if args.answer else [])),
                "review_history": source_run.review_history,
                "issue_ledger": source_run.issue_ledger,
                "analysis": project.analysis.model_dump(),
                "module_tree": project.module_tree.model_dump(),
                "cases": [case.model_dump() for case in project.cases],
            }, {})
        ok = bool(calls) and all(call["status"] == "success" for call in calls)
        report = {**settings, "status": "reviewed" if ok else "review_incomplete", "calls": calls,
                  "trace_id": trace.trace_id, "stream": worker.llm.stream_json,
                  "timeout_seconds": worker.llm.timeout, "review": result["review"],
                  "usage": [span.usage for span in trace.spans if span.kind == "llm"]}
        path = folder / ("report-review-" + str(time.time_ns()) + ".json")
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        emit(event="finished", report=str(path), status=report["status"],
             score=result["review"]["score"], calls=len(calls))
        return 0 if ok else 1
    supervisor = AgenticSupervisor(worker)
    started = time.perf_counter()
    with worker.tracer.run("live_smoke.agentic", project_id=project.id) as trace:
        if args.confirm_test_modules:
            if not args.continue_run or not project.module_tree or not project.module_tree.modules:
                raise ValueError("Only an existing smoke test module tree can be confirmed")
            worker.confirm_modules(project, [m.model_dump() for m in project.module_tree.modules])
            emit(event="test_fixture_modules_confirmed", module_count=len(project.module_tree.modules))
        run = supervisor.resume(project.id, manifest["run_id"], args.answer, args.additional_steps) if args.continue_run else supervisor.start(project.id, GOAL + ("\n" + args.answer if args.answer else ""), args.max_steps)
        manifest["run_id"] = run.id
    store.attach_trace_run(project.id, trace.trace_id)
    project = store.get_project(project.id)
    manifest["status"] = run.status
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        **manifest, **settings, "seconds": round(time.perf_counter() - started, 2), "calls": calls,
        "timeout_seconds": worker.llm.timeout,
        "stream": worker.llm.stream_json,
        "trace_id": trace.trace_id, "degraded": run.degraded, "question": run.question,
        "error": run.error, "steps": len(run.steps), "case_count": len(project.cases),
        "review": project.review.model_dump() if project.review else None,
        "modules": project.module_tree.model_dump() if project.module_tree else None,
        "worker_traces": [{"agent": t.agent, "mode": t.mode, "status": t.status, "error": t.error} for t in project.traces],
        "usage": {key: sum(span.usage.get(key, 0) for span in trace.spans if span.kind == "llm")
                  for key in ("input_tokens", "output_tokens", "total_tokens")},
    }
    report_path = folder / ("report-restart.json" if args.restart_run else "report-continue.json" if args.continue_run else "report-start.json")
    if report_path.exists():
        archive = folder / (report_path.stem + "-" + str(time.time_ns()) + ".json")
        archive.write_bytes(report_path.read_bytes())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(event="finished", report=str(report_path), **{k: report[k] for k in ("status", "seconds", "steps", "case_count", "degraded", "error", "question", "usage")})
    return 1 if run.status in {"failed", "needs_attention"} or run.degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
