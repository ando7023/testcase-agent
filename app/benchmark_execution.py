"""Isolated benchmark execution; no implicit human answers or budget extensions."""
import time
import json

from .review_policy import is_blocking
from .clarification_policy import behavior_blockers, scope_summary


class BenchmarkExecution:
    def __init__(self, workspace, factory, mode, execution, human_policy, max_steps, llm_options=None, on_event=None, clarification_policy="strict"):
        self.workspace, self.factory = workspace, factory
        self.mode, self.execution = mode, execution
        self.human_policy, self.max_steps = human_policy, max_steps
        self.count = 0
        self.llm_options = {"stream": True, "reasoning_effort": "low", "timeout_seconds": 180}
        self.llm_options.update(llm_options or {})
        self.on_event = on_event
        if clarification_policy not in {"strict", "evidence_only"}:
            raise ValueError("Invalid clarification policy")
        self.clarification_policy = clarification_policy

    def emit(self, event):
        if self.on_event:
            self.on_event(event)

    def sample(self, sample_id, operation, quality_applicable=True):
        self.count += 1
        self.emit({"event": "sample_start", "sample_id": sample_id, "sample_index": self.count})
        # Index rather than external sample ID: neither traversal nor duplicate IDs
        # can cause different samples to share state.
        worker = self.factory(self.workspace / "sample-{:04d}".format(self.count))
        worker.knowledge_policy = "sample_only"
        worker.clarification_policy = self.clarification_policy
        worker.harness.allow_fallback = self.mode != "live"
        worker.llm.strict_review_failures = self.mode == "live"
        worker.llm.stream_json = self.llm_options["stream"]
        worker.llm.reasoning_effort = self.llm_options["reasoning_effort"]
        worker.llm.timeout = self.llm_options["timeout_seconds"]
        if self.mode == "offline":
            worker.llm.api_key = ""
        elif not worker.llm.enabled:
            raise ValueError("真实模型评测需要启用模型并配置密钥，不能静默改为离线模式")
        start = time.perf_counter()
        original = worker.llm.generate_json
        calls, failures, diagnostics = [], [], []
        active_agent = ["supervisor" if self.execution == "agentic" else "worker"]
        def agent_event(event):
            if event["event"] == "agent_start":
                active_agent[0] = event["agent"]
            self.emit(dict(event, sample_id=sample_id))
            if event["event"] == "agent_end":
                active_agent[0] = "supervisor" if self.execution == "agentic" else "worker"
        worker.harness.on_event = agent_event

        def observed(*args, **kwargs):
            calls.append(1)
            worker.llm.last_call_diagnostics = {}
            call_id = "{}-{}".format(self.count, len(calls))
            meta = {"sample_id": sample_id, "call_id": call_id, "call": len(calls), "agent": active_agent[0]}
            self.emit(dict(meta, event="llm_start", stream=worker.llm.stream_json))
            pending, last_flush, last_progress = [], [time.perf_counter()], [0.0]
            def flush():
                if pending:
                    self.emit(dict(meta, event="delta", text="".join(pending)))
                    pending.clear()
                    last_flush[0] = time.perf_counter()
            def delta(text):
                if not self.on_event:
                    return
                pending.append(text)
                if time.perf_counter() - last_flush[0] >= 0.06 or len(pending) >= 32:
                    flush()
            def progress(data):
                now = time.perf_counter()
                if now - last_progress[0] >= 1 or data.get("phase") in {"connected", "complete", "timeout"}:
                    # Forward counters only, never reasoning_content or raw errors.
                    safe = {k: data[k] for k in ("phase", "seconds", "event_count", "content_chars") if k in data}
                    self.emit(dict(meta, event="llm_progress", **safe))
                    last_progress[0] = now
            previous_delta = worker.llm.on_content_delta
            previous_progress = worker.llm.on_stream_progress
            worker.llm.on_content_delta = delta
            worker.llm.on_stream_progress = progress
            status = "success"
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                status = "error"
                failures.append(getattr(exc, "code", type(exc).__name__))
                raise
            finally:
                flush()
                self.emit(dict(meta, event="llm_end", status=status))
                worker.llm.on_content_delta = previous_delta
                worker.llm.on_stream_progress = previous_progress
                detail = {"call": len(calls), **worker.llm.last_call_diagnostics}
                try:
                    prompt = json.loads(args[1] if len(args) > 1 else kwargs.get("user_prompt", ""))
                    if isinstance(prompt, dict) and prompt.get("review_phase"):
                        detail.update(review_phase=prompt["review_phase"], case_count=len(prompt.get("cases", [])))
                except (TypeError, ValueError):
                    pass
                diagnostics.append(detail)

        worker.llm.generate_json = observed
        trace = None
        result = {"id": sample_id, "flow_completed": False, "quality_passed": None,
                  "quality_applicable": quality_applicable,
                  "technical_failure": False, "degraded": False, "status": "incomplete"}
        try:
            with worker.tracer.run("benchmark.sample") as trace:
                result.update(operation(worker))
        except Exception as exc:
            # Provider exceptions can contain response bodies/credentials. Keep
            # only a safe class/code in the aggregate report; traces stay local.
            result.update(technical_failure=True, error_code=getattr(exc, "code", type(exc).__name__),
                          error="样本执行异常；请检查该样本的本地 Trace。")
        finally:
            worker.llm.generate_json = original
        project = result.pop("_project", None)
        if project is None:
            projects = worker.store.list_projects()
            project = projects[0] if projects else None
        modes = result.pop("_worker_modes", []) + ([t.mode for t in project.traces] if project else [])
        result["degraded"] = self.mode == "live" and bool(
            result["degraded"] or failures or any(m in {"demo", "fallback"} for m in modes))
        if result["technical_failure"]:
            result.update(status="technical_failed", quality_passed=None)
        elif result["degraded"]:
            result.update(status="degraded", quality_passed=None)
        elif result["flow_completed"]:
            result["status"] = ("passed" if result["quality_passed"] is True else
                                "quality_failed" if result["quality_passed"] is False else "completed")
        result.update(seconds=round(time.perf_counter() - start, 3), llm_calls=len(calls),
                      knowledge_policy="sample_only", llm_config=worker.llm.effective_settings(),
                      call_diagnostics=diagnostics,
                      llm_error_count=len(failures), worker_modes=modes,
                      validation_level="model" if self.mode == "live" else "offline_structural",
                      trace_id=trace.trace_id if trace else None, workspace=str(worker.store.root.relative_to(self.workspace)),
                      model=worker.llm.model if self.mode == "live" else None)
        if project:
            result["project_id"] = project.id
            result.update(scope_summary(project))
            if project.review:
                details = [f.model_dump() for f in project.review.findings if f.disposition == "clarification"]
                result["review_clarifications"] = details
                blocking_findings = [f for f in project.review.findings if is_blocking(f, project.clarification_policy)]
                result["blocked_requirement_ids"] = sorted(set(result["blocked_requirement_ids"]) | {
                    rid for f in blocking_findings for rid in f.requirement_ids})
                if blocking_findings:
                    result["execution_readiness"] = "blocked"
                elif any(f.clarification_kind == "execution_detail" for f in project.review.findings):
                    result["execution_readiness"] = "needs_preparation"
            if self._module_blocked(project):
                result["execution_readiness"] = "blocked"
        self.emit({"event": "sample_end", "sample_id": sample_id, "status": result["status"],
                   "question": result.get("question", ""), "technical_failure": result["technical_failure"]})
        return result

    def pipeline(self, worker, title, requirement, target="cases"):
        project = worker.store.create_project(title, requirement)
        project.clarification_policy = self.clarification_policy
        worker.store.save_project(project)
        run = None
        confirmations = []

        if self.execution == "agentic":
            from .supervisor import AgenticSupervisor
            controller = AgenticSupervisor(worker, on_event=self.emit)
            goal = ("仅完成需求理解、模块规划和模块评审后收尾，不生成用例。" if target == "modules" else
                    "生成满足需求的测试用例，根据评审反馈修复并完成收尾；按配置的澄清策略判断缺失信息，不能补造契约。")
            run = controller.start(project.id, goal, self.max_steps, target=target)
            project = worker.store.get_project(project.id)
            status = run.status
            complete = status in {"completed", "needs_attention"}
        else:
            project = worker.analyze(project)
            project = worker.plan_modules(project)
            complete = bool(project.analysis and project.module_tree and project.module_tree.modules)
            status = "completed" if complete else "waiting_input"
            if target == "cases":
                complete = False
                if (project.module_tree and project.module_tree.modules
                        and not behavior_blockers(project) and not self._module_blocked(project)):
                    project = worker.generate_cases(project)
                    project = worker.review(project)
                    complete, status = True, "completed"
            if behavior_blockers(project) or self._module_blocked(project):
                complete, status = False, "waiting_input"

        review = project.review if target == "cases" else project.module_review
        incomplete = bool(review and any(f.category == "review_incomplete" for f in review.findings))
        gate = None
        if review and not incomplete:
            gate = (bool(project.cases) and not any(is_blocking(f, project.clarification_policy) for f in review.findings) if target == "cases" else
                    not any(f.severity in {"high", "critical", "error"} for f in review.findings))
            if behavior_blockers(project):
                gate = False
        if run and target == "cases" and gate:
            from .supervisor import artifact_fingerprint
            gate = run.review_fingerprint == artifact_fingerprint(project)
        result = {"_project": project, "flow_completed": complete, "status": status,
                  "quality_passed": gate if complete else None, "gate_passed": gate,
                  "quality_basis": "case_review_gate" if target == "cases" else "module_review_gate",
                  "technical_failure": incomplete or bool(run and run.status == "failed"),
                  "degraded": bool(run and run.degraded), "simulated_confirmations": confirmations,
                  "completion_criterion": "finish" if target == "cases" else "modules_planned",
                  "run_id": run.id if run else None, "run_status": run.status if run else status,
                  "steps": len(run.steps) if run else 0,
                  "question": run.question if run else "",
                  "module_confirmation_required": False}
        if incomplete:
            result["error_code"] = "review_incomplete"
        if run:
            result["actions"] = [{"index": s.index, "status": s.status,
                                  "capability": s.decision.capability or s.decision.action if s.decision else "planner_error"}
                                 for s in run.steps]
        if not run and behavior_blockers(project):
            result["question"] = "\n".join(g.question for g in behavior_blockers(project))
        elif not run and self._module_blocked(project):
            result["question"] = "模块评审存在阻塞问题，请先修正模块范围或补充业务依据。"
        return result

    @staticmethod
    def _module_blocked(project):
        return (project.module_review and any(f.severity in {"high", "critical", "error"}
                or f.category == "review_incomplete" for f in project.module_review.findings))
