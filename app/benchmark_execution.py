"""Isolated benchmark execution; no implicit human answers or budget extensions."""
import time
import json

from .review_policy import is_blocking


class BenchmarkExecution:
    def __init__(self, workspace, factory, mode, execution, human_policy, max_steps, llm_options=None):
        self.workspace, self.factory = workspace, factory
        self.mode, self.execution = mode, execution
        self.human_policy, self.max_steps = human_policy, max_steps
        self.count = 0
        self.llm_options = {"stream": True, "reasoning_effort": "low", "timeout_seconds": 180}
        self.llm_options.update(llm_options or {})

    def sample(self, sample_id, operation, quality_applicable=True):
        self.count += 1
        # Index rather than external sample ID: neither traversal nor duplicate IDs
        # can cause different samples to share state.
        worker = self.factory(self.workspace / "sample-{:04d}".format(self.count))
        worker.knowledge_policy = "sample_only"
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

        def observed(*args, **kwargs):
            calls.append(1)
            worker.llm.last_call_diagnostics = {}
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                failures.append(getattr(exc, "code", type(exc).__name__))
                raise
            finally:
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
        return result

    def pipeline(self, worker, title, requirement, target="cases"):
        project = worker.store.create_project(title, requirement)
        run = None
        confirmations = []

        def confirm():
            # This is an explicitly selected benchmark fixture, NOT a human
            # acceptance event. Do not write human_gate facts/feedback.
            project.module_tree.confirmed = True
            worker.store.save_project(project)
            confirmations.append({"source": "benchmark_simulator", "action": "confirm_modules"})

        if self.execution == "agentic":
            from .supervisor import AgenticSupervisor
            controller = AgenticSupervisor(worker)
            goal = ("仅完成需求理解与模块规划，随后请求人工确认模块，不生成用例。" if target == "modules" else
                    "生成满足需求的测试用例，根据评审反馈修复并完成收尾；缺少业务依据时请求澄清。")
            run = controller.start(project.id, goal, self.max_steps)
            project = worker.store.get_project(project.id)
            if (target == "cases" and run.status == "waiting_confirmation" and
                    self.human_policy == "simulate_confirm" and len(run.steps) < run.max_steps and
                    project.module_tree and project.module_tree.modules):
                confirm()
                run = controller.resume(project.id, run.id)
                project = worker.store.get_project(project.id)
            status = run.status
            complete = (status in {"completed", "needs_attention"} if target == "cases" else
                        status == "waiting_confirmation" and bool(project.analysis and project.module_tree and project.module_tree.modules))
        else:
            project = worker.analyze(project)
            project = worker.plan_modules(project)
            complete = bool(project.analysis and project.module_tree and project.module_tree.modules)
            status = "completed" if target == "modules" else "waiting_confirmation"
            if target == "cases":
                complete = False
                if self.human_policy == "simulate_confirm" and project.module_tree and project.module_tree.modules:
                    confirm()
                    project = worker.generate_cases(project)
                    project = worker.review(project)
                    complete, status = True, "completed"

        review = project.review if target == "cases" else project.module_review
        incomplete = bool(review and any(f.category == "review_incomplete" for f in review.findings))
        gate = None
        if review and not incomplete:
            gate = (bool(project.cases) and not any(is_blocking(f) for f in review.findings) if target == "cases" else
                    not any(f.severity in {"high", "critical", "error"} for f in review.findings))
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
                  "question": run.question if run else ("模块等待确认；批量生成需显式选择模拟确认。" if status == "waiting_confirmation" else "")}
        if incomplete:
            result["error_code"] = "review_incomplete"
        if run:
            result["actions"] = [{"index": s.index, "status": s.status,
                                  "capability": s.decision.capability or s.decision.action if s.decision else "planner_error"}
                                 for s in run.steps]
        return result
