import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agents import CaseGenerationAgent
from app.llm import LLMError
from app.models import ParsedDocument, ReviewFinding, ReviewReport
from app.orchestrator import TestCaseOrchestrator
from app.skills import SKILLS
from app.store import JsonStore
from app.supervisor import AgenticSupervisor, artifact_fingerprint


REQUIREMENT = "用户登录后领取优惠券，每个用户每次活动最多领取一张，库存不足时不能领取。支付失败退回优惠券。管理员可以发布活动，普通用户不能访问管理功能。"


def invoke(name, **kwargs):
    return dict(action="invoke_tool" if name in {"inspect_materials", "search_knowledge"} else "invoke_agent",
                capability=name, reason="根据当前产物决策", **kwargs)


def finish():
    return dict(action="finish", reason="完成本轮目标")


def ask():
    return dict(action="request_input", reason="需要业务确认", question="请确认模块或补充权限规则。")


class ScriptedPlanner:
    enabled = True

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.inputs = []

    def generate_json(self, system, user, schema):
        self.inputs.append(json.loads(user))
        if not self.decisions:
            raise AssertionError("Unexpected extra planner call")
        decision = self.decisions.pop(0)
        if isinstance(decision, Exception):
            raise decision
        return decision


class SupervisorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = JsonStore(Path(self.temp.name))
        self.worker = TestCaseOrchestrator(self.store)
        self.worker.llm.api_key = ""
        self.project = self.store.create_project("优惠券", REQUIREMENT)

    def prepared(self, cases=False, confirmed=True):
        p = self.worker.analyze(self.project)
        p = self.worker.plan_modules(p)
        if confirmed:
            p = self.worker.confirm_modules(p, [m.model_dump() for m in p.module_tree.modules])
        if cases:
            p = self.worker.generate_cases(p)
        return p

    def run_script(self, decisions, max_steps=12):
        planner = ScriptedPlanner(decisions)
        run = AgenticSupervisor(self.worker, planner).start(self.project.id, max_steps=max_steps)
        return run, planner

    def test_user_constraints_and_original_requirement_reach_real_critic_prompt(self):
        self.prepared(cases=True)
        planner = ScriptedPlanner([ask(), invoke("quality_critic"), ask()])
        supervisor = AgenticSupervisor(self.worker, planner)
        run = supervisor.start(self.project.id)
        self.worker.llm.api_key = "test-no-network"
        prompts = []
        def critique(system, user, *args):
            prompts.append(json.loads(user))
            return {"findings": []}
        with patch.object(self.worker.llm, "generate_json", side_effect=critique):
            supervisor.resume(self.project.id, run.id, "USER_CONSTRAINT_NO_INVENTED_ERROR_CODES")
        self.assertEqual(prompts[0]["raw_requirement"], REQUIREMENT)
        self.assertIn("USER_CONSTRAINT_NO_INVENTED_ERROR_CODES", prompts[0]["user_constraints"])
        self.assertTrue(prompts[0]["analysis"]["atomic_requirements"])

    def policy_review(self, disposition="defect", severity="medium"):
        def review(project, **kwargs):
            project.review = ReviewReport(score=90, findings=[ReviewFinding(
                severity=severity, category="semantic", case_id=project.cases[0].id,
                message="需要核对业务契约", disposition=disposition, issue_type="assertion_semantics")])
            self.store.save_project(project)
            return project
        return review

    def test_nonblocking_suggestions_allow_finish_but_remain_in_ledger(self):
        self.prepared(cases=True)
        with patch.object(self.worker, "review", side_effect=self.policy_review("suggestion", "low")):
            run, _ = self.run_script([invoke("quality_critic"), finish()])
        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.issue_ledger), 1)
        self.assertEqual(len(self.store.get_project(self.project.id).review.findings), 1)

    def test_missing_contract_pauses_before_any_repair_or_finish(self):
        self.prepared(cases=True)
        with patch.object(self.worker, "review", side_effect=self.policy_review("clarification")):
            run, planner = self.run_script([invoke("quality_critic"), finish()])
        self.assertEqual(run.status, "waiting_input")
        self.assertEqual(len(planner.inputs), 1)
        self.assertIn("依据", run.question)
        self.assertTrue(AgenticSupervisor._blocking(self.store.get_project(self.project.id)))

    def test_two_repair_rounds_stop_for_adjudication_instead_of_budget_extension(self):
        self.prepared(cases=True)
        def revision(project, **kwargs):
            project.cases[0].preconditions.append("修改但问题仍存在")
            project.review = None
            self.store.save_project(project)
            return project
        decisions = [invoke("quality_critic"), invoke("case_revision", skills=["boundary"]),
                     invoke("quality_critic"), invoke("case_revision", skills=["boundary"]), invoke("quality_critic"), finish()]
        with patch.object(self.worker, "review", side_effect=self.policy_review()), patch.object(self.worker, "revise_once", side_effect=revision):
            run, planner = self.run_script(decisions)
        self.assertEqual(run.status, "waiting_input")
        self.assertEqual(run.repair_rounds, 2)
        self.assertEqual(len(planner.inputs), 5)
        self.assertIn("两轮", run.question)
        self.assertIn("不代表缺少业务依据", run.question)
        self.assertEqual(next(iter(run.issue_ledger.values()))["seen_reviews"], 3)
        self.assertEqual(self.store.get_agent_run(self.project.id, run.id).repair_rounds, 2)

    def test_noop_repair_is_detected_even_with_a_new_score(self):
        self.prepared(cases=True)
        def revision(project, **kwargs):
            project.review = None
            self.store.save_project(project)
            return project
        with patch.object(self.worker, "review", side_effect=self.policy_review()), patch.object(self.worker, "revise_once", side_effect=revision):
            run, _ = self.run_script([invoke("quality_critic"), invoke("case_revision", skills=["boundary"]), invoke("quality_critic")])
        self.assertEqual(run.status, "waiting_input")
        self.assertIn("未变化", run.question)
        self.assertIn("请检查修复产物", run.question)
        self.assertNotIn("请补充以下问题的业务依据", run.question)

    def test_explicit_budget_extension_preserves_run_and_guards(self):
        self.prepared(cases=True)
        run, _ = self.run_script([ask()], max_steps=1)
        planner = ScriptedPlanner([invoke("quality_critic"), ask()])
        supervisor = AgenticSupervisor(self.worker, planner)
        with self.assertRaisesRegex(ValueError, "explicitly add"):
            supervisor.resume(self.project.id, run.id, "修复后重新评审")
        self.assertEqual(planner.inputs, [])
        result = supervisor.resume(self.project.id, run.id, "修复后重新评审", additional_steps=3)
        self.assertEqual(result.id, run.id)
        self.assertEqual(result.max_steps, 4)
        self.assertEqual(result.budget_extensions, [3])
        self.assertEqual(result.steps[0], run.steps[0])
        self.assertEqual(result.steps[1].decision.capability, "quality_critic")
        self.assertEqual(self.store.get_agent_run(self.project.id, run.id).budget_extensions, [3])

    def test_exhausted_run_can_resume_with_explicit_budget(self):
        self.prepared(cases=True)
        run, _ = self.run_script([invoke("quality_critic")], max_steps=1)
        self.assertEqual(run.status, "budget_exhausted")
        result = AgenticSupervisor(self.worker, ScriptedPlanner([ask()])).resume(
            self.project.id, run.id, additional_steps=1)
        self.assertEqual(result.status, "waiting_input")
        self.assertEqual(len(result.steps), 2)

    def test_budget_validation_does_not_mutate_saved_run(self):
        run, _ = self.run_script([ask()], max_steps=1)
        supervisor = AgenticSupervisor(self.worker, ScriptedPlanner([]))
        for steps in [-1, 21, True, 1.5]:
            with self.assertRaises(ValueError):
                supervisor.resume(self.project.id, run.id, "继续", additional_steps=steps)
        run.max_steps = 100
        self.store.save_agent_run(run)
        with self.assertRaisesRegex(ValueError, "100"):
            supervisor.resume(self.project.id, run.id, "继续", additional_steps=1)
        self.assertEqual(self.store.get_agent_run(self.project.id, run.id).responses, [])

    def test_offline_pause_and_resume_after_reloading_run(self):
        run = AgenticSupervisor(self.worker).start(self.project.id, max_steps=2)
        self.assertEqual(run.mode, "deterministic")
        self.assertEqual(run.status, "budget_exhausted")
        self.assertFalse(self.store.get_project(self.project.id).cases)
        # Old confirmation pauses remain resumable without changing human provenance.
        run.status = "waiting_confirmation"
        self.store.save_agent_run(run)
        result = AgenticSupervisor(self.worker).resume(self.project.id, run.id, additional_steps=6)
        self.assertEqual(result.status, "completed")
        self.assertFalse(self.store.get_project(self.project.id).module_tree.confirmed)
        self.assertEqual(self.store.get_agent_run(self.project.id, run.id).status, "completed")
        self.assertLessEqual(len(result.steps), result.max_steps)

    def test_model_selects_skills_and_replans_from_actual_review_feedback(self):
        self.prepared()
        class FeedbackPlanner(ScriptedPlanner):
            def generate_json(planner, system, user, schema):
                data = json.loads(user)
                planner.inputs.append(data)
                state = data["state"]
                if not state["case_count"]:
                    return invoke("case_generation", skills=["permission"])
                if not data["fresh_review"]:
                    return invoke("quality_critic")
                if any(f["category"] in {"case_type", "module_coverage", "assertion", "requirement_coverage"} for f in state["review"]["findings"]):
                    return invoke("case_revision", skills=["boundary", "exception_recovery"])
                return finish()
        planner = FeedbackPlanner([])
        with patch.object(self.worker.case_agent, "run", wraps=self.worker.case_agent.run) as worker_run:
            run = AgenticSupervisor(self.worker, planner).start(self.project.id)
        self.assertEqual(run.status, "needs_attention")  # Real planner protocol with intentionally offline workers.
        names = [s.decision.capability for s in run.steps]
        self.assertNotIn("requirement_understanding", names)
        self.assertNotIn("module_planning", names)
        self.assertIn("case_revision", names)
        self.assertEqual(names.count("quality_critic"), 2)
        self.assertEqual(worker_run.call_args_list[0][0][1]["selected_skills"], ["permission"])
        self.assertEqual(worker_run.call_args_list[1][0][1]["selected_skills"], ["boundary", "exception_recovery"])
        self.assertTrue(any(o.get("review", {}).get("findings") for d in planner.inputs for o in [d["state"]] if o.get("review")))
        self.assertFalse(AgenticSupervisor._blocking(self.store.get_project(self.project.id)))

    def test_uploaded_materials_and_search_results_reach_worker_context(self):
        p = self.prepared()
        p.source_documents = [ParsedDocument(filename="contract.md", document_type="markdown", normalized_text="UNIQUE_MATERIAL_EVIDENCE")]
        self.store.save_project(p)
        search_result = {"ticket_type": "COMMON", "hits": [], "formatted_text": "UNIQUE_SEARCH_EVIDENCE"}
        with patch.object(self.worker, "search_knowledge", return_value=search_result), \
                patch.object(self.worker.case_agent, "run", wraps=self.worker.case_agent.run) as worker_run:
            run, planner = self.run_script([
                invoke("inspect_materials"), invoke("search_knowledge", instruction="权限规则"),
                invoke("case_generation", skills=["permission"]), ask(),
            ])
        self.assertEqual(run.status, "waiting_input")
        self.assertEqual(planner.inputs[0]["state"]["materials"][0]["type"], "markdown")
        knowledge = worker_run.call_args[0][1]["knowledge"]
        self.assertIn("UNIQUE_MATERIAL_EVIDENCE", knowledge)
        self.assertIn("UNIQUE_SEARCH_EVIDENCE", knowledge)

    def test_model_skill_selection_is_authoritative_in_live_generation_prompt(self):
        p = self.prepared(cases=True)
        class WorkerModel:
            enabled = True
            prompt = ""
            def generate_json(model, system, user, schema):
                model.prompt = user
                return {"cases": [c.model_dump() for c in p.cases]}
        llm = WorkerModel()
        CaseGenerationAgent(llm).run({"analysis": p.analysis.model_dump(), "module_tree": p.module_tree.model_dump()}, {"selected_skills": ["permission"]})
        section = llm.prompt.split("Testing skills:\n", 1)[1].split("\n\nAdopted", 1)[0]
        self.assertEqual(section, SKILLS["permission"].instruction)

    def test_invalid_decision_observation_drives_next_choice(self):
        run, planner = self.run_script([invoke("shell_execute"), ask()])
        self.assertEqual(run.steps[0].status, "rejected")
        self.assertIn("Unknown capability", planner.inputs[1]["observations"][0]["observation"]["error"])
        self.assertEqual(run.status, "waiting_input")

    def test_schema_errors_count_against_budget(self):
        run, _ = self.run_script([dict(action="invoke_agent", capability="requirement_understanding", reason="test", arbitrary=True)], max_steps=1)
        self.assertEqual(run.status, "budget_exhausted")
        self.assertIsNone(run.steps[0].decision)
        self.assertIsNone(self.store.get_project(self.project.id).analysis)

    def test_generation_does_not_require_human_confirmation(self):
        self.prepared(confirmed=False)
        run, _ = self.run_script([invoke("case_generation", skills=["happy_path"]), ask()])
        self.assertEqual(run.steps[0].status, "success")
        self.assertEqual(run.status, "waiting_input")
        self.assertTrue(self.store.get_project(self.project.id).cases)
        self.assertFalse(self.store.get_project(self.project.id).module_tree.confirmed)

    def test_module_only_target_finishes_without_cases_or_confirmation(self):
        run = AgenticSupervisor(self.worker).start(self.project.id, target="modules")
        self.assertEqual(run.status, "completed")
        self.assertEqual(self.store.get_agent_run(self.project.id, run.id).target, "modules")
        project = self.store.get_project(self.project.id)
        self.assertFalse(project.module_tree.confirmed)
        self.assertFalse(project.cases)
        from app.supervisor_models import SupervisorDecision
        with self.assertRaisesRegex(ValueError, "Module-only"):
            AgenticSupervisor(self.worker)._validate(SupervisorDecision.model_validate(
                invoke("case_generation", skills=["happy_path"])), project, run)

    def test_module_blocker_still_prevents_autonomous_completion(self):
        project = self.prepared(confirmed=False)
        project.module_review.findings = [ReviewFinding(severity="high", category="coverage", message="Missing core requirement")]
        self.store.save_project(project)
        run = AgenticSupervisor(self.worker).start(project.id, target="modules")
        self.assertEqual(run.status, "waiting_input")
        self.assertFalse(self.store.get_project(project.id).cases)
        from app.supervisor_models import SupervisorDecision
        with self.assertRaisesRegex(ValueError, "Blocking module"):
            AgenticSupervisor(self.worker)._validate(SupervisorDecision.model_validate(finish()), project, run)

    def test_existing_artifacts_and_unknown_skills_are_protected(self):
        p = self.prepared(cases=True)
        before = artifact_fingerprint(p)
        run, _ = self.run_script([
            invoke("requirement_understanding"), invoke("module_planning"),
            invoke("case_generation", skills=["happy_path"]),
            invoke("case_generation", mode="continue", skills=["invented_skill"]), ask(),
        ])
        self.assertTrue(all(s.status == "rejected" for s in run.steps[:4]))
        self.assertEqual(artifact_fingerprint(self.store.get_project(p.id)), before)

    def test_existing_review_cannot_allow_early_finish_or_revision(self):
        p = self.prepared(cases=True)
        self.worker.review(p)
        run, _ = self.run_script([finish(), invoke("case_revision", skills=["boundary"]), ask()])
        self.assertTrue(all(s.status == "rejected" for s in run.steps[:2]))
        self.assertIn("current artifacts", run.steps[0].observation["error"])

    def test_repeated_tool_call_is_blocked_even_if_reason_changes(self):
        second = invoke("inspect_materials")
        second["reason"] = "换一个理由重复调用"
        run, _ = self.run_script([invoke("inspect_materials"), second], max_steps=2)
        self.assertEqual(run.status, "budget_exhausted")
        self.assertEqual(run.steps[1].status, "rejected")
        self.assertIn("Repeated", run.steps[1].observation["error"])

    def test_planner_failure_is_visible_without_silent_offline_fallback(self):
        run, _ = self.run_script([LLMError("provider unavailable")])
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.mode, "model")
        self.assertIn("provider unavailable", run.error)
        self.assertIsNone(self.store.get_project(self.project.id).analysis)

    def test_review_output_limit_pauses_as_technical_failure_and_reuses_remaining_budget(self):
        self.prepared(cases=True)
        self.worker.llm.api_key = "test-no-network"
        planner = ScriptedPlanner([invoke("quality_critic"), invoke("quality_critic"), finish()])
        supervisor = AgenticSupervisor(self.worker, planner)
        with patch.object(self.worker.llm, "generate_json", side_effect=LLMError(
                "provider body contains secret-not-to-persist", code="output_limit")):
            run = supervisor.start(self.project.id, max_steps=8)
        self.assertEqual(run.status, "waiting_input")
        self.assertEqual(run.steps[-1].status, "error")
        self.assertEqual(run.steps[-1].observation["convergence_stop"], "review_failed")
        self.assertEqual(run.steps[-1].observation["error_code"], "output_limit")
        self.assertEqual(run.steps[-1].observation["remaining_steps"], 7)
        self.assertIn("finish_reason=length", run.error)
        self.assertIn("无需追加", run.question)
        self.assertNotIn("secret-not-to-persist", run.model_dump_json())
        self.assertEqual(run.review_fingerprint, "")
        self.assertFalse(run.review_history)
        with patch.object(self.worker, "review", side_effect=self.policy_review("suggestion", "low")):
            run = supervisor.resume(self.project.id, run.id, "已调整调用配置，重试评审")
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.max_steps, 8)
        self.assertEqual(len(run.steps), 3)
        self.assertFalse(run.error)
        self.assertFalse(run.budget_extensions)

    def test_user_response_is_given_to_replanner_and_analysis(self):
        planner = ScriptedPlanner([ask(), invoke("requirement_understanding"), ask()])
        supervisor = AgenticSupervisor(self.worker, planner)
        run = supervisor.start(self.project.id)
        with patch.object(self.worker.requirement_agent, "run", wraps=self.worker.requirement_agent.run) as worker_run:
            run = supervisor.resume(self.project.id, run.id, "CLARIFIED_ADMIN_ONLY")
        self.assertEqual(planner.inputs[1]["responses"], ["CLARIFIED_ADMIN_ONLY"])
        self.assertIn("CLARIFIED_ADMIN_ONLY", worker_run.call_args[0][0].context)
        self.assertEqual(run.status, "waiting_input")

    def test_revision_is_single_step_and_invalidates_review_with_version(self):
        p = self.prepared(cases=True)
        p.cases[0].steps[0].expected = ""
        self.worker.review(p)
        old_versions = len(p.case_versions)
        result = self.worker.revise_once(p, selected_skills=["happy_path"])
        self.assertIsNone(result.review)
        self.assertEqual(len(result.case_versions), old_versions + 1)
        self.assertTrue(result.cases[0].steps[0].expected)
        self.assertEqual(result.traces[-1].agent, "case_generation")

    def test_run_scope_and_lease(self):
        run = AgenticSupervisor(self.worker).start(self.project.id, max_steps=1)
        other = self.store.create_project("Other", REQUIREMENT)
        self.assertIsNone(self.store.get_agent_run(other.id, run.id))
        self.assertIsNone(self.store.get_agent_run(self.project.id, "../projects"))
        with self.store.project_lease(self.project.id):
            with self.assertRaisesRegex(ValueError, "already running"):
                AgenticSupervisor(self.worker).start(self.project.id)
            self.assertEqual(AgenticSupervisor(self.worker).start(other.id, max_steps=1).status, "budget_exhausted")

    def test_api_start_continue_ownership_and_manual_write_exclusion(self):
        from fastapi.testclient import TestClient
        import app.api as api
        with patch.object(api, "store", self.store), patch.object(api, "orchestrator", self.worker):
            client = TestClient(api.app)
            base = "/api/projects/" + self.project.id
            self.assertEqual(client.post(base + "/agent-runs", json={"max_steps": 21}).status_code, 422)
            response = client.post(base + "/agent-runs", json={})
            self.assertEqual(response.status_code, 200)
            run = response.json()
            self.assertEqual(run["status"], "completed")
            resume = base + "/agent-runs/" + run["id"] + "/continue"
            self.assertEqual(client.post(resume, json={"answer": "yes"}).status_code, 409)
            with self.store.project_lease(self.project.id):
                self.assertEqual(client.post(base + "/analyze").status_code, 409)
                self.assertEqual(client.post(base + "/agent-runs", json={}).status_code, 409)
                self.assertEqual(client.get(base + "/agent-runs").status_code, 200)
            p = self.store.get_project(self.project.id)
            self.assertEqual(client.put(base + "/modules/confirm", json={"modules": [m.model_dump() for m in p.module_tree.modules]}).status_code, 200)
            self.assertEqual(client.post(resume, json={}).status_code, 409)
            self.assertTrue(client.get(base + "/agent-runs/" + run["id"]).json()["steps"])
            other = self.store.create_project("Other", REQUIREMENT)
            self.assertEqual(client.get("/api/projects/" + other.id + "/agent-runs/" + run["id"]).status_code, 404)

    def test_api_budget_extension_validation_and_confirmation_gate(self):
        from fastapi.testclient import TestClient
        import app.api as api
        with patch.object(api, "store", self.store), patch.object(api, "orchestrator", self.worker):
            client = TestClient(api.app)
            base = "/api/projects/" + self.project.id
            run = client.post(base + "/agent-runs", json={"max_steps": 3}).json()
            self.assertEqual(run["status"], "budget_exhausted")
            path = base + "/agent-runs/" + run["id"] + "/continue"
            for amount in [-1, 21, True, 1.5]:
                self.assertEqual(client.post(path, json={"additional_steps": amount}).status_code, 422)
            self.assertEqual(client.post(path, json={}).status_code, 409)
            result = client.post(path, json={"additional_steps": 8})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "completed")
            self.assertEqual(result.json()["budget_extensions"], [8])


if __name__ == "__main__":
    unittest.main()
