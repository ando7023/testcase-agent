import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.agents import CaseReviewAgent, RequirementUnderstandingAgent
from app.benchmark_execution import BenchmarkExecution
from app.clarification_policy import validate_analysis
from app.llm import LLMError
from app.models import (AtomicRequirement, ClarificationItem, ModuleReviewFinding, ModuleReviewReport,
                        ModuleTree, RequirementAnalysis, RequirementInput, ReviewFinding, ReviewReport,
                        TestCase, TestModule)
from app.orchestrator import TestCaseOrchestrator
from app.review_policy import is_blocking, record_review
from app.store import JsonStore
from app.supervisor import AgenticSupervisor, artifact_fingerprint
from app.supervisor_models import SupervisorDecision, SupervisorRun

RAW = "Only registered subscribers shall be allowed to establish traces."


def gap(kind="execution_detail"):
    return ClarificationItem(id="G1", kind=kind, question="Which interface establishes a trace?",
                             requirement_ids=["R1"], affected_scenarios=["Attempt to establish a trace"],
                             source_quote=RAW, reason="The rejection outcome is known; interface setup is not specified.")


def analysis(items=None):
    return RequirementAnalysis(summary=RAW, atomic_requirements=[AtomicRequirement(id="R1", statement=RAW, source_quote=RAW)],
                               clarification_items=items or [])


def tree():
    return ModuleTree(confirmed=True, modules=[TestModule(id="M1", name="Authorization", objective=RAW, requirement_ids=["R1"])])


def case():
    return TestCase(id="C1", module_id="M1", title="Unregistered subscriber cannot establish a trace",
                    case_type="permission", requirement_ids=["R1"], source_evidence=[RAW],
                    preconditions=["A test subscriber with known unregistered status"],
                    steps=[{"action": "Attempt to establish a trace", "expected": "The trace is not established"}])


def finding(**changes):
    return {"severity": "medium", "case_id": "C1", "disposition": "clarification", "issue_type": "missing_contract",
            "requirement_ids": ["R1"], "evidence": RAW, "message": "Interface setup remains unspecified.",
            "clarification_kind": "execution_detail", "clarification_reason": "The behavior is known without the API address.", **changes}


class ClarificationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def review(self, findings, policy="evidence_only", fail=None):
        generate = Mock(side_effect=fail) if fail else Mock(return_value={"findings": findings})
        agent = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate))
        result = agent.run({"requirement": RAW, "analysis": analysis().model_dump(),
                            "module_tree": tree().model_dump(), "cases": [case().model_dump()]},
                           {"clarification_policy": policy})
        return ReviewReport.model_validate(result["review"]), result, generate

    def test_execution_detail_is_nonblocking_only_in_behavior_mode(self):
        report, result, generate = self.review([finding()])
        f = report.findings[0]
        self.assertFalse(is_blocking(f, "evidence_only"))
        self.assertTrue(is_blocking(f))
        self.assertEqual(result["cases"][0]["review_status"], "approved")
        self.assertIn("evidence_only", generate.call_args[0][0])
        self.assertFalse(any(f.category == "case_type" for f in report.findings))
        strict, _, _ = self.review([finding()], "strict")
        self.assertTrue(any(f.category == "case_type" for f in strict.findings))
        self.assertTrue(any(is_blocking(f) for f in strict.findings))

    def test_high_defects_and_critical_ambiguity_still_block(self):
        for item in [finding(severity="high"), finding(disposition="defect"), finding(clarification_kind="behavior_blocker")]:
            with self.subTest(item=item):
                report, result, _ = self.review([item])
                self.assertTrue(is_blocking(report.findings[0], "evidence_only"))
                self.assertEqual(result["cases"][0]["review_status"], "needs_attention")

    def test_review_checks_supplied_evidence_with_same_source_as_analysis(self):
        quote = "A valid subscriber manager is available."
        generate = Mock(return_value={"findings": [finding(evidence=quote)]})
        agent = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate))
        payload = {"requirement": RAW, "analysis": analysis().model_dump(),
                   "module_tree": tree().model_dump(), "cases": [case().model_dump()]}
        context = {"clarification_policy": "evidence_only",
                   "input_evidence": {"EBT-TEST": {"content": quote}}}
        result = agent.run(payload, context)
        report = ReviewReport.model_validate(result["review"])
        self.assertTrue(report.findings[0].clarification_basis_verified)
        self.assertFalse(is_blocking(report.findings[0], "evidence_only"))
        self.assertIn(quote, generate.call_args[0][1])
        result = agent.run(payload, {"clarification_policy": "evidence_only"})
        self.assertFalse(result["review"]["findings"][0]["clarification_basis_verified"])

    def test_unverified_or_legacy_clarifications_do_not_get_downgraded(self):
        for item in [finding(evidence="Invented quote"), finding(requirement_ids=["unknown"]),
                     finding(clarification_reason=""), finding(clarification_kind={}),
                     finding(clarification_kind="unspecified")]:
            report, _, _ = self.review([item])
            self.assertTrue(is_blocking(report.findings[0], "evidence_only"))

    def test_technical_review_failure_remains_incomplete(self):
        report, result, _ = self.review([], fail=LLMError("private", code="timeout"))
        self.assertTrue(any(f.category == "review_incomplete" and is_blocking(f, "evidence_only") for f in report.findings))
        self.assertEqual(result["cases"][0]["review_status"], "needs_attention")

    def test_analysis_rejects_fabricated_quotes_unknown_refs_and_unclassified_ambiguity(self):
        valid = analysis([gap()])
        validate_analysis(valid, RAW)
        for change in [lambda a: setattr(a.atomic_requirements[0], "source_quote", "Payment is mandatory"),
                       lambda a: setattr(a.clarification_items[0], "requirement_ids", ["unknown"]),
                       lambda a: setattr(a, "ambiguities", ["Should the outcome be allowed or denied?"]),
                       lambda a: setattr(a.clarification_items[0], "reason", " ")]:
            item = valid.model_copy(deep=True)
            change(item)
            with self.assertRaises(LLMError):
                validate_analysis(item, RAW)

    def prepared(self, items=None, confirmed=True):
        worker = TestCaseOrchestrator(JsonStore(self.root), knowledge_policy="sample_only")
        project = worker.store.create_project("Trace", RAW)
        project.clarification_policy = "evidence_only"
        project.analysis, project.module_tree, project.cases = analysis(items), tree(), [case()]
        project.module_tree.confirmed = confirmed
        project.review = ReviewReport(score=100)
        worker.store.save_project(project)
        run = SupervisorRun(id="AR-" + "a" * 32, project_id=project.id, goal="Design tests", mode="model",
                            clarification_policy="evidence_only", review_fingerprint=artifact_fingerprint(project))
        return worker, project, run, AgenticSupervisor(worker)

    def test_supervisor_rejects_pause_for_execution_details_but_allows_finish(self):
        _, project, run, controller = self.prepared([gap()])
        with self.assertRaisesRegex(ValueError, "do not justify pausing"):
            controller._validate(SupervisorDecision(action="request_input", reason="missing API", question="API address?"), project, run)
        controller._validate(SupervisorDecision(action="finish", reason="Behavior review complete"), project, run)
        run.max_steps = 2
        with self.assertRaisesRegex(ValueError, "do not justify pausing"):
            controller._validate(SupervisorDecision(action="request_input", reason="missing API", question="More budget?"), project, run)

    def test_confirmation_cannot_smuggle_business_answers(self):
        _, project, run, controller = self.prepared([gap()], confirmed=False)
        decision = SupervisorDecision(action="request_input", reason="Confirm", question="Confirm and provide age/payment/GDPR rules")
        with self.assertRaisesRegex(ValueError, "do not justify pausing"):
            controller._validate(decision, project, run)

    def test_actual_behavior_blocker_prevents_finish_even_with_perfect_review(self):
        item = gap("behavior_blocker")
        item.question, item.reason = "Allow or deny the same subscriber?", "Input rules give conflicting outcomes."
        _, project, run, controller = self.prepared([item])
        with self.assertRaisesRegex(ValueError, "Blocking"):
            controller._validate(SupervisorDecision(action="finish", reason="Perfect score"), project, run)
        decision = SupervisorDecision(action="request_input", reason="Conflict", question="Payment?")
        controller._validate(decision, project, run)
        self.assertIn(item.question, decision.question)
        self.assertNotIn("Payment", decision.question)

    def test_policy_persists_and_invalidates_review_fingerprint(self):
        worker, project, run, _ = self.prepared()
        self.assertEqual(worker.store.get_project(project.id).clarification_policy, "evidence_only")
        before = artifact_fingerprint(project)
        project.clarification_policy = "strict"
        self.assertNotEqual(before, artifact_fingerprint(project))
        report, _, _ = self.review([finding()])
        record_review(run, report, before)
        self.assertFalse(next(iter(run.issue_ledger.values()))["blocking"])

    def live_factory(self, items=None, module_failure=False):
        systems = []
        def factory(root):
            worker = TestCaseOrchestrator(JsonStore(root), knowledge_policy="sample_only")
            worker.llm.api_key, worker.llm.disabled = "test-placeholder", False
            worker.requirement_agent.research.run = Mock(return_value={})
            worker.case_agent.research.run = Mock(return_value={})
            def respond(system, user, schema=None):
                systems.append(system)
                if "CaseForge's Supervisor" in system:
                    snapshot = json.loads(user)["state"]
                    if snapshot["analysis"] is None:
                        name = "requirement_understanding"
                    elif snapshot["modules"] is None:
                        name = "module_planning"
                    elif snapshot["blocking_clarifications"] or (snapshot["module_review"] and any(
                            f["severity"] == "high" for f in snapshot["module_review"]["findings"])):
                        return {"action": "request_input", "reason": "Behavior conflict", "question": "Clarify conflicting outcomes"}
                    elif not snapshot["case_count"]:
                        name = "case_generation"
                    elif snapshot["review"] is None:
                        name = "quality_critic"
                    else:
                        return {"action": "finish", "reason": "Behavior-level review complete"}
                    return {"action": "invoke_agent", "capability": name, "reason": "Next supported step",
                            "skills": ["permission"] if name == "case_generation" else []}
                if "senior QA requirement analyst" in system:
                    return analysis(items).model_dump()
                if "interactive module-tree editor" in system:
                    return tree().model_dump()
                if "test architecture critic" in system:
                    return {"findings": [{"severity": "high", "module_id": "M1", "message": "Module invents payment requirements"}] if module_failure else []}
                if "meticulous senior test case writer" in system:
                    return {"cases": [case().model_dump()]}
                if "independent senior test case critic" in system:
                    return {"findings": [finding()]}
                raise AssertionError(system[:80])
            worker.llm.generate_json = respond
            return worker
        return factory, systems

    def test_behavior_pipeline_completes_but_does_not_claim_executable_readiness(self):
        for execution in ["workflow", "agentic"]:
            factory, systems = self.live_factory([gap()])
            runner = BenchmarkExecution(self.root / execution, factory, "live", execution, "simulate_confirm", 12,
                                        clarification_policy="evidence_only")
            result = runner.sample("102", lambda w: runner.pipeline(w, "Trace", RAW))
            self.assertTrue(result["flow_completed"], result)
            self.assertTrue(result["quality_passed"], result)
            self.assertFalse(result["technical_failure"], result)
            self.assertEqual(result["case_design_level"], "behavior")
            self.assertEqual(result["execution_readiness"], "needs_preparation")
            self.assertEqual(result["uncovered_requirement_ids"], [])
            self.assertTrue(all("evidence_only" in system for system in systems))

    def test_simulator_does_not_confirm_behavior_conflict_or_invalid_modules(self):
        for items, module_failure in [([gap("behavior_blocker")], False), ([], True)]:
            for execution in ["workflow", "agentic"]:
                factory, _ = self.live_factory(items, module_failure)
                runner = BenchmarkExecution(self.root / (execution + str(module_failure)), factory, "live", execution, "simulate_confirm", 12,
                                            clarification_policy="evidence_only")
                result = runner.sample("A", lambda w: runner.pipeline(w, "Trace", RAW))
                self.assertFalse(result["flow_completed"], result)
                self.assertIsNone(result["quality_passed"])
                self.assertEqual(result["simulated_confirmations"], [])
                self.assertEqual(result["status"], "waiting_input")

    def test_analysis_prompt_and_legacy_strict_default(self):
        generate = Mock(return_value=analysis([gap()]).model_dump())
        agent = RequirementUnderstandingAgent(SimpleNamespace(enabled=True, generate_json=generate))
        agent.run(RequirementInput(title="Trace", content=RAW), {"clarification_policy": "evidence_only"})
        self.assertIn("necessary condition", generate.call_args[0][0])
        agent.run(RequirementInput(title="Trace", content=RAW), {})
        self.assertNotIn("evidence_only", generate.call_args[0][0])

    def test_api_forwards_policy_on_both_transports_and_rejects_unknown_policy(self):
        from fastapi.testclient import TestClient
        from app import api
        client = TestClient(api.app)
        for path in ["/api/benchmarks/run", "/api/benchmarks/stream"]:
            with patch.object(api.orchestrator, "run_public_benchmark", return_value={}) as run:
                self.assertEqual(client.post(path, json={"suite": "ebt_generation", "clarification_policy": "evidence_only"}).status_code, 200)
                self.assertEqual(run.call_args[1]["clarification_policy"], "evidence_only")
            self.assertEqual(client.post(path, json={"suite": "ebt_generation", "clarification_policy": "ignore_all"}).status_code, 422)
