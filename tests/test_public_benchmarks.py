import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.benchmark_execution import BenchmarkExecution
from app.llm import LLMError
from app.models import ReviewFinding
from app.ebt_dataset import EBT_FILES, EBTRepository
from app.orchestrator import TestCaseOrchestrator
from app.public_benchmarks import (
    PublicBenchmarkService,
    SRSRepository,
    STORYSEEK_FILES,
    StorySeekRepository,
)
from app.store import JsonStore


class PublicBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    def _ebt(self):
        root = self.root / "external" / "ebt"
        root.mkdir(parents=True, exist_ok=True)
        for name in EBT_FILES:
            (root / name).write_text("", encoding="utf-8")
        self._write_csv(
            root / "artifacts.csv",
            ["id", "content", "layer", "summary"],
            [
                ["100", "A registered user shall submit a request and receive a visible result.", "Requirement", ""],
                ["141", "Test case: submit request. Preconditions user registered. Steps submit. Postconditions result is visible.", "Test", ""],
            ],
        )
        self._write_csv(
            root / "traces.csv",
            ["s_id", "t_id", "label"],
            [["141", "100", "1"]],
        )
        return EBTRepository(root)

    def test_storyseek_uses_project_level_splits(self):
        repository = StorySeekRepository(self.root / "storyseek")
        header = [
            "storyid", "background", "problems", "solutions", "goal",
            "actor", "impact", "deliverable", "us_actor", "us_action",
            "us_expected_outcome",
        ]
        for index, name in enumerate(STORYSEEK_FILES, 1):
            self._write_csv(
                repository.root / name,
                header,
                [[index, "Background", "Problem", "Solution", "Goal", "User", "Impact", "Feature", "User", "submit ticket", "ticket is created"]],
            )

        self.assertTrue(repository.ready)
        self.assertEqual(len(repository.load("development")), 6)
        self.assertEqual(len(repository.load("validation")), 2)
        self.assertEqual(len(repository.load("test")), 2)
        self.assertEqual(repository.status()["license"], "MIT")

    def test_srs_discovers_original_and_normalized_pairs(self):
        repository = SRSRepository(self.root / "srs")
        folder = repository.files_root / "project-a"
        folder.mkdir(parents=True)
        (folder / "spec.md").write_text("# Requirement\nUser submits a request.", encoding="utf-8")
        (folder / "spec_Raw.txt").write_text("Requirement User submits a request.", encoding="utf-8")
        (folder / "FunctionalRequirements.txt").write_text("User submits a request.", encoding="utf-8")

        pairs = repository.pairs()

        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["source_path"].endswith("spec.md"))
        self.assertTrue(pairs[0]["relevant_path"].endswith("FunctionalRequirements.txt"))

    def test_critic_and_ebt_generation_suites_persist_reports(self):
        ebt = self._ebt()
        service = PublicBenchmarkService(self.root, ebt)
        factory = lambda root: TestCaseOrchestrator(JsonStore(root))

        critic = service.run("critic_mutation", 4, "test", "offline", factory)
        generation = service.run("ebt_generation", 1, "test", "offline", factory, human_policy="simulate_confirm")

        self.assertEqual(critic["metrics"]["defect_detection_recall"], 1.0)
        self.assertEqual(critic["metrics"]["baseline_finding_count"], 0.0)
        self.assertEqual(critic["metrics"]["quality_pass_rate"], 1.0)
        self.assertEqual(critic["metrics"]["quality_applicable_count"], 4)
        self.assertEqual(generation["sample_count"], 1)
        self.assertEqual(generation["metrics"]["flow_completion_rate"], 1.0)
        self.assertIsNone(generation["score"])
        self.assertEqual(len(service.list_reports()), 2)

    def _run(self, **options):
        service = PublicBenchmarkService(self.root, self._ebt())
        return service.run("ebt_generation", 1, "test", "offline",
                           lambda root: TestCaseOrchestrator(JsonStore(root)), **options)

    def test_default_waits_for_confirmation(self):
        result = self._run()
        sample = result["samples"][0]
        self.assertEqual(sample["run_status"], "waiting_confirmation")
        self.assertFalse(sample["flow_completed"])
        self.assertIsNone(sample["quality_passed"])
        self.assertEqual(sample["generated_case_count"], 0)
        self.assertEqual(sample["simulated_confirmations"], [])
        self.assertEqual(result["status"], "incomplete")

    def test_agentic_runs_supervisor_and_explicit_confirmation(self):
        result = self._run(execution="agentic", human_policy="simulate_confirm")
        sample = result["samples"][0]
        self.assertTrue(sample["flow_completed"], sample)
        self.assertEqual(sample["run_status"], "completed")
        self.assertTrue(sample["run_id"].startswith("AR-"))
        self.assertIn("finish", [a["capability"] for a in sample["actions"]])
        self.assertEqual(sample["simulated_confirmations"],
                         [{"source": "benchmark_simulator", "action": "confirm_modules"}])
        self.assertEqual(sample["validation_level"], "offline_structural")
        workspace = self.root / result["workspace"] / sample["workspace"]
        self.assertNotIn('"human_gate"', "\n".join(p.read_text(encoding="utf-8") for p in workspace.rglob("*.json")))

    def test_agentic_budget_never_auto_extends(self):
        sample = self._run(execution="agentic", human_policy="simulate_confirm", max_steps=1)["samples"][0]
        self.assertFalse(sample["flow_completed"])
        self.assertEqual(sample["steps"], 1)
        self.assertEqual(sample["simulated_confirmations"], [])
        self.assertIsNone(sample["quality_passed"])

    def test_blocking_findings_are_not_success_even_with_high_score(self):
        original = TestCaseOrchestrator.review
        def blocked(worker, project):
            project = original(worker, project)
            project.review.score = 100
            project.review.findings = [ReviewFinding(severity="high", category="semantic", message="Confirmed defect")]
            worker.store.save_project(project)
            return project
        with patch.object(TestCaseOrchestrator, "review", blocked):
            sample = self._run(human_policy="simulate_confirm")["samples"][0]
        self.assertTrue(sample["flow_completed"])
        self.assertFalse(sample["quality_passed"])
        self.assertEqual(sample["status"], "quality_failed")
        self.assertFalse(sample["technical_failure"])

    def _live_factory(self, response=None, error=None):
        clients = []
        def factory(root):
            worker = TestCaseOrchestrator(JsonStore(root))
            worker.llm.api_key, worker.llm.disabled = "test-placeholder", False
            worker.llm.generate_json = Mock(return_value=response or {"findings": []}, side_effect=error)
            clients.append(worker.llm.generate_json)
            return worker
        return factory, clients

    def test_live_critic_actually_calls_model_and_isolates_each_sample(self):
        factory, calls = self._live_factory()
        service = PublicBenchmarkService(self.root, self._ebt())
        report = service.run("critic_mutation", 1, "test", "live", factory)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(c.called for c in calls))
        self.assertTrue(all(s["semantic_review_complete"] for s in report["samples"]))
        self.assertEqual(len({s["workspace"] for s in report["samples"]}), 5)
        self.assertEqual(report["metrics"]["technical_failure_rate"], 0)
        self.assertEqual(report["metrics"]["degraded_rate"], 0)

    def test_failed_model_review_cannot_pass_structural_mutations(self):
        factory, _ = self._live_factory(error=LLMError("private provider details", code="timeout"))
        service = PublicBenchmarkService(self.root, self._ebt())
        report = service.run("critic_mutation", 4, "test", "live", factory)
        self.assertEqual(report["metrics"]["technical_failure_rate"], 1)
        self.assertEqual(report["metrics"]["defect_detection_recall"], 0)
        self.assertTrue(all(s["quality_passed"] is None for s in report["samples"]))
        self.assertNotIn("private provider details", json.dumps(report))

    def test_live_mode_without_key_refuses_offline_fallback(self):
        def factory(root):
            worker = TestCaseOrchestrator(JsonStore(root))
            worker.llm.api_key = ""
            return worker
        service = PublicBenchmarkService(self.root, self._ebt())
        with self.assertRaisesRegex(ValueError, "不能静默"):
            service.run("critic_mutation", 1, "test", "live", factory)

    def test_fallback_is_separate_from_completed_and_passed(self):
        factory, _ = self._live_factory()
        runner = BenchmarkExecution(self.root / "runs", factory, "live", "workflow", "pause", 12)
        sample = runner.sample("a", lambda worker: {
            "flow_completed": True, "quality_passed": True, "_worker_modes": ["fallback"]})
        self.assertTrue(sample["flow_completed"])
        self.assertTrue(sample["degraded"])
        self.assertIsNone(sample["quality_passed"])
        self.assertEqual(sample["status"], "degraded")

    def test_sample_exception_does_not_abort_following_sample(self):
        factory, _ = self._live_factory()
        runner = BenchmarkExecution(self.root / "runs", factory, "offline", "workflow", "pause", 12)
        def fail(worker):
            raise RuntimeError("secret detail")
        first = runner.sample("../same", fail)
        second = runner.sample("../same", lambda worker: {"flow_completed": True})
        self.assertTrue(first["technical_failure"])
        self.assertNotIn("secret detail", json.dumps(first))
        self.assertTrue(second["flow_completed"])
        self.assertNotEqual(first["workspace"], second["workspace"])

    def test_component_rejects_agentic_before_model_call(self):
        factory = Mock()
        service = PublicBenchmarkService(self.root, self._ebt())
        with self.assertRaisesRegex(ValueError, "does not support"):
            service.run("critic_mutation", 1, "test", "live", factory, execution="agentic")
        factory.assert_not_called()

    def test_old_reports_are_preserved(self):
        service = PublicBenchmarkService(self.root, self._ebt())
        old = {"report_id": "BR-old", "score": 99, "samples": [{"passed": True}]}
        path = service.report_root / "BR-old.json"
        path.write_text(json.dumps(old), encoding="utf-8")
        self.assertEqual(service.list_reports()[0]["score"], 99)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), old)

    def test_storyseek_agentic_stops_at_module_checkpoint(self):
        self.test_storyseek_uses_project_level_splits()
        service = PublicBenchmarkService(self.root, self._ebt())
        service.storyseek = StorySeekRepository(self.root / "storyseek")
        result = service.run("storyseek_pipeline", 1, "test", "offline",
                             lambda root: TestCaseOrchestrator(JsonStore(root)), execution="agentic")
        sample = result["samples"][0]
        self.assertTrue(sample["flow_completed"], sample)
        self.assertEqual(sample["completion_criterion"], "modules_planned")
        self.assertEqual(sample["run_status"], "waiting_confirmation")
        self.assertNotIn("case_generator", [a["capability"] for a in sample["actions"]])
        self.assertEqual(sample["simulated_confirmations"], [])

    def test_srs_does_not_invent_quality_gate(self):
        self.test_srs_discovers_original_and_normalized_pairs()
        service = PublicBenchmarkService(self.root, self._ebt())
        service.srs = SRSRepository(self.root / "srs")
        result = service.run("srs_document", 1, "test", "offline",
                             lambda root: TestCaseOrchestrator(JsonStore(root)))
        self.assertEqual(result["metrics"]["flow_completion_rate"], 1)
        self.assertIsNone(result["metrics"]["quality_pass_rate"])
        self.assertEqual(result["metrics"]["quality_applicable_count"], 0)

    def test_empty_report_does_not_claim_success(self):
        report = PublicBenchmarkService._report("empty", [], {})
        self.assertEqual(report["status"], "empty")
        self.assertIsNone(report["score"])
        self.assertIsNone(report["metrics"]["quality_pass_rate"])

    def test_api_forwards_execution_options_and_rejects_invalid_budget(self):
        from app import api
        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        with patch.object(api.orchestrator, "run_public_benchmark", return_value={"schema_version": 2}) as run:
            response = client.post("/api/benchmarks/run", json={
                "suite": "ebt_generation", "execution": "agentic", "human_policy": "simulate_confirm", "max_steps": 7})
            self.assertEqual(response.status_code, 200)
            run.assert_called_once_with("ebt_generation", 3, "test", "offline",
                                        execution="agentic", human_policy="simulate_confirm", max_steps=7,
                                        llm_options={"stream": True, "reasoning_effort": "low", "timeout_seconds": 180})
            self.assertEqual(client.post("/api/benchmarks/run", json={
                "suite": "ebt_generation", "max_steps": 21}).status_code, 422)
        with patch.object(api.orchestrator, "run_public_benchmark", side_effect=ValueError("unsupported")):
            self.assertEqual(client.post("/api/benchmarks/run", json={"suite": "critic_mutation"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
