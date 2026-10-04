import io
import json
import tempfile
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents import CaseReviewAgent, ModuleCriticAgent
from app.benchmark_execution import BenchmarkExecution
from app.llm import LLMError, OpenAICompatibleClient
from app.models import KnowledgeDocument, ModuleTree, RequirementAnalysis, TestCase
from app.observability import TraceManager
from app.orchestrator import TestCaseOrchestrator
from app.semantic_review import review_cases
from app.store import JsonStore


def case(i):
    return TestCase(id=f"C{i}", module_id="M", title=f"Scenario {i}", source_evidence=["requirement"],
                    steps=[{"action": "submit", "expected": "record is created"}]).model_dump()


def finding(case_id="C0", **extra):
    return {"case_id": case_id, "severity": "high", "disposition": "defect",
            "issue_type": "requirement_conflict", "requirement_ids": [],
            "evidence": "record is created", "message": "Contradicts the required outcome", **extra}


class BenchmarkReliabilityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def factory(self, root):
        return TestCaseOrchestrator(JsonStore(root), knowledge_policy="sample_only")

    def test_sample_only_blocks_seed_retrieval_and_domain_tools(self):
        worker = self.factory(self.root)
        worker.llm.api_key = ""
        self.assertEqual(worker.store.project_knowledge(""), [])
        self.assertEqual(worker._retrieve("equipment_borrow subscriber").hits, [])
        self.assertFalse(worker._tool_get_domain_facts({"ticket_type": "EQUIPMENT_BORROW"})["available"])
        project = worker.store.create_project("Subscriber", "On registration a subscriber shall register itself under the control of a subscribermanager")
        project = worker.analyze(project)
        statements = " ".join(r.statement for r in project.analysis.atomic_requirements)
        self.assertNotIn("Borrow", statements)
        self.assertNotIn("设备", statements)
        self.assertEqual(len(project.analysis.atomic_requirements), 1)

    def test_sample_scoped_ebt_evidence_is_retrievable_without_global_knowledge(self):
        worker = self.factory(self.root)
        evidence = [
            {
                "id": "EBT-SAMPLE-TEST-143",
                "title": "EBT linked test artifact 143",
                "content": "Test case: subscriber registers with subscriber manager. Postconditions: subscriber is registered under the subscriber manager.",
                "doc_type": "benchmark_evidence",
                "source": "thearod5/ebt",
                "source_id": "EBT-RAG-V1-103",
                "metadata": {
                    "scope": "benchmark_sample",
                    "benchmark_id": "EBT-RAG-V1-103",
                    "artifact_id": "143",
                    "evidence_kind": "linked_test_example",
                    "ticket_type": "COMMON",
                },
            }
        ]
        worker.benchmark_evidence = [KnowledgeDocument.model_validate(item) for item in evidence]

        context = worker._retrieve("subscriber registers postconditions")

        self.assertTrue(context.hits)
        self.assertEqual(context.hits[0].document_id, "EBT-SAMPLE-TEST-143")
        self.assertEqual(worker.store.project_knowledge(""), [])

    def test_application_still_has_builtin_knowledge(self):
        worker = TestCaseOrchestrator(JsonStore(self.root))
        self.assertTrue(worker.store.project_knowledge(""))
        self.assertIn("interfaces", worker._tool_get_domain_facts({"ticket_type": "EQUIPMENT_BORROW"}))

    def test_benchmark_module_review_failure_is_not_swallowed(self):
        def fail(*args, **kwargs):
            raise LLMError("read timeout", code="timeout")
        client = SimpleNamespace(enabled=True, strict_review_failures=True, generate_json=fail)
        payload = {"analysis": RequirementAnalysis(summary="registration").model_dump(),
                   "module_tree": ModuleTree(modules=[]).model_dump()}
        with self.assertRaises(LLMError):
            ModuleCriticAgent(client).run(payload, {})

    def test_requested_settings_reach_http_body_and_diagnostics_report(self):
        def factory(root):
            worker = self.factory(root)
            worker.llm.api_key, worker.llm.disabled = "test-placeholder", False
            worker.llm.reasoning_effort = "max"
            worker.llm.stream_json = False
            return worker
        runner = BenchmarkExecution(self.root / "runs", factory, "live", "workflow", "pause", 12)
        response = b'data: {"choices":[{"delta":{"content":"{}"}}]}\ndata: [DONE]\n'
        def execute(worker):
            worker.llm.generate_json("Return JSON", "input")
            return {"flow_completed": True}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(response)) as request:
            sample = runner.sample("settings", execute)
        body = json.loads(request.call_args[0][0].data)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertTrue(body["stream"])
        self.assertEqual(request.call_args[1]["timeout"], 180)
        self.assertEqual(sample["call_diagnostics"][0]["phase"], "complete")
        self.assertIsNotNone(sample["call_diagnostics"][0]["first_event_ms"])

    def test_live_failure_stops_before_planning_and_records_config(self):
        workers = []
        def factory(root):
            worker = self.factory(root)
            worker.llm.api_key, worker.llm.disabled = "test-placeholder", False
            def fail(*args, **kwargs):
                raise LLMError("timed out", code="timeout")
            worker.requirement_agent.run = fail
            workers.append(worker)
            return worker
        runner = BenchmarkExecution(self.root / "runs", factory, "live", "workflow", "simulate_confirm", 12,
                                    {"stream": True, "reasoning_effort": "low", "timeout_seconds": 240})
        sample = runner.sample("104", lambda w: runner.pipeline(w, "Registration", "Register a subscriber"))
        self.assertTrue(sample["technical_failure"])
        self.assertEqual(sample["error_code"], "timeout")
        self.assertIsNone(sample["quality_passed"])
        project = workers[0].store.list_projects()[0]
        self.assertIsNone(project.module_tree)
        self.assertEqual(project.cases, [])
        self.assertEqual(sample["llm_config"]["timeout_seconds"], 240)
        self.assertEqual(sample["llm_config"]["reasoning_effort"], "low")
        self.assertEqual(sample["failure_agent"], "worker")
        self.assertEqual(sample["failure_phase"], "error")
        self.assertEqual(sample["failure_error_code"], "timeout")
        self.assertFalse(sample["degraded"])

        self.assertTrue(sample["llm_config"]["stream"])

    def test_batches_cover_all_cases_and_cross_findings_require_full_verification(self):
        cases, prompts = [case(i) for i in range(25)], []
        def generate(raw):
            data = json.loads(raw)
            prompts.append(data)
            if data["review_phase"] == "cross_batch_screen":
                return {"findings": [finding(related_case_ids=["C0", "C24"])]}
            if data["review_phase"] == "cross_batch_verify":
                self.assertEqual({c["id"] for c in data["cases"]}, {"C0", "C24"})
                self.assertEqual(data["cases"][1]["steps"], cases[24]["steps"])
                return {"findings": [finding()]}
            return {"findings": []}
        result = review_cases(generate, {"raw_requirement": "requirement"}, cases)
        local = [p for p in prompts if p["review_phase"] == "case_batch"]
        self.assertEqual([c["id"] for p in local for c in p["cases"]], [c["id"] for c in cases])
        self.assertLessEqual(max(len(p["cases"]) for p in local), 12)
        self.assertEqual(result["findings"], [finding()])

    def test_screen_candidate_rejected_by_full_review_is_not_published(self):
        def generate(raw):
            data = json.loads(raw)
            return {"findings": [finding(related_case_ids=["C0", "C12"])] if data["review_phase"] == "cross_batch_screen" else []}
        self.assertEqual(review_cases(generate, {}, [case(i) for i in range(13)])["findings"], [])

    def test_cross_screens_include_distant_groups(self):
        pairs = set()
        def generate(raw):
            data = json.loads(raw)
            if data["review_phase"] == "cross_batch_screen":
                ids = {c["id"] for c in data["case_cards"]}
                if {"C0", "C111"} <= ids:
                    pairs.add("distant")
            return {"findings": []}
        review_cases(generate, {}, [case(i) for i in range(112)])
        self.assertIn("distant", pairs)

    def test_later_batch_failure_never_approves_cases(self):
        calls = []
        def generate(system, raw, schema):
            data = json.loads(raw)
            calls.append(data)
            if data["review_phase"] == "case_batch" and data["batch_index"] == 1:
                raise LLMError("timeout", code="timeout")
            return {"findings": []}
        payload = {"analysis": RequirementAnalysis(summary="register").model_dump(),
                   "module_tree": ModuleTree(modules=[]).model_dump(), "cases": [case(i) for i in range(13)]}
        result = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(payload, {})
        self.assertTrue(any(f["category"] == "review_incomplete" and f["detail"] == "timeout" for f in result["review"]["findings"]))
        self.assertTrue(all(c["review_status"] == "needs_attention" for c in result["cases"]))
        self.assertEqual(len(calls), 2)

    def test_oversized_case_fails_without_silent_truncation(self):
        large = case(0)
        large["steps"][0]["action"] = "x" * 25000
        def forbidden(raw):
            self.fail("must validate budget before calling model")
        with self.assertRaises(LLMError) as caught:
            review_cases(forbidden, {}, [large])
        self.assertEqual(caught.exception.code, "context_limit")

    def test_cross_screen_failure_blocks_completion(self):
        def generate(raw):
            if json.loads(raw)["review_phase"] == "cross_batch_screen":
                raise LLMError("timeout", code="timeout")
            return {"findings": []}
        with self.assertRaises(LLMError):
            review_cases(generate, {}, [case(i) for i in range(13)])

    def test_findings_are_not_truncated_to_twenty_across_batches(self):
        def generate(raw):
            data = json.loads(raw)
            return {"findings": [finding(c["id"]) for c in data["cases"]] if data["review_phase"] == "case_batch" else []}
        self.assertEqual(len(review_cases(generate, {}, [case(i) for i in range(25)])["findings"]), 25)


class TimeoutDiagnosticsTest(unittest.TestCase):
    def client(self):
        client = OpenAICompatibleClient()
        client.api_key, client.disabled = "test-placeholder", False
        return client

    def test_connect_timeouts_are_typed_in_both_modes_including_urlerror(self):
        for stream in (False, True):
            for error in (TimeoutError("timeout"), urllib.error.URLError(TimeoutError("timeout"))):
                client = self.client()
                client.stream_json = stream
                with patch("urllib.request.urlopen", side_effect=error), self.assertRaises(LLMError) as caught:
                    client.generate_json("system", "user")
                self.assertEqual(caught.exception.code, "timeout")
                self.assertEqual(client.last_call_diagnostics["phase"], "awaiting_headers")
                self.assertIsNone(client.last_call_diagnostics["connected_ms"])

    def test_stream_timeout_preserves_partial_progress_in_trace_without_reasoning(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}\n'
                raise TimeoutError("read")
        with tempfile.TemporaryDirectory() as folder:
            client = self.client()
            client.stream_json = True
            client.tracer = TraceManager(Path(folder))
            with client.tracer.run("test") as run:
                with patch("urllib.request.urlopen", return_value=Response()), self.assertRaises(LLMError):
                    client.generate_json("system", "user")
            diagnostic = client.last_call_diagnostics
            self.assertIsNotNone(diagnostic["connected_ms"])
            self.assertIsNotNone(diagnostic["first_event_ms"])
            self.assertIsNotNone(diagnostic["last_receive_ms"])
            self.assertIsNone(diagnostic["first_content_ms"])
            self.assertEqual(diagnostic["event_count"], 1)
            self.assertEqual(diagnostic["error_code"], "timeout")
            saved = (Path(folder) / "traces" / (run.trace_id + ".json")).read_text(encoding="utf-8")
            self.assertNotIn("private reasoning", saved)
            self.assertIn('"first_event_ms"', saved)

    def test_nonstream_body_timeout_reports_headers_and_last_chunk(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, size): raise TimeoutError("read")
            def read1(self, size):
                if not getattr(self, "sent", False):
                    self.sent = True
                    return b'{"choices":'
                raise TimeoutError("read")
        client = self.client()
        client.stream_json = False
        with patch("urllib.request.urlopen", return_value=Response()), self.assertRaises(LLMError) as caught:
            client.generate_json("system", "user")
        self.assertEqual(caught.exception.code, "timeout")
        self.assertIsNotNone(client.last_call_diagnostics["connected_ms"])
        self.assertIsNotNone(client.last_call_diagnostics["last_receive_ms"])
        self.assertIsNone(client.last_call_diagnostics["first_event_ms"])

    def test_success_diagnostics_reset_after_a_failure(self):
        client = self.client()
        client.stream_json = True
        with patch("urllib.request.urlopen", side_effect=TimeoutError()), self.assertRaises(LLMError):
            client.generate_json("system", "user")
        response = b'data: {"choices":[{"delta":{"content":"{}"}}]}\ndata: [DONE]\n'
        with patch("urllib.request.urlopen", return_value=io.BytesIO(response)):
            self.assertEqual(client.generate_json("system", "user"), {})
        self.assertEqual(client.last_call_diagnostics["phase"], "complete")
        self.assertNotIn("error_code", client.last_call_diagnostics)
        self.assertIsNotNone(client.last_call_diagnostics["first_content_ms"])

    def test_shared_client_diagnostics_do_not_mix_concurrent_requests(self):
        barrier = Barrier(2)
        class Response:
            def __init__(self, fail): self.fail = fail
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                barrier.wait(timeout=5)
                if self.fail:
                    raise TimeoutError("read")
                yield b'data: {"choices":[{"delta":{"content":"{}"}}]}\n'
                yield b'data: [DONE]\n'
        client = self.client()
        client.stream_json = True
        def respond(request, **kwargs):
            return Response(json.loads(request.data)["messages"][1]["content"] == "fail")
        def run(user):
            try:
                client.generate_json("system", user)
            except LLMError:
                pass
            return dict(client.last_call_diagnostics)
        with patch("urllib.request.urlopen", side_effect=respond), ThreadPoolExecutor(max_workers=2) as pool:
            failed, succeeded = list(pool.map(run, ["fail", "ok"]))
        self.assertEqual(failed["error_code"], "timeout")
        self.assertIsNone(failed["first_event_ms"])
        self.assertNotIn("error_code", succeeded)
        self.assertEqual(succeeded["phase"], "complete")
