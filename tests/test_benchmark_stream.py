import asyncio
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.benchmark_execution import BenchmarkExecution
from app.benchmark_stream import EventBridge, benchmark_stream
from app.orchestrator import TestCaseOrchestrator
from app.store import JsonStore


class BenchmarkStreamTests(unittest.TestCase):
    def test_first_event_arrives_before_job_finishes(self):
        release, finished = threading.Event(), threading.Event()

        def job(emit):
            emit({"event": "delta", "text": "已开始"})
            release.wait(3)
            finished.set()
            return {"id": "test"}

        async def check():
            response = benchmark_stream(job)
            try:
                first = await asyncio.wait_for(response.body_iterator.__anext__(), 1)
                second = await asyncio.wait_for(response.body_iterator.__anext__(), 1)
                self.assertIn('"started"', first)
                self.assertIn("已开始", second)
                self.assertFalse(finished.is_set())
                release.set()
                rest = "".join([item async for item in response.body_iterator])
                self.assertIn('"finished"', rest)
            finally:
                release.set()
                await response.body_iterator.aclose()
        asyncio.run(check())

    def test_disconnect_unblocks_full_queue_without_cancelling_job(self):
        bridge = EventBridge()
        for i in range(bridge.queue.maxsize):
            bridge.emit({"event": "delta", "text": str(i)})

        async def detach():
            events = bridge.events()
            await events.__anext__()
            bridge.emit({"event": "delta"})
            producer = threading.Thread(target=lambda: bridge.emit({"event": "finished"}))
            producer.start()
            await events.aclose()
            producer.join(1)
            self.assertFalse(producer.is_alive())
            self.assertTrue(bridge.detached.is_set())
        asyncio.run(detach())

    def test_worker_failure_is_terminal_and_does_not_expose_exception(self):
        def job(emit):
            raise ValueError("credential-should-not-appear")
        async def consume():
            return "".join([item async for item in benchmark_stream(job).body_iterator])
        output = asyncio.run(consume())
        self.assertIn('"error"', output)
        self.assertNotIn("credential-should-not-appear", output)
        self.assertNotIn('"finished"', output)

    def test_busy_rejects_before_starting_new_job(self):
        with patch("app.benchmark_stream._slots") as slots:
            slots.acquire.return_value = False
            with self.assertRaises(HTTPException) as caught:
                benchmark_stream(lambda emit: self.fail("must not run"))
            self.assertEqual(caught.exception.status_code, 429)

    def test_api_stream_and_sync_keep_matching_options(self):
        from app import api
        received = []

        def run(*args, **kwargs):
            received.append(kwargs)
            if kwargs.get("on_event"):
                kwargs["on_event"]({"event": "sample_start", "sample_id": "A", "sample_index": 1})
            return {"id": "BR-test", "samples": []}

        client = TestClient(api.app)
        payload = {"suite": "ebt_generation", "mode": "offline", "stream": True,
                   "reasoning_effort": "low", "timeout_seconds": 240}
        with patch.object(api.orchestrator, "run_public_benchmark", side_effect=run):
            response = client.post("/api/benchmarks/stream", json=payload)
            synchronous = client.post("/api/benchmarks/run", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        frames = [json.loads(f[6:]) for f in response.text.strip().split("\n\n")]
        self.assertEqual([f["event"] for f in frames], ["started", "sample_start", "finished"])
        self.assertEqual(frames[-1]["report"], synchronous.json())
        self.assertEqual(received[0]["llm_options"], received[1]["llm_options"])
        self.assertEqual(client.post("/api/benchmarks/stream", json=dict(payload, timeout_seconds=0)).status_code, 422)

    def run_model(self, body, stream=True):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        events, workers = [], []
        def factory(path):
            worker = TestCaseOrchestrator(JsonStore(path), knowledge_policy="sample_only")
            worker.llm.api_key, worker.llm.disabled = "test-placeholder", False
            workers.append(worker)
            return worker
        runner = BenchmarkExecution(root, factory, "live", "workflow", "pause", 12,
                                    {"stream": stream}, events.append)
        def operation(worker):
            worker.llm.generate_json("Return JSON", "Synthetic requirement")
            return {"flow_completed": True}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
            sample = runner.sample("A", operation)
        return events, sample, workers[0]

    def test_stream_only_emits_content_and_keeps_failed_partial_unaccepted(self):
        body = (b'data: {"choices":[{"delta":{"reasoning_content":"private-thought"}}]}\n'
                b'data: {"choices":[{"delta":{"content":"{bad json"}}]}\n'
                b'data: [DONE]\n')
        events, sample, worker = self.run_model(body)
        self.assertNotIn("private-thought", json.dumps(events))
        self.assertEqual("".join(e["text"] for e in events if e["event"] == "delta"), "{bad json")
        self.assertEqual(next(e for e in events if e["event"] == "llm_end")["status"], "error")
        self.assertTrue(sample["technical_failure"])
        self.assertIsNone(sample["quality_passed"])
        self.assertIsNone(worker.llm.on_content_delta)
        self.assertIsNone(worker.llm.on_stream_progress)
        self.assertEqual(events[-1]["event"], "sample_end")

    def test_nonstream_body_is_one_content_event_without_token_progress(self):
        body = json.dumps({"choices": [{"message": {"content": '{"summary":"hello"}'}}]}).encode()
        events, sample, _ = self.run_model(body, stream=False)
        self.assertEqual([e["text"] for e in events if e["event"] == "delta"], ['{"summary":"hello"}'])
        self.assertFalse(next(e for e in events if e["event"] == "llm_start")["stream"])
        self.assertFalse(sample["technical_failure"])
        self.assertFalse(any(e["event"] == "llm_progress" for e in events))

    def test_separate_runs_keep_distinct_event_sinks(self):
        async def collect(tag):
            def job(emit):
                emit({"event": "delta", "text": tag})
                return {"tag": tag}
            return "".join([item async for item in benchmark_stream(job).body_iterator])
        async def check():
            first, second = await asyncio.gather(collect("unique-first"), collect("unique-second"))
            self.assertNotIn("unique-second", first)
            self.assertNotIn("unique-first", second)
        asyncio.run(check())
