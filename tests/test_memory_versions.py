import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from app.memory_versions import MemoryVersions, digest, stable, version_diff
from app.models import ModuleTree, TestCase, TestModule, TestStep, RequirementAnalysis
from app.orchestrator import TestCaseOrchestrator
from app.store import JsonStore
from app.supervisor import artifact_fingerprint


class MemoryVersionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = JsonStore(Path(self.temp.name))
        self.worker = TestCaseOrchestrator(self.store)
        self.worker.llm.api_key = ""
        self.memory = self.worker.adaptive_memory
        self.versions = MemoryVersions(self.store)
        self.project = self.store.create_project("Version test", "User can claim one coupon after login")
        self.project.analysis = RequirementAnalysis(summary="coupon")
        self.project.module_tree = ModuleTree(modules=[TestModule(id="M1", name="claim", objective="claim once")], confirmed=True)
        self.project.cases = [TestCase(id="C1", module_id="M1", title="old title", steps=[TestStep(action="claim", expected="one record")])]
        self.store.save_project(self.project)
        self.worker.record_feedback(self.project, "C1", "adopted")
        self.fact = self.memory.add_fact("limit one", project_id=self.project.id, fact_key="limit")["records"][0]

    def capture(self):
        return self.versions.create(self.project.id, "baseline")

    def fingerprint(self):
        return digest(stable(self.versions._capture(self.project.id)))

    def test_diff_uses_ids_and_ignores_access_telemetry(self):
        before = [{**self.fact.model_dump(), "content": "old", "access_count": 0}, {"id": "b", "content": "unchanged"}]
        after = [{"id": "b", "content": "unchanged"}, {**self.fact.model_dump(), "content": "new", "access_count": 7}]
        result = version_diff(before, after)
        self.assertEqual(result["summary"], {"added": 0, "removed": 0, "modified": 1})
        self.assertEqual(result["changes"][0]["path"], "/" + self.fact.id + "/content")

    def test_business_fields_named_like_telemetry_are_not_hidden(self):
        snapshot = self.capture()
        preview = self.versions.compare(self.project.id, "snapshot", snapshot["id"])
        self.project.cases[0].test_data = {"updated_at": "business timestamp", "access_count": 7}
        self.store.save_project(self.project)
        diff = self.versions.compare(self.project.id, "snapshot", snapshot["id"])
        self.assertIn("/project/cases/C1/test_data/updated_at", [v["path"] for v in diff["changes"]])
        with self.assertRaisesRegex(ValueError, "项目已变化"):
            self.versions.restore(self.project.id, snapshot["id"], preview["current_fingerprint"])

    def test_running_agent_blocks_restore_and_backup_can_undo_restore(self):
        from app.supervisor_models import SupervisorRun
        snapshot = self.capture()
        run = SupervisorRun(id="AR-" + "a" * 32, project_id=self.project.id, goal="test", mode="deterministic")
        self.store.save_agent_run(run)
        with self.assertRaisesRegex(ValueError, "运行中的"):
            self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        run.status = "waiting_input"
        self.store.save_agent_run(run)
        self.project.cases[0].title = "latest title"
        self.store.save_project(self.project)
        receipt = self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        self.versions.restore(self.project.id, receipt["backup_id"], self.fingerprint())
        self.assertEqual(self.store.get_project(self.project.id).cases[0].title, "latest title")
        self.assertEqual(self.store.get_agent_run(self.project.id, run.id).model_dump(), run.model_dump())

    def test_fact_rollback_creates_revision_and_preserves_chain(self):
        updated = self.memory.revise(self.fact.id, "limit two", "changed rule")
        original = next(m for m in self.store.list_memory_records(True) if m.id == self.fact.id).model_dump()
        rollback = self.memory.rollback(updated.id, self.fact.id, updated.id)
        self.assertEqual(rollback.content, "limit one")
        self.assertEqual(rollback.supersedes, updated.id)
        self.assertNotIn(rollback.id, {self.fact.id, updated.id})
        self.assertEqual(rollback.project_id, self.project.id)
        self.assertEqual(original, next(m for m in self.store.list_memory_records(True) if m.id == self.fact.id).model_dump())
        self.assertEqual(self.memory.history(self.fact.id)["current_id"], rollback.id)
        self.assertEqual(len(self.memory.history(rollback.id)["versions"]), 3)
        hits = self.memory.search("limit", project_id=self.project.id).hits
        self.assertEqual([h.memory.id for h in hits], [rollback.id])

    def test_rollback_rejects_stale_head_cross_scope_and_unrelated_fact(self):
        updated = self.memory.revise(self.fact.id, "limit two", "changed")
        unrelated = self.memory.add_fact("unrelated", project_id=self.project.id)["records"][0]
        foreign = self.memory.add_fact("foreign", project_id="other")["records"][0]
        before = self.store.adaptive_memory_file.read_bytes()
        for head, target, expected in ((self.fact.id, self.fact.id, self.fact.id),
                                       (updated.id, foreign.id, updated.id),
                                       (updated.id, unrelated.id, updated.id)):
            with self.assertRaises(ValueError):
                self.memory.rollback(head, target, expected)
        self.assertEqual(before, self.store.adaptive_memory_file.read_bytes())

    def test_rollback_does_not_revive_expired_business_validity(self):
        expired = self.memory.add_fact("expired", project_id=self.project.id, valid_to="2000-01-01T00:00:00Z")["records"][0]
        new = self.memory.revise(expired.id, "updated", "new contract")
        with self.assertRaisesRegex(ValueError, "有效期"):
            self.memory.rollback(new.id, expired.id, new.id)

    def test_rollback_preserves_business_expiry_and_rejects_unknown_legacy_expiry(self):
        target = self.memory.add_fact("temporary rule", project_id=self.project.id, valid_to="2999-01-01T00:00:00Z")["records"][0]
        new = self.memory.revise(target.id, "new rule", "changed")
        restored = self.memory.rollback(new.id, target.id, new.id)
        self.assertEqual(restored.valid_to, target.valid_to)
        records = self.store.list_memory_records(True)
        next(m for m in records if m.id == target.id).metadata.pop("revision_previous_valid_to")
        self.store.save_memory_records(records)
        with self.assertRaisesRegex(ValueError, "原有效期"):
            self.memory.rollback(restored.id, target.id, restored.id)

    def test_module_and_case_versions_share_diff_format(self):
        snapshot = self.capture()
        self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        restored = self.store.get_project(self.project.id)
        restored.module_tree.modules[0].objective = "new objective"
        restored.cases[0].title = "new title"
        self.store.save_project(restored)
        catalog = self.versions.catalog(self.project.id)
        for kind, expected in (("modules", "/module_tree/modules/M1/objective"), ("cases", "/cases/C1/title")):
            result = self.versions.compare(self.project.id, kind, catalog[kind][-1]["id"])
            self.assertIn(expected, [v["path"] for v in result["changes"]])

    def test_project_restore_restores_bundle_preserving_other_scope_and_rejection(self):
        snapshot = self.capture()
        other = self.store.create_project("other", "other requirement")
        other_fact = self.memory.add_fact("other fact", project_id=other.id)["records"][0]
        shared = self.memory.add_fact("shared fact")["records"][0]
        new = self.memory.revise(self.fact.id, "limit two", "changed")
        self.worker.record_feedback(self.project, "C1", "rejected")
        self.project.cases[0].title = "changed title"
        self.store.save_project(self.project)
        old_fingerprint = artifact_fingerprint(self.project)
        receipt = self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        restored = self.store.get_project(self.project.id)
        self.assertEqual(restored.cases[0].title, "old title")
        self.assertEqual(restored.cases[0].human_status, "rejected")
        self.assertFalse(restored.module_tree.confirmed)
        self.assertIsNone(restored.review)
        self.assertNotEqual(old_fingerprint, artifact_fingerprint(restored))
        records = {m.id: m for m in self.store.list_memory_records(True)}
        self.assertEqual(records[self.fact.id].status, "active")
        self.assertEqual(records[new.id].status, "inactive")
        self.assertEqual(records[other_fact.id].model_dump(), other_fact.model_dump())
        self.assertEqual(records[shared.id].model_dump(), shared.model_dump())
        self.assertFalse([d for d in self.store.project_knowledge(self.project.id) if d.doc_type == "case_example" and d.status == "active"])
        backup = self.versions._load(self.project.id, receipt["backup_id"])
        self.assertEqual(backup["bundle"]["project"]["cases"][0]["title"], "changed title")
        self.assertEqual(self.store.get_project(other.id).requirement, "other requirement")

    def test_preview_staleness_and_snapshot_integrity_are_checked_before_writes(self):
        snapshot = self.capture()
        preview = self.versions.compare(self.project.id, "snapshot", snapshot["id"])
        self.memory.revise(self.fact.id, "limit two", "changed")
        with self.assertRaisesRegex(ValueError, "项目已变化"):
            self.versions.restore(self.project.id, snapshot["id"], preview["current_fingerprint"])
        path = self.versions._path(self.project.id, snapshot["id"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["bundle"]["project"]["title"] = "tampered"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        self.assertEqual(len(list(path.parent.glob("MS-*.json"))), 1)

    def test_cross_project_snapshot_and_path_traversal_rejected(self):
        snapshot = self.capture()
        other = self.store.create_project("other", "other requirement")
        with self.assertRaises(ValueError):
            self.versions.restore(other.id, snapshot["id"], "a" * 64)
        with self.assertRaises(ValueError):
            self.versions.compare(self.project.id, "snapshot", "../../project")
        with self.assertRaises(ValueError):
            self.store.atomic_write({"../outside.json": {}})

    def test_write_failure_rolls_back_every_file_and_no_backup_is_published(self):
        snapshot = self.capture()
        self.memory.revise(self.fact.id, "limit two", "changed")
        paths = [self.store.projects_dir / (self.project.id + ".json"), self.store.adaptive_memory_file, self.store.knowledge_file]
        before = {path: path.read_bytes() for path in paths}
        writer = self.store._write_json_file
        def fail(path, payload):
            if path.name == "adaptive_memory.json":
                raise OSError("injected write failure")
            writer(path, payload)
        with patch.object(self.store, "_write_json_file", side_effect=fail):
            with self.assertRaises(OSError):
                self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        self.assertTrue(all(path.read_bytes() == body for path, body in before.items()))
        self.assertFalse(self.store._journal.exists())
        self.assertEqual(len(self.versions.catalog(self.project.id)["snapshots"]), 1)

    def test_interrupted_transaction_is_undone_on_new_store_open(self):
        class SimulatedCrash(BaseException):
            pass
        snapshot = self.capture()
        before = self.store.get_project(self.project.id).model_dump()
        writer = self.store._write_json_file
        def crash(path, payload):
            if path.name == "adaptive_memory.json":
                raise SimulatedCrash()
            writer(path, payload)
        with patch.object(self.store, "_write_json_file", side_effect=crash):
            with self.assertRaises(SimulatedCrash):
                self.versions.restore(self.project.id, snapshot["id"], self.fingerprint())
        self.assertTrue(self.store._journal.exists())
        reopened = JsonStore(self.store.root)
        self.assertFalse(reopened._journal.exists())
        self.assertEqual(reopened.get_project(self.project.id).model_dump(), before)

    def test_readers_cannot_observe_partial_transaction(self):
        snapshot = self.capture()
        reader_store = JsonStore(self.store.root)
        writer = self.store._write_json_file
        entered, release, reader_started = threading.Event(), threading.Event(), threading.Event()
        def delayed(path, payload):
            writer(path, payload)
            if path.name == self.project.id + ".json":
                entered.set()
                self.assertTrue(release.wait(5))
        def read():
            reader_started.set()
            return reader_store.get_project(self.project.id)
        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(self.store, "_write_json_file", side_effect=delayed):
            expected = self.fingerprint()
            write = pool.submit(self.versions.restore, self.project.id, snapshot["id"], expected)
            try:
                self.assertTrue(entered.wait(5))
                reader = pool.submit(read)
                self.assertTrue(reader_started.wait(5))
                self.assertFalse(reader.done())
            finally:
                release.set()
            write.result(timeout=5)
            self.assertFalse(reader.result(timeout=5).module_tree.confirmed)

    def test_api_workflow_previews_backs_up_and_restores_without_model(self):
        from fastapi.testclient import TestClient
        from app import api
        with patch.object(api, "store", self.store), patch.object(api, "orchestrator", self.worker), \
                patch.object(self.worker.llm, "generate_json", side_effect=AssertionError("unexpected model call")):
            client = TestClient(api.app)
            base = "/api/projects/" + self.project.id + "/memory-versions"
            snapshot = client.post(base, json={"label": "baseline"})
            self.assertEqual(snapshot.status_code, 200)
            sid = snapshot.json()["id"]
            current = self.memory.revise(self.fact.id, "limit two", "changed")
            history = client.get("/api/memory/" + current.id + "/history")
            self.assertEqual(history.json()["current_id"], current.id)
            diff = client.get("/api/memory/" + current.id + "/diff", params={"target_id": self.fact.id})
            self.assertGreater(diff.json()["summary"]["modified"], 0)
            rollback = client.post("/api/memory/" + current.id + "/rollback", json={"target_id": self.fact.id, "expected_current_id": current.id})
            self.assertEqual(rollback.status_code, 200)
            preview = client.post(base + "/diff", json={"kind": "snapshot", "left_id": sid})
            self.assertEqual(preview.status_code, 200)
            result = client.post(base + "/restore", json={"snapshot_id": sid, "expected_fingerprint": preview.json()["current_fingerprint"]})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertTrue(result.json()["backup_id"].startswith("MS-"))
            self.assertEqual(len(client.get(base).json()["snapshots"]), 2)
