import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.case_memory import CaseConversationMemory
from app.module_memory import ModuleConversationMemory
from app.models import ModuleTree, ProjectState
from app.orchestrator import TestCaseOrchestrator
from app.store import JsonStore


class MemoryV2Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JsonStore(Path(self.temp.name))
        self.pipeline = TestCaseOrchestrator(self.store)
        self.pipeline.llm.api_key = ""
        self.memory = self.pipeline.adaptive_memory

    def tearDown(self):
        self.temp.cleanup()

    def test_scoped_manual_rule_never_creates_global_legacy_copy(self):
        self.pipeline.add_memory_rule("Payment audit must be retained", user_id="alice", project_id="A")
        self.memory.migrate_legacy()
        restarted = TestCaseOrchestrator(self.store)
        restarted.llm.api_key = ""
        result = restarted.adaptive_memory.search("Payment audit", user_id="bob", project_id="B")
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(len(self.store.list_memory_records()), 1)
        self.assertFalse(self.store.get_memory().get("scoped_rules"))

    def test_migration_quarantines_widened_copies_and_retains_audit(self):
        self.memory.add_fact("Private audit rule", project_id="A")
        global_copy = self.memory.add_fact("Private audit rule", source="legacy_memory")["records"][0]
        self.store.add_memory_rule("Private audit rule")
        self.store.add_memory_rule("Shared retry rule")
        result = self.memory.migrate_legacy()
        self.assertEqual(result["quarantined"], 1)
        records = {item.id: item for item in self.store.list_memory_records(True)}
        self.assertEqual(records[global_copy.id].invalidation_reason, "legacy_scope_ambiguous")
        self.assertEqual(records[global_copy.id].status, "inactive")
        visible = self.memory.search("audit", project_id="B")
        self.assertFalse(any(hit.memory.content == "Private audit rule" for hit in visible.hits))
        self.assertEqual(self.memory.migrate_legacy()["migrated"], 0)

    def test_legacy_revocation_is_not_resurrected_on_restart_or_new_migration(self):
        self.store.add_memory_rule("Old retry rule")
        self.memory.migrate_legacy()
        record = self.store.list_memory_records()[0]
        self.memory.invalidate(record.id, "obsolete")
        self.store.add_memory_rule("New unrelated rule")
        self.memory.migrate_legacy()
        self.assertFalse(any(item.content == "Old retry rule" for item in self.store.list_memory_records()))

    def test_migrates_old_conversation_run_scope_but_preserves_real_run_scope(self):
        self.memory.remember("Keep payment cases separate", project_id="A", run_id="CV-old",
                             source="case_conversation", source_ref="CV-old", infer=False)
        self.memory.add_fact("Only this execution", project_id="A", run_id="RUN-private")
        # Simulate an installation predating migration v2.
        self.store._write_json(self.store.memory_migration_file, {})
        self.memory.migrate_legacy()
        hits = self.memory.search("payment execution", project_id="A").hits
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].memory.source_version_id, "CV-old")
        self.assertEqual(hits[0].memory.run_id, "")
        self.assertFalse(self.memory.search("payment", project_id="B").hits)

    def test_new_provenance_does_not_restrict_later_rounds(self):
        self.memory.remember("Keep callback retry cases separate", project_id="A",
                             agent_id="case_generation", source_run_id="RUN-1",
                             source_version_id="CV-1", infer=False)
        hit = self.memory.search("callback retry", project_id="A", agent_id="case_generation").hits[0]
        self.assertEqual(hit.memory.source_run_id, "RUN-1")
        self.assertEqual(hit.memory.source_version_id, "CV-1")
        self.assertFalse(self.memory.search("callback", project_id="A", agent_id="module_planning").hits)

    def test_revision_preserves_scope_history_and_is_idempotent(self):
        old = self.memory.add_fact("Retry at most 3 times", project_id="A", fact_key="retry_limit")["records"][0]
        new = self.memory.revise(old.id, "Retry at most 5 times", "Updated requirement")
        self.assertEqual(new.project_id, "A")
        self.assertEqual(new.supersedes, old.id)
        self.assertEqual(self.memory.revise(old.id, new.content, "retry").id, new.id)
        with self.assertRaises(ValueError):
            self.memory.revise(old.id, "Retry 9 times", "stale edit")
        records = {item.id: item for item in self.store.list_memory_records(True)}
        self.assertEqual(records[old.id].status, "superseded")
        self.assertEqual(records[old.id].superseded_by, new.id)
        self.assertEqual([hit.memory.id for hit in self.memory.search("retry", project_id="A").hits], [new.id])
        self.assertFalse(self.memory.search("retry", project_id="B").hits)
        self.memory.invalidate(new.id, "No longer relevant")
        self.memory.invalidate(new.id, "request retry")
        self.assertFalse(self.memory.search("retry", project_id="A").hits)

    def test_conflicting_key_is_pending_until_explicit_revision(self):
        old = self.memory.add_fact("Retry 3 times", fact_key="retry")["records"][0]
        conflict = self.memory.add_fact("Retry 7 times", fact_key="retry")["records"][0]
        self.assertEqual(conflict.status, "pending_conflict")
        self.assertEqual([h.memory.id for h in self.memory.search("retry").hits], [old.id])
        updated = self.memory.revise(old.id, "Retry 7 times", "Confirmed new limit")
        self.assertEqual([h.memory.id for h in self.memory.search("retry").hits], [updated.id])

    def test_expired_and_future_memories_are_not_retrieved(self):
        self.memory.add_fact("Expired retry rule", valid_to="2000-01-01T00:00:00Z")
        self.memory.add_fact("Future retry rule", valid_from="2999-01-01T00:00:00Z")
        self.assertFalse(self.memory.search("retry").hits)
        with self.assertRaises(ValueError):
            self.memory.add_fact("Invalid", valid_from="not-a-date")
        with self.assertRaises(ValueError):
            self.memory.add_fact("Invalid", valid_from="2026-01-01T00:00:00")

    def test_concurrent_revisions_cannot_both_replace_same_fact(self):
        old = self.memory.add_fact("Retry 3 times")["records"][0]
        def revise(value):
            try:
                return self.memory.revise(old.id, value, "concurrent edit").id
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(revise, ["Retry 5 times", "Retry 7 times"]))
        self.assertEqual(sum(value is not None for value in ids), 1)
        self.assertEqual(len(self.store.list_memory_records()), 1)

    def test_decisions_survive_long_conversations_and_reload_for_both_workspaces(self):
        for kind, memory in [("module", ModuleConversationMemory()), ("case", CaseConversationMemory())]:
            with self.subTest(kind=kind):
                project = self.store.create_project(kind, "Requirement to retain audit records")
                first = memory.remember_user(project, "Always retain all audit records", "chat")
                for i in range(45):
                    memory.remember_user(project, "Change item {}".format(i), "chat")
                self.store.save_project(project)
                project = self.store.get_project(project.id)
                context = memory.context(project)
                self.assertIn("Always retain all audit records", context)
                self.assertIn(first.id, context)
                count = len(getattr(project, kind + "_decision_state").entries)
                memory.context(project)
                self.assertEqual(len(getattr(project, kind + "_decision_state").entries), count)

    def test_explicit_decision_replacement_and_revocation(self):
        project = self.store.create_project("decision", "Requirement decision test")
        memory = ModuleConversationMemory()
        memory.remember_user(project, "约束[retry]: 最多重试3次", "chat")
        memory.remember_user(project, "约束[retry]: 最多重试5次", "chat")
        self.assertNotIn("最多重试3次", project.module_memory_summary)
        self.assertIn("最多重试5次", project.module_memory_summary)
        memory.remember_user(project, "撤销[retry]", "chat")
        self.assertNotIn("最多重试5次", project.module_memory_summary)
        self.assertEqual([e.status for e in project.module_decision_state.entries],
                         ["superseded", "revoked", "revoked"])

    def test_snapshot_restore_excludes_future_decisions_and_recent_turns(self):
        for kind, memory, result in [("module", ModuleConversationMemory(), ModuleTree(modules=[])),
                                     ("case", CaseConversationMemory(), [])]:
            with self.subTest(kind=kind):
                project = self.store.create_project(kind, "Requirement snapshot test")
                memory.remember_user(project, "Always keep audit logs", "chat")
                version = memory.remember_result(project, result, "chat", "audit logs")
                memory.remember_user(project, "FUTURE_ONLY change", "chat")
                memory.remember_result(project, result, "chat", "later")
                memory.restore_state(project, version)
                context = memory.context(project)
                self.assertIn("Always keep audit logs", context)
                self.assertNotIn("FUTURE_ONLY", context)
                memory.remember_user(project, "New branch decision", "chat")
                self.assertIn("New branch decision", memory.context(project))

    def test_old_project_log_rebuilds_structured_state(self):
        project = ProjectState(id="old", title="old", requirement="Old project requirement",
                               module_conversation=[dict(id="M1", role="user", content="Must retain audit logs")])
        self.assertIn("Must retain audit logs", ModuleConversationMemory().context(project))

    def test_restore_reconciles_long_term_branch_memories(self):
        project = self.store.create_project("restore", "Requirement snapshot test")
        memory = ModuleConversationMemory()
        memory.remember_user(project, "Always keep audit logs", "chat")
        v1 = memory.remember_result(project, ModuleTree(modules=[]), "chat", "audit")
        self.memory.remember("Keep audit logs", project_id=project.id, source="module_conversation",
                             source_version_id=v1.id, infer=False)
        memory.remember_user(project, "FUTURE_ONLY change", "chat")
        v2 = memory.remember_result(project, ModuleTree(modules=[]), "chat", "future")
        self.memory.remember("FUTURE_ONLY change", project_id=project.id, source="module_conversation",
                             source_version_id=v2.id, infer=False)
        memory.restore_state(project, v1)
        self.pipeline._restore_conversation_memories(project, "module_conversation", v1)
        self.assertFalse(any("FUTURE_ONLY" in h.memory.content
                             for h in self.memory.search("change", project_id=project.id).hits))
        memory.restore_state(project, v2)
        self.pipeline._restore_conversation_memories(project, "module_conversation", v2)
        self.assertTrue(any("FUTURE_ONLY" in h.memory.content
                            for h in self.memory.search("change", project_id=project.id).hits))

    def test_memory_lifecycle_api(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        import app.api as api
        with patch.object(api, "store", self.store), patch.object(api, "orchestrator", self.pipeline):
            client = TestClient(api.app)
            created = client.post("/api/memory/rules", json={
                "rule": "Retry 3 times", "project_id": "A", "fact_key": "retry"})
            self.assertEqual(created.status_code, 200)
            old = created.json()["records"][0]
            revised = client.post("/api/memory/{}/revise".format(old["id"]),
                                  json={"content": "Retry 5 times", "reason": "Updated rule"})
            self.assertEqual(revised.status_code, 200)
            self.assertEqual(revised.json()["project_id"], "A")
            self.assertEqual(client.post("/api/memory/{}/invalidate".format(revised.json()["id"]),
                                         json={"reason": "Removed requirement"}).status_code, 200)
            self.assertEqual(len(client.get("/api/memory?include_inactive=true").json()["records"]), 2)
            self.assertEqual(client.get("/api/memory").json()["records"], [])
            self.assertEqual(client.post("/api/memory/rules", json={
                "rule": "Invalid date", "valid_to": "invalid"}).status_code, 422)

    def test_module_operations_sync_repeated_updates_and_revocation_to_long_term_memory(self):
        project = self.store.create_project(
            "Retry testing", "Users submit payment requests. Failed requests can be retried. Admins inspect audit logs."
        )
        project = self.pipeline.plan_modules(self.pipeline.analyze(project))
        module_id = project.module_tree.modules[0].id
        for content in ("约束[retry_limit]: 必须验证最多重试3次", "约束[retry_limit]: 必须验证最多重试5次",
                        "约束[retry_limit]: 必须验证最多重试5次"):
            project = self.pipeline.operate_modules(project, "chat", content, module_id)
            facts = [item for item in self.store.list_memory_records()
                     if item.project_id == project.id and item.source == "module_conversation"]
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].content, content.split(": ", 1)[1])
            self.assertFalse(facts[0].run_id)
            self.assertEqual(facts[0].source_version_id, project.module_versions[-1].id)
            recalled = self.memory.search("重试", project_id=project.id, agent_id="module_planning",
                                          ticket_type=(project.analysis.ticket_types or ["COMMON"])[0])
            self.assertIn(facts[0].id, [hit.memory.id for hit in recalled.hits])
        project = self.pipeline.operate_modules(project, "chat", "撤销[retry_limit]", module_id)
        self.assertFalse([item for item in self.store.list_memory_records()
                          if item.project_id == project.id and item.source == "module_conversation"])

    def test_migration_does_not_widen_explicit_execution_memories(self):
        self.memory.remember("Execution specific rule", project_id="A", run_id="RUN-7",
                             source="case_conversation", infer=False)
        self.store._write_json(self.store.memory_migration_file, {})
        self.memory.migrate_legacy()
        self.assertFalse(self.memory.search("Execution", project_id="A").hits)
        self.assertEqual(len(self.memory.search("Execution", project_id="A", run_id="RUN-7").hits), 1)

    def test_failed_module_restore_does_not_deactivate_current_memories(self):
        from unittest.mock import patch
        project = self.store.create_project("Restore", "Users submit payment requests. Admins inspect audit logs.")
        project = self.pipeline.plan_modules(self.pipeline.analyze(project))
        old_version = project.module_versions[0].id
        project = self.pipeline.operate_modules(project, "chat", "Always retain audit logs",
                                                project.module_tree.modules[0].id)
        before = {item.id for item in self.store.list_memory_records()}
        with patch.object(self.pipeline, "_review_module_tree", side_effect=ValueError("critic failed")):
            with self.assertRaises(ValueError):
                self.pipeline.restore_module_version(self.store.get_project(project.id), old_version)
        self.assertEqual({item.id for item in self.store.list_memory_records()}, before)


if __name__ == "__main__":
    unittest.main()
