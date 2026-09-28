import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.feedback import case_fingerprint
from app.models import CaseFeedback, CaseSetVersion, ReviewReport, TestCase, TestStep
from app.orchestrator import TestCaseOrchestrator
from app.store import JsonStore
from app.supervisor import artifact_fingerprint


class FeedbackLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = JsonStore(Path(self.temp.name))
        self.worker = TestCaseOrchestrator(self.store)
        self.worker.llm.api_key = ""
        self.project = self.store.create_project("Coupon", "Login required to claim one coupon")
        self.project.cases = [TestCase(id="TC-1", module_id="MOD-1", title="coupon claim",
            steps=[TestStep(action="claim coupon", expected="stock decreases once")])]
        self.project.review = ReviewReport(score=100)
        self.store.save_project(self.project)

    def examples(self, active=True):
        return [d for d in self.store.list_knowledge() if d.doc_type == "case_example"
                and (not active or d.status == "active")]

    def adopt(self):
        return self.worker.record_feedback(self.project, "TC-1", "adopted")

    def test_adoption_preserves_review_and_binds_snapshot(self):
        before = artifact_fingerprint(self.project)
        self.adopt()
        self.assertEqual(artifact_fingerprint(self.project), before)
        self.assertEqual(self.project.review.score, 100)
        self.assertEqual(self.project.human_acceptance["status"], "accepted")
        feedback = self.project.feedback[-1]
        self.assertEqual(feedback.case_fingerprint, case_fingerprint(self.project.cases[0]))
        self.assertEqual(feedback.case_version_id, self.project.case_versions[-1].id)
        self.assertEqual(self.store.get_project(self.project.id).human_acceptance["accepted"], 1)

    def test_body_change_revokes_acceptance_and_retrievable_example(self):
        self.adopt()
        fingerprint = artifact_fingerprint(self.project)
        self.project.cases[0].steps[0].expected = "new requirement"
        self.store.save_project(self.project)
        self.assertNotEqual(artifact_fingerprint(self.project), fingerprint)
        self.assertEqual(self.project.cases[0].human_status, "pending")
        self.assertEqual(self.project.human_acceptance["pending"], 1)
        self.assertEqual(self.examples(), [])
        self.assertEqual(len(self.examples(active=False)), 1)

    def test_rejection_withdraws_example_and_reacceptance_reactivates(self):
        self.adopt()
        original = self.examples()[0].id
        self.worker.record_feedback(self.project, "TC-1", "rejected")
        self.assertEqual(self.project.human_acceptance["rejected"], 1)
        self.assertEqual(self.examples(), [])
        self.assertFalse([d for d in self.store.project_knowledge(self.project.id) if d.doc_type == "case_example"])
        self.adopt()
        self.assertEqual([d.id for d in self.examples()], [original])

    def test_edit_creates_version_and_invalidates_model_review(self):
        self.adopt()
        old = self.project.feedback[-1].case_version_id
        self.worker.record_feedback(self.project, "TC-1", "edited", edited_case={"title": "new coupon title"})
        self.assertIsNone(self.project.review)
        self.assertEqual(self.project.phase, "cases_generated")
        self.assertNotEqual(old, self.project.feedback[-1].case_version_id)
        self.assertEqual(self.project.human_acceptance["status"], "accepted")
        self.assertEqual(len(self.examples()), 1)
        self.assertEqual(len(self.examples(active=False)), 2)
        self.assertIn("new coupon title", self.examples()[0].title)

    def test_examples_are_project_scoped_even_for_common_business(self):
        self.adopt()
        example = self.examples()[0]
        other = self.store.create_project("Other", self.project.requirement)
        for project_id in ("", other.id):
            self.assertNotIn(example.id, [d.id for d in self.store.project_knowledge(project_id)])
            result = self.worker._context("coupon claim stock", project_id=project_id)
            self.assertNotIn(example.id, [d["document_id"] for d in result["knowledge_context"]["hits"]])
        own = self.worker._context("coupon claim stock", project_id=self.project.id)
        self.assertIn(example.id, [d["document_id"] for d in own["knowledge_context"]["hits"]])
        self.assertEqual(example.metadata["ticket_type"], "COMMON")

    def test_legacy_adoption_requires_matching_earlier_snapshot(self):
        p = self.project
        p.case_versions = [CaseSetVersion(id="CV-old", mode="full", cases=[p.cases[0].model_copy(deep=True)],
                                        created_at="2026-01-01T00:00:00Z")]
        p.feedback = [CaseFeedback(case_id="TC-1", action="adopted", created_at="2026-01-02T00:00:00Z")]
        self.store.save_project(p)
        self.assertEqual(p.human_acceptance["legacy_verified"], 1)
        p.cases[0].title = "changed after feedback"
        self.store.save_project(p)
        self.assertEqual(p.human_acceptance["accepted"], 0)
        self.assertEqual(self.examples(), [])
        p.case_versions = []
        self.assertEqual(p.human_acceptance["accepted"], 0)

    def test_startup_legacy_migration_is_idempotent_and_preserves_project_file(self):
        p = self.project
        p.case_versions = [CaseSetVersion(id="CV-old", mode="full", cases=[p.cases[0].model_copy(deep=True)],
                                        created_at="2026-01-01T00:00:00Z")]
        p.feedback = [CaseFeedback(case_id="TC-1", action="adopted", created_at="2026-01-02T00:00:00Z")]
        path = self.store.projects_dir / (p.id + ".json")
        path.write_text(p.model_dump_json(), encoding="utf-8")
        self.store.upsert_knowledge([dict(id="EX-{}-TC-1".format(p.id), title="old example",
                                          content="old body", doc_type="case_example")])
        before = path.read_bytes()
        TestCaseOrchestrator(self.store)
        first = self.store.knowledge_file.read_bytes()
        TestCaseOrchestrator(self.store)
        self.assertEqual(first, self.store.knowledge_file.read_bytes())
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(len(self.examples()), 1)
        self.assertEqual(self.examples()[0].metadata["provenance"], "legacy_snapshot")
        self.assertEqual(self.examples()[0].metadata["case_version_id"], "CV-old")

    def test_unrelated_case_change_does_not_revoke_other_acceptance(self):
        self.adopt()
        self.project.cases.append(TestCase(id="TC-2", module_id="MOD-1", title="new case",
                                 steps=[TestStep(action="read", expected="visible")]))
        self.store.save_project(self.project)
        self.assertEqual(self.project.cases[0].human_status, "adopted")
        self.assertEqual(self.project.human_acceptance["status"], "partial")
        self.assertEqual(len(self.examples()), 1)
        self.project.cases = []
        self.store.save_project(self.project)
        self.assertEqual(self.examples(), [])

    def test_later_snapshot_cannot_prove_earlier_legacy_adoption(self):
        p = self.project
        p.feedback = [CaseFeedback(case_id="TC-1", action="adopted", created_at="2026-01-01T00:00:00Z")]
        p.case_versions = [CaseSetVersion(id="CV-later", mode="full", cases=[p.cases[0]],
                                        created_at="2026-01-02T00:00:00Z")]
        self.store.save_project(p)
        self.assertEqual(p.cases[0].human_status, "pending")
        self.assertEqual(self.examples(), [])

    def test_api_exposes_acceptance_without_rewriting_degraded_run(self):
        from fastapi.testclient import TestClient
        from app import api
        from app.supervisor_models import SupervisorRun
        run = SupervisorRun(id="AR-" + "a" * 32, project_id=self.project.id, goal="generate cases",
                            mode="model", status="needs_attention", degraded=True)
        self.store.save_agent_run(run)
        path = self.store.root / "agent_runs" / (run.id + ".json")
        before = path.read_bytes()
        with patch.object(api, "store", self.store), patch.object(api, "orchestrator", self.worker):
            client = TestClient(api.app)
            base = "/api/projects/" + self.project.id
            response = client.post(base + "/cases/TC-1/feedback", json={"action": "adopted"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["human_acceptance"]["status"], "accepted")
            response = client.get(base)
            self.assertEqual(response.json()["human_acceptance"]["accepted"], 1)
            response = client.post(base + "/cases/TC-1/feedback", json={"action": "edited", "edited_case": {"title": "changed"}})
            self.assertEqual(response.status_code, 200)
            self.assertIsNone(response.json()["review"])
            self.assertEqual(response.json()["phase"], "cases_generated")
        self.assertEqual(before, path.read_bytes())

    def test_stale_or_tampered_index_cannot_override_current_decision(self):
        self.adopt()
        example = self.examples()[0]
        example.content = "unrelated injected body"
        self.store.upsert_knowledge([example.model_dump()])
        self.assertNotIn(example.id, [d.id for d in self.store.project_knowledge(self.project.id)])
        # Simulate interrupted synchronization: the project was saved but the index was not.
        p = self.project.model_copy(deep=True)
        p.feedback.append(CaseFeedback(case_id="TC-1", action="rejected", case_fingerprint=case_fingerprint(p.cases[0])))
        path = self.store.projects_dir / (p.id + ".json")
        path.write_text(p.model_dump_json(), encoding="utf-8")
        self.assertNotIn(example.id, [d.id for d in self.store.project_knowledge(p.id)])
