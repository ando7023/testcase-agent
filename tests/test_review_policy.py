import json
import unittest
from types import SimpleNamespace

from app.agents import CaseReviewAgent, CaseRevisionAgent
from app.llm import LLMError
from app.models import ReviewFinding, ReviewReport, ModuleTree, TestCase, RequirementAnalysis
from app.review_policy import is_blocking, is_auto_fixable, record_review, restore_review_history
from app.supervisor_models import SupervisorRun, SupervisorStep, SupervisorDecision


class ReviewPolicyTest(unittest.TestCase):
    def setUp(self):
        self.case = TestCase(id="C1", module_id="M1", title="领取", source_evidence=["领券需求"], steps=[
            {"action": "库存置零后注入写入失败", "expected": "写入失败"}])
        self.payload = {"requirement": "原始需求：库存不足拒绝领取", "project_context": "仅领取",
                        "review_constraints": "不得强加错误码契约", "review_focus": "检查场景串扰",
                        "review_history": [{"score": 80}], "issue_ledger": {"old": {"status": "not_observed"}},
                        "analysis": RequirementAnalysis(summary="领券").model_dump(),
                        "module_tree": ModuleTree(modules=[dict(id="M1", name="领取", objective="领取")]).model_dump(),
                        "cases": [self.case.model_dump(),
                                  self.case.model_copy(update={"id": "C2", "case_type": "boundary"}).model_dump(),
                                  self.case.model_copy(update={"id": "C3", "case_type": "exception"}).model_dump()]}

    def review(self, result):
        def generate(system, user, schema):
            self.prompt = json.loads(user)
            self.schema = schema
            return result
        return CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(self.payload, {})

    def finding(self, **kwargs):
        return dict(case_id="C1", severity="low", message="优化重复断言", disposition="suggestion",
                    issue_type="redundancy", evidence="库存置零后注入写入失败", requirement_ids=[], **kwargs)

    def test_full_context_reaches_critic(self):
        self.review({"findings": []})
        for key in ["raw_requirement", "analysis", "user_constraints", "previous_reviews", "issue_ledger"]:
            self.assertTrue(self.prompt[key])
        self.assertEqual(self.prompt["user_constraints"], self.payload["review_constraints"])
        self.assertEqual(self.prompt["raw_requirement"], self.payload["requirement"])
        self.assertEqual(self.schema["required"], ["findings"])

    def test_json_retry_keeps_full_context_and_validates_second_review(self):
        prompts = []
        def generate(system, user, schema):
            prompts.append((system, user, schema))
            if len(prompts) == 1:
                raise LLMError("Extra data", code="invalid_json")
            return {"findings": [{**self.finding(), "disposition": "defect", "severity": "high"}]}
        result = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(self.payload, {})
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0][1:], prompts[1][1:])
        self.assertIn("previous response failed JSON parsing", prompts[1][0])
        finding = ReviewFinding.model_validate(result["review"]["findings"][0])
        self.assertTrue(is_blocking(finding))
        self.assertEqual(result["cases"][0]["review_status"], "needs_attention")

    def test_json_retry_is_bounded_and_other_failures_do_not_retry(self):
        for code, attempts in [("invalid_json", 2), ("output_limit", 1), ("timeout", 1), ("request_failed", 1)]:
            calls = []
            def generate(*args):
                calls.append(args)
                raise LLMError("failure", code=code)
            result = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(self.payload, {})
            self.assertEqual(len(calls), attempts)
            failures = [f for f in result["review"]["findings"] if f["category"] == "review_incomplete"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["detail"], code)
            self.assertEqual(result["cases"][0]["review_status"], "needs_attention")

    def test_grounded_suggestion_is_retained_but_does_not_block_or_repair(self):
        result = self.review({"findings": [self.finding()]})
        f = ReviewFinding.model_validate(result["review"]["findings"][0])
        self.assertFalse(is_blocking(f))
        self.assertFalse(is_auto_fixable(f))
        self.assertTrue(f.issue_id)
        self.assertEqual(result["cases"][0]["review_status"], "approved")
        def unexpected(*args):
            raise AssertionError("Suggestions must not trigger model repair")
        revision = CaseRevisionAgent(SimpleNamespace(enabled=True, generate_json=unexpected), None)
        repaired = revision.run({**self.payload, "review": result["review"]}, {})
        self.assertEqual(repaired["cases"], self.payload["cases"])

    def test_unsupported_evidence_or_unknown_requirement_is_technical_failure(self):
        for changes in [{"evidence": "并不存在的缓存机制"}, {"requirement_ids": ["AR-FAKE"]}, {"disposition": "unknown"}]:
            item = self.finding();item.update(changes)
            f = ReviewFinding.model_validate(self.review({"findings": [item]})["review"]["findings"][0])
            self.assertEqual(f.category, "review_incomplete")
            self.assertEqual(f.detail, "invalid_schema")
            self.assertTrue(is_blocking(f))
            self.assertFalse(is_auto_fixable(f))

    def test_quote_feedback_preserves_nonblocking_suggestion(self):
        prompts = []
        def generate(system, user, schema):
            prompts.append((system, user))
            item = self.finding()
            if len(prompts) == 1:
                item["evidence"] = 'action": "' + item["evidence"]
            return {"findings": [item]}
        result = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(self.payload, {})
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0][1], prompts[1][1])
        self.assertIn("findings[0].evidence", prompts[1][0])
        f = ReviewFinding.model_validate(result["review"]["findings"][0])
        self.assertEqual(f.disposition, "suggestion")
        self.assertFalse(is_blocking(f))
        self.assertEqual(result["cases"][0]["review_status"], "approved")

    def test_quote_retry_does_not_discard_high_defect(self):
        calls = []
        def generate(*args):
            calls.append(args)
            return {"findings": [{**self.finding(), "severity": "high", "disposition": "defect",
                                   "evidence": 'action": "' + self.finding()["evidence"]}]}
        result = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(self.payload, {})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["review"]["findings"][0]["category"], "review_incomplete")
        self.assertEqual(result["review"]["findings"][0]["detail"], "invalid_schema")
        self.assertEqual(result["cases"][0]["review_status"], "needs_attention")

    def test_high_severity_cannot_be_hidden_as_a_suggestion(self):
        for severity in ["high", "critical", "error"]:
            item = self.finding();item["severity"] = severity
            f = ReviewFinding.model_validate(self.review({"findings": [item]})["review"]["findings"][0])
            self.assertTrue(is_blocking(f))
            self.assertEqual(f.disposition, "defect")

    def test_evidence_matches_original_text_including_context_and_escaped_characters(self):
        self.payload["project_context"] = '第一行\n失败显示"原始原因"'
        item = self.finding()
        item["evidence"] = self.payload["project_context"]
        result = self.review({"findings": [item]})
        self.assertEqual(result["review"]["findings"][0]["disposition"], "suggestion")
        item["evidence"] = "project_context"
        result = self.review({"findings": [item]})
        self.assertEqual(result["review"]["findings"][0]["category"], "review_incomplete")

    def test_legacy_findings_keep_conservative_gate(self):
        f = ReviewFinding(category="semantic", severity="low", case_id="C1", message="旧问题")
        self.assertTrue(is_blocking(f))
        self.assertTrue(is_auto_fixable(f))
        self.assertTrue(is_blocking(ReviewFinding(category="case_type", severity="medium", message="缺少类型", disposition="suggestion")))

    def test_malformed_review_cannot_pass_as_empty_findings(self):
        for raw in [{}, {"findings": "none"}, {"findings": [self.finding(), {"case_id": "FAKE", "message": "bad"}]},
                    {"findings": [self.finding()] * 21},
                    *[{"findings": [{**self.finding(), key: []}]} for key in ["severity", "case_id", "disposition", "issue_type"]]]:
            result = self.review(raw)
            self.assertTrue(any(f["category"] == "review_incomplete" for f in result["review"]["findings"]))
            self.assertEqual(result["cases"][0]["review_status"], "needs_attention")

    def test_ledger_tracks_absence_reopening_and_classification_changes(self):
        run = SupervisorRun(id="AR-" + "0"*32, project_id="P", goal="test", mode="model")
        item = self.finding();item["disposition"] = "defect"
        finding = ReviewFinding(category="semantic", **item)
        record_review(run, ReviewReport(score=90, findings=[finding]), "version1")
        key = next(iter(run.issue_ledger))
        record_review(run, ReviewReport(score=100), "version2")
        self.assertEqual(run.issue_ledger[key]["status"], "not_observed")
        finding.message = "同类问题改写说明"
        finding.disposition = "suggestion"
        record_review(run, ReviewReport(score=98, findings=[finding]), "version3")
        self.assertEqual(run.issue_ledger[key]["reopened"], 1)
        self.assertTrue(run.issue_ledger[key]["classification_changed"])
        incomplete = ReviewReport(score=88, findings=[ReviewFinding(category="review_incomplete", severity="high", message="失败")])
        record_review(run, incomplete, "version4")
        self.assertEqual(len(run.review_history), 3)
        restored = SupervisorRun.model_validate_json(run.model_dump_json())
        self.assertEqual(restored.issue_ledger, run.issue_ledger)

    def test_old_run_history_is_restored_without_marking_review_fresh(self):
        run = SupervisorRun(id="AR-" + "1"*32, project_id="P", goal="test", mode="model")
        report = ReviewReport(score=90, findings=[ReviewFinding(category="semantic", severity="medium", message="历史问题", case_id="C1")])
        run.steps = [SupervisorStep(index=1, status="success", decision=SupervisorDecision(
            action="invoke_agent", capability="quality_critic", reason="test"), observation={"review": report.model_dump()})]
        restore_review_history(run)
        self.assertEqual(len(run.review_history), 1)
        self.assertEqual(run.review_fingerprint, "")
        self.assertEqual(run.review_history[0]["fingerprint"], "")
        self.assertEqual(next(iter(run.issue_ledger.values()))["finding"]["disposition"], "legacy")
        restore_review_history(run)
        self.assertEqual(len(run.review_history), 1)
