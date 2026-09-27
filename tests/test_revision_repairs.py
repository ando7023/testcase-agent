import unittest
from types import SimpleNamespace

from app.agents import CaseRevisionAgent, CaseReviewAgent
from app.llm import LLMError
from app.models import ModuleTree, ReviewFinding, TestCase, RequirementAnalysis


class RevisionRepairsTest(unittest.TestCase):
    def setUp(self):
        self.tree = ModuleTree(modules=[dict(id="M1", name="领券", objective="领取")], confirmed=True)
        self.case = TestCase(id="C1", module_id="M1", title="失败原因", case_type="异常",
                             requirement_ids=["AR-1"], human_status="approved",
                             steps=[dict(action="库存置零", expected="库存为零"),
                                    dict(action="注入写入失败后领取", expected="写入失败")])
        self.finding = ReviewFinding(category="semantic", severity="high", case_id="C1",
                                     message="恢复库存后才能注入写入失败")

    def test_aliases_normalized_on_new_and_persisted_cases(self):
        for alias, canonical in [("正常", "functional"), ("normal", "functional"),
                                 ("边界", "boundary"), ("异常", "exception"),
                                 ("权限", "permission"), ("并发幂等", "state_transition"),
                                 ("concurrency_idempotency", "state_transition"),
                                 ("custom_type", "custom_type")]:
            raw = self.case.model_dump()
            raw["case_type"] = alias
            self.assertEqual(TestCase.model_validate(raw).case_type, canonical)

    def test_semantic_repair_reaches_model_and_survives_protection(self):
        edited = self.case.model_copy(deep=True)
        edited.steps[1].action = "恢复库存为5，注入写入失败后领取"
        edited.preconditions = ["独立活动ACT-4"]
        edited.test_data = {"activity_id": "ACT-4"}
        edited.requirement_ids = []
        edited.human_status = "pending"
        untouched = self.case.model_copy(update={"id": "C2"}, deep=True)
        unwanted = untouched.model_copy(update={"title": "不应修改"})
        prompts = []

        def generate(system, user, schema):
            prompts.append(user)
            return {"cases": [edited.model_dump(), unwanted.model_dump()]}

        agent = CaseRevisionAgent(SimpleNamespace(enabled=True, generate_json=generate), None)
        result = agent.run({"analysis": RequirementAnalysis(summary="优惠券领取").model_dump(),
                            "module_tree": self.tree.model_dump(),
                            "cases": [self.case.model_dump(), untouched.model_dump()],
                            "review": {"score": 58, "findings": [self.finding.model_dump()]}},
                           {"knowledge": "修复场景串扰"})
        repaired = TestCase.model_validate(result["cases"][0])
        self.assertIn('"semantic"', prompts[0])
        self.assertEqual(repaired.steps[1].action, edited.steps[1].action)
        self.assertEqual(repaired.test_data, edited.test_data)
        self.assertEqual(repaired.preconditions, edited.preconditions)
        self.assertEqual(repaired.requirement_ids, ["AR-1"])
        self.assertEqual(repaired.human_status, "approved")
        self.assertEqual(result["cases"][1], untouched.model_dump())

    def test_semantic_repair_cannot_delete_move_or_empty_case(self):
        for change in [None, {"module_id": "M2"}, {"steps": []}]:
            revised = [] if change is None else [self.case.model_copy(update=change)]
            with self.assertRaises(LLMError):
                CaseRevisionAgent._protect_existing_cases([self.case], revised, [self.finding], self.tree)

    def test_assertion_only_repair_still_cannot_rewrite_actions(self):
        finding = self.finding.model_copy(update={"category": "assertion"})
        edited = self.case.model_copy(deep=True)
        edited.steps[0].action = "改变业务动作"
        with self.assertRaisesRegex(LLMError, "Assertion repair"):
            CaseRevisionAgent._protect_existing_cases([self.case], [edited], [finding], self.tree)

    def test_failed_semantic_review_cannot_approve_cases(self):
        def fail(*args):
            raise LLMError("read timed out")
        agent = CaseReviewAgent(SimpleNamespace(enabled=True, generate_json=fail))
        result = agent.run({"analysis": RequirementAnalysis(summary="领取").model_dump(),
                            "module_tree": self.tree.model_dump(),
                            "cases": [self.case.model_dump()]}, {})
        self.assertTrue(any(f["category"] == "review_incomplete" and f["severity"] == "high"
                            for f in result["review"]["findings"]))
        self.assertEqual(result["cases"][0]["review_status"], "needs_attention")


if __name__ == "__main__":
    unittest.main()
