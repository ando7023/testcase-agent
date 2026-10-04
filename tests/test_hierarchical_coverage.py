import unittest
from types import SimpleNamespace

from app.agents import CaseReviewAgent
from app.models import ModuleTree, RequirementAnalysis, TestCase


class HierarchicalCoverageTest(unittest.TestCase):
    def review(self, assigned):
        tree = ModuleTree(modules=[dict(id="root", name="Registration", objective="Registration",
            children=[dict(id="group", name="Flows", objective="Flows", children=[
                dict(id="success", name="Register", objective="Register"),
                dict(id="exit", name="Exit", objective="Exit")])])])
        cases = [TestCase(id="C" + str(i), module_id=module, title=module,
                         source_evidence=["Registration"], steps=[dict(action="Act", expected="Observe")])
                 for i, module in enumerate(assigned)]
        result = CaseReviewAgent(SimpleNamespace(enabled=False)).run({
            "analysis": RequirementAnalysis(summary="Registration").model_dump(),
            "module_tree": tree.model_dump(), "cases": [c.model_dump() for c in cases]},
            {"clarification_policy": "evidence_only"})
        return {f["module_id"] for f in result["review"]["findings"] if f["category"] == "module_coverage"}

    def test_descendants_cover_grouping_modules(self):
        self.assertEqual(self.review(["success", "exit"]), set())

    def test_sibling_or_parent_case_does_not_cover_empty_leaf(self):
        self.assertEqual(self.review(["root", "success"]), {"exit"})

    def test_empty_subtree_and_unknown_assignment_remain_uncovered(self):
        self.assertEqual(self.review(["unknown"]), {"root", "group", "success", "exit"})
