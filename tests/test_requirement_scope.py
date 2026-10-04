import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents import RequirementUnderstandingAgent
from app.clarification_policy import behavior_blockers, scope_summary
from app.llm import LLMError
from app.models import RequirementAnalysis, RequirementInput, ProjectState
from app.requirement_scope import review_behavior_scope

RAW = "On registration, a subscriber registers under a manager."
QUOTE = "At least one valid manager exists. The subscriber is registered under the manager."
CONTEXT = {"clarification_policy": "evidence_only", "input_evidence": {"DOC-1": {"content": QUOTE}}}


def analysis():
    return RequirementAnalysis(summary=RAW, atomic_requirements=[dict(id="R1", statement=RAW, source_quote=RAW)],
        ambiguities=[dict(id="A1", question="No manager outcome?")], clarification_items=[dict(
            id="G1", kind="behavior_blocker", question="No manager outcome?", requirement_ids=["R1"],
            ambiguity_ids=["A1"], affected_scenarios=["No manager"], source_quote=QUOTE, evidence_ids=["DOC-1"],
            reason="The inverse fixture has an undefined outcome")])


def decision(kind="out_of_scope"):
    return {"decisions": [dict(clarification_id="G1", kind=kind,
        reason="The inverse fixture is not a required scenario; retain the documented valid-fixture outcome",
        source_quote=QUOTE, evidence_ids=["DOC-1"])]}


class RequirementScopeTest(unittest.TestCase):
    def test_reclassification_keeps_gaps_and_requirements_and_persists_audit(self):
        original = analysis()
        generate = Mock(return_value=decision())
        revised = review_behavior_scope(SimpleNamespace(generate_json=generate), original, RAW, CONTEXT)
        self.assertEqual(revised.atomic_requirements, original.atomic_requirements)
        self.assertEqual(revised.ambiguities, original.ambiguities)
        self.assertEqual(revised.clarification_items[0].question, original.clarification_items[0].question)
        self.assertEqual(revised.clarification_items[0].ambiguity_ids, ["A1"])
        self.assertEqual(original.clarification_items[0].kind, "behavior_blocker")
        project = ProjectState(id="P", title="Registration", requirement=RAW, analysis=revised, clarification_policy="evidence_only")
        restored = ProjectState.model_validate_json(project.model_dump_json())
        self.assertEqual(restored.analysis.clarification_scope_review[0].original_kind, "behavior_blocker")
        self.assertEqual(restored.analysis.clarification_scope_review[0].original_reason, original.clarification_items[0].reason)
        self.assertEqual(behavior_blockers(restored), [])
        self.assertEqual(scope_summary(restored)["clarification_scope_review"][0]["kind"], "out_of_scope")

    def test_genuine_conflict_is_retained_and_scope_failure_never_downgrades(self):
        original = analysis()
        revised = review_behavior_scope(SimpleNamespace(generate_json=Mock(return_value=decision("behavior_blocker"))), original, RAW, CONTEXT)
        project = ProjectState(id="P", title="Conflict", requirement=RAW, analysis=revised, clarification_policy="evidence_only")
        self.assertEqual(len(behavior_blockers(project)), 1)
        self.assertEqual(scope_summary(project)["execution_readiness"], "blocked")
        for malformed in [{"decisions": []}, {"decisions": [decision()["decisions"][0]] * 2},
                          {"decisions": [{**decision()["decisions"][0], "clarification_id": "unknown"}]},
                          {"decisions": [{**decision()["decisions"][0], "source_quote": "Registration always succeeds"}]},
                          {"decisions": [{**decision()["decisions"][0], "evidence_ids": ["foreign"]}]}]:
            generate = Mock(return_value=malformed)
            with self.assertRaises(LLMError):
                review_behavior_scope(SimpleNamespace(generate_json=generate), original, RAW, CONTEXT)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(original.clarification_items[0].kind, "behavior_blocker")

    def test_retry_preserves_full_input_and_transport_failures_do_not_retry(self):
        generate = Mock(side_effect=[LLMError("Extra data", code="invalid_json"), decision()])
        review_behavior_scope(SimpleNamespace(generate_json=generate), analysis(), RAW, CONTEXT)
        self.assertEqual(generate.call_args_list[0][0][1:], generate.call_args_list[1][0][1:])
        for code in ["timeout", "request_failed"]:
            generate = Mock(side_effect=LLMError("private", code=code))
            with self.assertRaises(LLMError):
                review_behavior_scope(SimpleNamespace(generate_json=generate), analysis(), RAW, CONTEXT)
            self.assertEqual(generate.call_count, 1)

    def test_linked_gap_classifications_must_be_consistent_after_review(self):
        original = analysis()
        second = original.clarification_items[0].model_copy(deep=True)
        second.id = "G2"
        original.clarification_items.append(second)
        first = decision()["decisions"][0]
        contradictory = {"decisions": [first, {**first, "clarification_id": "G2", "kind": "behavior_blocker"}]}
        coherent = {"decisions": [first, {**first, "clarification_id": "G2"}]}
        generate = Mock(side_effect=[contradictory, coherent])
        revised = review_behavior_scope(SimpleNamespace(generate_json=generate), original, RAW, CONTEXT)
        self.assertEqual(len(revised.clarification_items), 2)
        self.assertEqual(len(revised.clarification_scope_review), 2)
        self.assertTrue(all(g.kind == "out_of_scope" for g in revised.clarification_items))
        self.assertEqual(generate.call_count, 2)

    def test_agent_runs_independent_review_only_for_scoped_blockers(self):
        generate = Mock(side_effect=[analysis().model_dump(), decision()])
        result = RequirementUnderstandingAgent(SimpleNamespace(enabled=True, generate_json=generate)).run(
            RequirementInput(title="Register", content=RAW), CONTEXT)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.clarification_items[0].kind, "out_of_scope")
        self.assertIn("independent requirement scope reviewer", generate.call_args[0][0])
        for context in [{"clarification_policy": "strict"}, {"clarification_policy": "evidence_only"}]:
            generate = Mock()
            result = review_behavior_scope(SimpleNamespace(generate_json=generate), analysis(), RAW, context)
            self.assertEqual(result.clarification_items[0].kind, "behavior_blocker")
            generate.assert_not_called()
