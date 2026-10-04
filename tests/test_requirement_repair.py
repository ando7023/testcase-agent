import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents import RequirementUnderstandingAgent
from app.llm import LLMError
from app.models import RequirementInput, RequirementAnalysis
from app.clarification_policy import ambiguity_records, validate_analysis

RAW = "An unregistered user cannot establish a trace."


def valid():
    return {"summary": RAW, "atomic_requirements": [{"id": "R1", "statement": RAW, "source_quote": RAW}]}


class RequirementRepairTests(unittest.TestCase):
    def execute(self, results, policy="evidence_only"):
        generate = Mock(side_effect=results)
        agent = RequirementUnderstandingAgent(SimpleNamespace(enabled=True, generate_json=generate))
        return agent, generate, {"clarification_policy": policy}

    def test_invalid_json_has_one_feedback_retry_with_original_input(self):
        agent, generate, context = self.execute([LLMError("Extra data", code="invalid_json"), valid()])
        result = agent.run(RequirementInput(title="Trace", content=RAW), context)
        self.assertEqual(result.atomic_requirements[0].source_quote, RAW)
        self.assertEqual(generate.call_count, 2)
        prompt = generate.call_args_list[1][0][1]
        self.assertIn(RAW, prompt)
        self.assertIn("exactly one JSON object", prompt)

    def test_scope_feedback_keeps_previous_unresolved_gaps(self):
        bad = valid()
        bad["ambiguities"] = ["Which interface?"]
        records = ambiguity_records(RequirementAnalysis.model_validate(bad))
        fixed = valid()
        fixed["ambiguities"] = records
        fixed["clarification_items"] = [{"id": "G1", "kind": "execution_detail", "question": "Which API is used?",
            "ambiguity_ids": [records[0]["id"]], "requirement_ids": ["R1"], "affected_scenarios": ["Trace"],
            "source_quote": RAW, "reason": "Outcome known, interface unspecified"}]
        agent, generate, context = self.execute([bad, fixed])
        agent.run(RequirementInput(title="Trace", content=RAW), context)
        prompt = generate.call_args_list[1][0][1]
        self.assertIn("Which interface?", prompt)
        self.assertIn("instead of deleting", prompt)
        self.assertIn("invalid_scope", prompt)

    def test_retry_cannot_silently_drop_unresolved_ambiguity(self):
        bad = valid()
        bad["ambiguities"] = [{"id": "A1", "question": "Which interface?"}]
        agent, generate, context = self.execute([bad, valid()])
        with self.assertRaisesRegex(LLMError, "dropped unresolved"):
            agent.run(RequirementInput(title="Trace", content=RAW), context)
        self.assertEqual(generate.call_count, 2)

    def test_id_mapping_allows_rewording_but_rejects_missing_unknown_and_conflicting_links(self):
        data = valid()
        data["ambiguities"] = [{"id": "A1", "question": "Which endpoint?"}]
        data["clarification_items"] = [{"id": "G1", "kind": "execution_detail", "question": "How is the API invoked?",
            "ambiguity_ids": ["A1"], "requirement_ids": ["R1"], "affected_scenarios": ["Trace"],
            "source_quote": RAW, "reason": "Behavior is determined without endpoint details"}]
        model = RequirementAnalysis.model_validate(data)
        validate_analysis(model, RAW)
        for ids in [[], ["unknown"], ["A1", "A1"]]:
            bad = model.model_copy(deep=True)
            bad.clarification_items[0].ambiguity_ids = ids
            with self.assertRaises(LLMError):
                validate_analysis(bad, RAW)
        bad = model.model_copy(deep=True)
        other = bad.clarification_items[0].model_copy(deep=True)
        other.id, other.kind = "G2", "behavior_blocker"
        bad.clarification_items.append(other)
        with self.assertRaisesRegex(LLMError, "conflicting"):
            validate_analysis(bad, RAW)
        bad = model.model_copy(deep=True)
        bad.ambiguities.append(bad.ambiguities[0])
        with self.assertRaisesRegex(LLMError, "unique"):
            validate_analysis(bad, RAW)

    def test_legacy_analysis_keeps_exact_match_compatibility_without_fuzzy_downgrade(self):
        data = valid()
        data["ambiguities"] = ["Which endpoint?"]
        data["clarification_items"] = [{"id": "G1", "kind": "execution_detail", "question": "Which endpoint?",
            "requirement_ids": ["R1"], "affected_scenarios": ["Trace"], "source_quote": RAW, "reason": "Known outcome"}]
        model = RequirementAnalysis.model_validate(data)
        validate_analysis(model, RAW)
        model.clarification_items[0].question = "Does permission apply?"
        with self.assertRaises(LLMError):
            validate_analysis(model, RAW)

    def test_schema_failure_feedback_does_not_echo_invalid_input(self):
        agent, generate, context = self.execute([{"summary": {"secret": "private-value"}}, valid()])
        agent.run(RequirementInput(title="Trace", content=RAW), context)
        # Previous analysis is task data; error feedback itself excludes Pydantic input.
        prompt = generate.call_args_list[1][0][1].split("Previous analysis")[0]
        self.assertIn("invalid_schema", prompt)
        self.assertNotIn("private-value", prompt)

    def test_repeated_format_or_scope_failure_stops_after_two_calls(self):
        for failure in [LLMError("Extra data", code="invalid_json"),
                        LLMError("Unclassified ambiguity", code="invalid_scope")]:
            agent, generate, context = self.execute([failure, failure, valid()])
            with self.assertRaises(LLMError) as caught:
                agent.run(RequirementInput(title="Trace", content=RAW), context)
            self.assertEqual(caught.exception.code, failure.code)
            self.assertEqual(generate.call_count, 2)

    def test_transport_and_auth_failure_are_not_retried(self):
        for code in ["timeout", "request_failed", "http_error"]:
            agent, generate, context = self.execute([LLMError("Private provider details", code=code), valid()])
            with self.assertRaises(LLMError):
                agent.run(RequirementInput(title="Trace", content=RAW), context)
            self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
