import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents import RequirementUnderstandingAgent
from app.llm import LLMError
from app.models import RequirementInput, RequirementAnalysis
from app.clarification_policy import ambiguity_records, validate_analysis, validation_diagnostics

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
        with self.assertRaisesRegex(LLMError, "invalid_ambiguity_id"):
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

    def test_supplied_source_quote_requires_valid_citation(self):
        evidence = {"EBT-SAMPLE-TEST-143": {"content": "The subscriber is registered under the subscriber manager."}}
        data = valid()
        data["atomic_requirements"][0].update(source_quote=evidence["EBT-SAMPLE-TEST-143"]["content"],
                                             evidence_ids=["EBT-SAMPLE-TEST-143"])
        model = RequirementAnalysis.model_validate(data)
        validate_analysis(model, RAW, evidence)
        for references in ([], ["another-sample"], ["EBT-SAMPLE-TEST-143", "invented"]):
            bad = model.model_copy(deep=True)
            bad.atomic_requirements[0].evidence_ids = references
            with self.assertRaises(LLMError):
                validate_analysis(bad, RAW, evidence)
        with self.assertRaises(LLMError):
            validate_analysis(model, RAW)  # Ordinary RAG cannot authorize new input evidence.
        model.atomic_requirements[0].source_quote = "Registration always succeeds."
        with self.assertRaises(LLMError):
            validate_analysis(model, RAW, evidence)

    def test_evidence_citation_correction_and_clarification_are_grounded(self):
        quote = "The subscriber is registered under the subscriber manager."
        evidence = {"EBT-SAMPLE-TEST-143": {"content": quote}}
        bad = valid()
        bad["atomic_requirements"][0]["source_quote"] = quote
        fixed = valid()
        fixed["atomic_requirements"][0].update(source_quote=quote, evidence_ids=["EBT-SAMPLE-TEST-143"])
        fixed["clarification_items"] = [{"id": "G1", "kind": "execution_detail", "question": "Which registration interface?",
            "requirement_ids": ["R1"], "affected_scenarios": ["Registration"], "source_quote": quote,
            "evidence_ids": ["EBT-SAMPLE-TEST-143"], "reason": "Outcome is defined, invocation is not."}]
        agent, generate, context = self.execute([bad, fixed])
        context["input_evidence"] = evidence
        result = agent.run(RequirementInput(title="Registration", content=RAW), context)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.atomic_requirements[0].evidence_ids, ["EBT-SAMPLE-TEST-143"])
        self.assertIn(quote, generate.call_args_list[0][0][0])
        result.clarification_items[0].evidence_ids = []
        with self.assertRaises(LLMError):
            validate_analysis(result, RAW, evidence)

    def test_document_id_confusion_and_missing_quote_citation_are_both_reported(self):
        quote = "A valid subscriber manager exists."
        evidence = {"EBT-SAMPLE-REQ-103": {"content": RAW}, "EBT-SAMPLE-TEST-143": {"content": quote}}
        bad = valid()
        bad["ambiguities"] = [{"id": "A1", "question": "Which API?"}]
        bad["clarification_items"] = [{"id": "G1", "kind": "execution_detail", "question": "Which API?",
            "requirement_ids": ["EBT-SAMPLE-REQ-103"], "source_quote": quote, "reason": "Invocation detail",
            "affected_scenarios": ["Registration"], "ambiguity_ids": []}]
        model = RequirementAnalysis.model_validate(bad)
        with self.assertRaises(LLMError) as caught:
            validate_analysis(model, RAW, evidence)
        self.assertEqual(caught.exception.validation_issues, [
            {"field": "clarification_items[0].requirement_ids", "code": "unknown_requirement_id"},
            {"field": "clarification_items[0].source_quote", "code": "ungrounded_quote"},
            {"field": "ambiguities[0]", "code": "unclassified_ambiguity"}])
        fixed = RequirementAnalysis.model_validate(bad)
        fixed.clarification_items[0].requirement_ids = ["R1"]
        fixed.clarification_items[0].evidence_ids = ["EBT-SAMPLE-TEST-143"]
        fixed.clarification_items[0].ambiguity_ids = ["A1"]
        agent, generate, context = self.execute([bad, fixed.model_dump()])
        context["input_evidence"] = evidence
        result = agent.run(RequirementInput(title="Registration", content=RAW), context)
        self.assertEqual(generate.call_count, 2)
        prompt = generate.call_args[0][1]
        self.assertIn("clarification_items[0].requirement_ids", prompt)
        self.assertIn('"allowed_requirement_ids": ["R1"]', prompt)
        self.assertIn('"matching_evidence_ids": ["EBT-SAMPLE-TEST-143"]', prompt)
        self.assertEqual(result.clarification_items[0].ambiguity_ids, ["A1"])

    def test_atomic_id_cannot_reuse_source_id_and_diagnostics_do_not_leak_values(self):
        bad = valid()
        bad["atomic_requirements"][0]["id"] = "EBT-SAMPLE-REQ-103"
        evidence = {"EBT-SAMPLE-REQ-103": {"content": RAW}}
        with self.assertRaises(LLMError) as caught:
            validate_analysis(RequirementAnalysis.model_validate(bad), RAW, evidence)
        self.assertEqual(caught.exception.validation_issues[0]["code"], "source_id_as_requirement")
        self.assertNotIn("EBT-SAMPLE-REQ-103", str(caught.exception))
        self.assertEqual(validation_diagnostics([None, {"field": "private-value", "code": "ungrounded_quote"},
            {"field": "atomic_requirements", "code": []}, {"field": "atomic_requirements", "code": "private-value"}]), [])


if __name__ == "__main__":
    unittest.main()
