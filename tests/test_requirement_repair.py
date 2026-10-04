import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.agents import RequirementUnderstandingAgent
from app.llm import LLMError
from app.models import RequirementInput

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
        agent, generate, context = self.execute([bad, valid()])
        agent.run(RequirementInput(title="Trace", content=RAW), context)
        prompt = generate.call_args_list[1][0][1]
        self.assertIn("Which interface?", prompt)
        self.assertIn("instead of deleting", prompt)
        self.assertIn("invalid_scope", prompt)

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
