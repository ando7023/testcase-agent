import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from app.llm import LLMError
from app.tooling import ReActRuntime


class ResearchRepairTests(unittest.TestCase):
    def test_repair_preserves_observations_without_replaying_tools(self):
        generate = Mock(side_effect=[{"type": "tool", "tool": "read", "arguments": {}},
            LLMError("Extra data", code="invalid_json"), {"type": "finish", "summary": "Evidence collected"}])
        registry = SimpleNamespace(schemas_for=lambda _: [], execute=Mock(return_value="Recorded evidence"))
        result = ReActRuntime(SimpleNamespace(enabled=True, generate_json=generate), registry).run("requirement", "Analyze", [])
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(registry.execute.call_count, 1)
        self.assertEqual(result["mode"], "react")
        self.assertIn("Recorded evidence", generate.call_args_list[2][0][1])
        self.assertIn("invalid_json", generate.call_args_list[2][0][1])

    def test_research_has_one_format_repair_for_the_entire_run(self):
        generate = Mock(side_effect=[LLMError("Extra", code="invalid_json"),
            {"type": "tool", "tool": "read", "arguments": {}}, LLMError("Extra", code="invalid_json")])
        registry = SimpleNamespace(schemas_for=lambda _: [], execute=Mock(return_value="Evidence"))
        with self.assertRaises(LLMError):
            ReActRuntime(SimpleNamespace(enabled=True, generate_json=generate), registry).run("requirement", "Analyze", [])
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(registry.execute.call_count, 1)

    def test_network_errors_do_not_retry_research(self):
        generate = Mock(side_effect=LLMError("timeout", code="timeout"))
        registry = SimpleNamespace(schemas_for=lambda _: [], execute=Mock())
        with self.assertRaises(LLMError):
            ReActRuntime(SimpleNamespace(enabled=True, generate_json=generate), registry).run("requirement", "Analyze", [])
        self.assertEqual(generate.call_count, 1)
        registry.execute.assert_not_called()
