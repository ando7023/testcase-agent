import io
import json
import os
import unittest
from unittest.mock import patch

from app.llm import LLMError, OpenAICompatibleClient


GLM_ENV = {
    "ZHIPU_API_KEY": "test-zhipu-key",
    "DEEPSEEK_API_KEY": "test-old-provider-key",
    "LLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4/",
    "LLM_MODEL": "glm-5.3-flash",
    "LLM_REASONING_EFFORT": "max",
    "LLM_THINKING_ENABLED": "true",
    "LLM_TEMPERATURE": "1",
    "LLM_TOP_P": "0.95",
    "LLM_MAX_TOKENS": "32768",
    "LLM_TIMEOUT_SECONDS": "180",
}


class GLMClientTest(unittest.TestCase):
    def client(self, **overrides):
        with patch.dict(os.environ, dict(GLM_ENV, **overrides), clear=True):
            return OpenAICompatibleClient()

    def test_glm_nonstream_request_and_structured_response(self):
        client = self.client()
        response = {"choices": [{"message": {"content": '{"action":"finish"}', "reasoning_content": "private reasoning"}}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as request:
            result = client.generate_json("Choose one action", "Finish", {"type": "object"})
        self.assertEqual(result, {"action": "finish"})
        req = request.call_args[0][0]
        self.assertEqual(req.full_url, "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-zhipu-key")
        body = json.loads(req.data)
        self.assertEqual(body["model"], "glm-5.3-flash")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["thinking"], {"type": "enabled", "clear_thinking": False})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual((body["temperature"], body["top_p"], body["max_tokens"]), (1, 0.95, 32768))
        self.assertIn("JSON", body["messages"][0]["content"])
        self.assertEqual(request.call_args[1]["timeout"], 180)

    def test_glm_stream_uses_same_settings_and_ignores_reasoning_deltas(self):
        client = self.client()
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "reasoning only"}}]},
            {"choices": [{"delta": {"content": '{"ok":'}}]},
            {"choices": [{"delta": {"content": 'true}'}}]},
            {"choices": [], "usage": {"total_tokens": 10}},
        ]
        data = "\n".join("data: " + json.dumps(item) for item in chunks) + "\ndata: [DONE]\n"
        deltas = []
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data.encode())) as request:
            self.assertEqual(client.generate_json_stream("Return an object", "OK", on_delta=deltas.append), {"ok": True})
        body = json.loads(request.call_args[0][0].data)
        self.assertTrue(body.pop("stream"))
        self.assertEqual(body, client._chat_body("Return an object", "OK"))
        self.assertEqual("".join(deltas), '{"ok":true}')

    def test_bigmodel_never_falls_back_to_deepseek_credential(self):
        client = self.client(ZHIPU_API_KEY="")
        self.assertFalse(client.enabled)
        with patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(LLMError, "ZHIPU_API_KEY"):
                client.generate_json("system", "user")
            request.assert_not_called()

    def test_generic_key_override_and_explicit_empty_disable(self):
        self.assertEqual(self.client(LLM_API_KEY="explicit-key").api_key, "explicit-key")
        self.assertFalse(self.client(LLM_API_KEY="").enabled)
        self.assertEqual(self.client(ZHIPU_API_KEY="", ZAI_API_KEY="zai-key").api_key, "zai-key")

    def test_flash_cannot_disable_thinking(self):
        client = self.client(LLM_THINKING_ENABLED="false")
        self.assertEqual(client._chat_body("system", "user")["thinking"]["type"], "enabled")

    def test_glm53_uses_existing_endpoint_and_required_thinking(self):
        client = self.client(LLM_MODEL="glm-5.3", LLM_THINKING_ENABLED="false")
        body = client._chat_body("system", "user", stream=True)
        self.assertEqual(body["model"], "glm-5.3")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertTrue(body["stream"])
        self.assertEqual(client.base_url, "https://open.bigmodel.cn/api/paas/v4")

    def test_existing_deepseek_settings_remain_supported(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "legacy-key", "LLM_BASE_URL": "https://api.deepseek.com", "LLM_MODEL": "deepseek-v4-flash"}, clear=True):
            client = OpenAICompatibleClient()
        body = client._chat_body("system", "user")
        self.assertEqual(client.api_key, "legacy-key")
        self.assertEqual(body["temperature"], 0.2)
        self.assertNotIn("top_p", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("thinking", body)

    def test_invalid_sampling_settings_fail_before_network(self):
        for setting in ({"LLM_TEMPERATURE": "-1"}, {"LLM_TOP_P": "1.2"}, {"LLM_MAX_TOKENS": "-100"}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.client(**setting)

    def test_json_can_use_stream_and_progress_never_contains_reasoning_text(self):
        client = self.client(LLM_STREAM_JSON="true")
        events = [
            {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
            {"choices": [{"delta": {"content": '{"ok":true}'}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
        ]
        data = "\n".join("data: " + json.dumps(e) for e in events) + "\ndata: [DONE]\n"
        progress = []
        client.on_stream_progress = progress.append
        metadata = {}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data.encode())) as request:
            self.assertEqual(client.generate_json("system", "user"), {"ok": True})
        self.assertTrue(json.loads(request.call_args[0][0].data)["stream"])
        self.assertEqual(progress[-1]["phase"], "complete")
        self.assertEqual(progress[-1]["reasoning_chars"], len("private reasoning"))
        self.assertNotIn("private reasoning", json.dumps(progress))
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data.encode())):
            client._generate_json_stream_impl("system", "user", metadata=metadata)
        self.assertEqual(metadata["usage"]["total_tokens"], 30)

    def test_stream_rejects_cutoff_error_and_truncation_even_with_valid_json(self):
        client = self.client()
        for tail, reason in [("", None), ("\ndata: [DONE]\n", "length")]:
            data = "data: " + json.dumps({"choices": [{"delta": {"content": '{}'}, "finish_reason": reason}]}) + "\n" + tail
            with patch("urllib.request.urlopen", return_value=io.BytesIO(data.encode())):
                with self.assertRaises(LLMError):
                    client.generate_json_stream("system", "user")
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b'data: {"error":{"message":"failed"}}\n')):
            with self.assertRaisesRegex(LLMError, "error event"):
                client.generate_json_stream("system", "user")

    def test_timeout_is_a_model_error_in_both_transports(self):
        for stream in [False, True]:
            client = self.client(LLM_STREAM_JSON=str(stream).lower())
            with patch("urllib.request.urlopen", side_effect=TimeoutError("read timed out")):
                with self.assertRaisesRegex(LLMError, "timed out"):
                    client.generate_json("system", "user")

    def test_json_fence_variants_and_embedded_markers(self):
        raw = '{"text":"keep ``` inside a value","ok":true}'
        for content in [raw, raw + '\n```', raw + '\n``', raw + '\r\n  ``  \r\n',
                        '```json\n' + raw + '\n``', '```json\n' + raw + '\n```',
                        '```\n' + raw + '\n```', '```json\n' + raw]:
            with self.subTest(content=content):
                self.assertEqual(OpenAICompatibleClient._parse_json_content(content), json.loads(raw))

    def test_json_parser_does_not_discard_extra_answers_or_prose(self):
        for content in ['{}\n{}', '{}\n{}\n```', '{}\nexplanation\n```',
                        'Here is the answer:\n{}', '```python\n{}\n```',
                        '{"unfinished":', '[]', 'null', '{}\n````', '{}\n`',
                        '{}\n``\nexplanation', '{}\n{}\n``', '{}\n```\n``',
                        '{"unfinished":\n``', '{} ``', '{}```', '{"findings": []}\n``补充说明：另一条评审结论']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                OpenAICompatibleClient._parse_json_content(content)

    def test_stream_captures_final_response_before_parse_and_recovers_closing_fence(self):
        client = self.client(LLM_STREAM_JSON="true")
        captured = []
        client.on_json_response = captured.append
        for content, valid in [('{"findings":[]}\n```', True), ('{"findings":[]}\n``', True), ('{}\n{}', False)]:
            event = {"choices": [{"delta": {"content": content, "reasoning_content": "not captured"}, "finish_reason": "stop"}]}
            data = 'data: ' + json.dumps(event) + '\ndata: [DONE]\n'
            with patch("urllib.request.urlopen", return_value=io.BytesIO(data.encode())):
                if valid:
                    self.assertEqual(client.generate_json("system", "user"), {"findings": []})
                else:
                    with self.assertRaises(LLMError):
                        client.generate_json("system", "user")
            self.assertEqual(captured[-1], content)

    def test_nonstream_recovers_isolated_double_backticks_without_changing_values(self):
        client = self.client()
        expected = {"cases": [{"id": "C1", "text": "literal `` and ``` inside JSON"}]}
        response = {"choices": [{"message": {"content": json.dumps(expected) + '\n``'}}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())):
            self.assertEqual(client.generate_json("system", "user"), expected)

    def test_length_finish_reason_is_a_typed_failure_even_with_parseable_json(self):
        for stream in [False, True]:
            client = self.client(LLM_STREAM_JSON="true" if stream else "false")
            if stream:
                event = {"choices": [{"delta": {"content": '{"findings":[]}'}, "finish_reason": "length"}]}
                response = ('data: ' + json.dumps(event) + '\ndata: [DONE]\n').encode()
            else:
                response = json.dumps({"choices": [{"message": {"content": '{"findings":[]}'},
                                                     "finish_reason": "length"}]}).encode()
            with patch("urllib.request.urlopen", return_value=io.BytesIO(response)):
                with self.assertRaises(LLMError) as caught:
                    client.generate_json("system", "user")
                self.assertEqual(caught.exception.code, "output_limit")


if __name__ == "__main__":
    unittest.main()
