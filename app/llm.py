import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def load_local_env(path: Optional[Path] = None) -> None:
    """Load project-local secrets without overriding process environment."""
    env_path = path or (Path(__file__).resolve().parent.parent / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_local_env()


class LLMError(RuntimeError):
    def __init__(self, message, code="request_failed"):
        super().__init__(message)
        self.code = code


class OpenAICompatibleClient:
    """Small OpenAI-compatible client with local secret loading."""

    def __init__(self, tracer: Any = None) -> None:
        self.tracer = tracer
        self.base_url = os.getenv(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.is_bigmodel = urllib.parse.urlparse(self.base_url).hostname == "open.bigmodel.cn"
        provider_key = (
            os.getenv("ZHIPU_API_KEY") or os.getenv("ZAI_API_KEY", "")
            if self.is_bigmodel else os.getenv("DEEPSEEK_API_KEY", "")
        )
        # An explicitly empty generic key still disables calls (e.g. offline tests).
        # Never send a legacy DeepSeek key to the BigModel endpoint.
        self.api_key = os.getenv("LLM_API_KEY", provider_key).strip()
        self.model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
        self.stream_json = os.getenv("LLM_STREAM_JSON", "").lower() in {"1", "true", "yes"}
        self.on_stream_progress = None
        self.on_json_response = None
        self._diagnostic_context = ContextVar("llm_diagnostics", default=None)
        self._started_context = ContextVar("llm_request_started", default=0.0)
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        top_p = os.getenv("LLM_TOP_P", "").strip()
        self.top_p = float(top_p) if top_p else None
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "0") or "0")
        if not 0 <= self.temperature <= 2 or (self.top_p is not None and not 0 < self.top_p <= 1) or self.max_tokens < 0:
            raise ValueError("Invalid LLM_TEMPERATURE, LLM_TOP_P or LLM_MAX_TOKENS")
        self.thinking_enabled = os.getenv(
            "LLM_THINKING_ENABLED", ""
        ).lower() in {"1", "true", "yes", "enabled"}
        self.disabled = os.getenv('LLM_DISABLED', '').lower() in {
            '1', 'true', 'yes', 'enabled'
        }

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and not self.disabled

    @property
    def last_call_diagnostics(self):
        return self._diagnostic_context.get() or {}

    @last_call_diagnostics.setter
    def last_call_diagnostics(self, value):
        self._diagnostic_context.set(value)

    def effective_settings(self):
        return {"model": self.model, "stream": self.stream_json, "reasoning_effort": self.reasoning_effort,
                "timeout_seconds": self.timeout, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "top_p": self.top_p,
                "thinking_enabled": self.thinking_enabled or (self.is_bigmodel and self.model.lower() in
                    {"glm-5.3", "glm-5.3-flash", "glm-5.3-flashx"})}

    @contextmanager
    def _diagnostics(self, span, stream):
        self._started_context.set(time.perf_counter())
        self.last_call_diagnostics = {"stream": stream, "phase": "awaiting_headers",
            "connected_ms": None, "first_event_ms": None, "first_content_ms": None,
            "last_receive_ms": None, "event_count": 0, "reasoning_chars": 0, "content_chars": 0}
        try:
            yield
            self.last_call_diagnostics["phase"] = "complete"
        except LLMError as exc:
            self.last_call_diagnostics["error_code"] = exc.code
            raise
        finally:
            self.last_call_diagnostics["elapsed_ms"] = self._elapsed_ms()
            if span:
                span.attributes.update(self.effective_settings())
                span.attributes.update(self.last_call_diagnostics)

    def _elapsed_ms(self):
        return round((time.perf_counter() - self._started_context.get()) * 1000)

    def _received(self, event=False, content=False):
        diagnostics = self.last_call_diagnostics
        diagnostics["last_receive_ms"] = self._elapsed_ms()
        for flag, key in ((event, "first_event_ms"), (content, "first_content_ms")):
            if flag and diagnostics[key] is None:
                diagnostics[key] = diagnostics["last_receive_ms"]
        diagnostics["phase"] = "receiving"

    def _chat_body(self, system_prompt, user_prompt, schema=None, stream=False):
        if not self.enabled:
            key_name = "ZHIPU_API_KEY (or ZAI_API_KEY / LLM_API_KEY)" if self.is_bigmodel else "LLM_API_KEY or DEEPSEEK_API_KEY"
            raise LLMError("{} is not configured, or LLM_DISABLED is true".format(key_name))
        system = system_prompt + "\nReturn only valid JSON in the final answer."
        if schema:
            system += "\nReturn JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        # GLM-5.3 and Flash/FlashX require thinking.type=enabled.
        glm_flash = self.is_bigmodel and self.model.lower() in {"glm-5.3-flash", "glm-5.3-flashx"}
        glm_53 = self.is_bigmodel and self.model.lower() == "glm-5.3"
        if self.thinking_enabled or glm_flash or glm_53:
            body["thinking"] = {"type": "enabled"}
            if glm_flash:
                body["thinking"]["clear_thinking"] = False
        if stream:
            body["stream"] = True
        return body

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.stream_json:
            return self.generate_json_stream(system_prompt, user_prompt, schema)
        with self._trace_span(
            "llm.chat",
            kind="llm",
            attributes={
                "model": self.model,
                "stream": False,
                "schema": bool(schema),
                "provider": self.base_url,
            },
            input_value={"system": system_prompt, "user": user_prompt},
        ) as span:
            with self._diagnostics(span, False):
                result, usage = self._generate_json_impl(system_prompt, user_prompt, schema)
            if span:
                span.output_summary = json.dumps(
                    result, ensure_ascii=False, default=str
                )[:2000]
                span.usage = usage
            return result

    def _generate_json_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Any:
        body = self._chat_body(system_prompt, user_prompt, schema)
        payload = self._post_json("/chat/completions", body)
        try:
            finish_reason = payload["choices"][0].get("finish_reason")
            if finish_reason and finish_reason != "stop":
                raise LLMError("LLM ended without a complete answer: finish_reason={}".format(finish_reason),
                               code="output_limit" if finish_reason == "length" else "incomplete_response")
            content = payload["choices"][0]["message"]["content"]
            if self.on_json_response and isinstance(content, str):
                self.on_json_response(content)
            return self._parse_json_content(content), self._usage(
                payload, system_prompt + user_prompt, str(content)
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("LLM returned invalid JSON: {}".format(exc), code="invalid_json")

    def generate_json_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Stream OpenAI-compatible chat deltas and parse the final JSON."""
        with self._trace_span(
            "llm.chat.stream",
            kind="llm",
            attributes={
                "model": self.model,
                "stream": True,
                "schema": bool(schema),
                "provider": self.base_url,
            },
            input_value={"system": system_prompt, "user": user_prompt},
        ) as span:
            metadata = {}
            with self._diagnostics(span, True):
                result, content, first_token_ms = self._generate_json_stream_impl(
                    system_prompt, user_prompt, schema, on_delta, metadata)
            if span:
                span.output_summary = content[:2000]
                span.usage = self._usage(
                    metadata, system_prompt + user_prompt, content
                )
                span.attributes["first_token_ms"] = first_token_ms
                span.attributes.update({key: metadata[key] for key in
                                        ("event_count", "reasoning_chars", "content_chars")})
            return result

    def _generate_json_stream_impl(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        body = self._chat_body(system_prompt, user_prompt, schema, stream=True)
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        fragments: List[str] = []
        request_started = time.perf_counter()
        first_token_ms = 0
        metadata = metadata if metadata is not None else {}
        metadata.update(event_count=0, reasoning_chars=0, content_chars=0)
        complete = False
        finish_reason = None

        def progress(phase):
            if self.on_stream_progress:
                self.on_stream_progress({"phase": phase, "seconds": round(time.perf_counter() - request_started, 2),
                                         **{key: self.last_call_diagnostics.get(key) for key in
                                            ("connected_ms", "first_event_ms", "first_content_ms", "last_receive_ms")},
                                         **{key: metadata[key] for key in ("event_count", "reasoning_chars", "content_chars")}})

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.last_call_diagnostics.update(connected_ms=self._elapsed_ms(), phase="awaiting_first_event")
                progress("connected")
                for raw_line in response:
                    self._received()
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        complete = True
                        break
                    try:
                        event = json.loads(data)
                        self._received(event=True)
                        if event.get("error"):
                            raise LLMError("LLM stream returned an error event")
                        metadata["event_count"] += 1
                        self.last_call_diagnostics["event_count"] = metadata["event_count"]
                        if event.get("usage"):
                            metadata["usage"] = event["usage"]
                        choices = event.get("choices") or []
                        if not choices:
                            progress("receiving")
                            continue
                        choice = choices[0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        metadata["reasoning_chars"] += len(delta.get("reasoning_content") or "")
                        metadata["content_chars"] += len(content)
                        self.last_call_diagnostics.update({key: metadata[key] for key in
                            ("event_count", "reasoning_chars", "content_chars")})
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue
                    if content:
                        self._received(content=True)
                        if not first_token_ms:
                            first_token_ms = int(
                                (time.perf_counter() - request_started) * 1000
                            )
                        fragments.append(content)
                        if on_delta:
                            on_delta(content)
                    progress("receiving")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMError(
                "LLM request failed with HTTP {}: {}".format(exc.code, detail)
            )
        except TimeoutError as exc:
            progress("timeout")
            raise LLMError("LLM stream read timed out (timeout={}s); events={}, reasoning_chars={}, content_chars={}".format(
                self.timeout, metadata["event_count"], metadata["reasoning_chars"], metadata["content_chars"]), code="timeout") from exc
        except (urllib.error.URLError, ValueError) as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise LLMError("LLM connection timed out", code="timeout") from exc
            raise LLMError("LLM request failed: {}".format(exc))
        if finish_reason and finish_reason != "stop":
            raise LLMError("LLM stream ended without a complete answer: finish_reason={}".format(finish_reason),
                           code="output_limit" if finish_reason == "length" else "incomplete_response")
        if not complete and finish_reason != "stop":
            raise LLMError("LLM stream disconnected before completion", code="incomplete_response")
        progress("complete")
        try:
            self.last_call_diagnostics["phase"] = "parsing"
            content = "".join(fragments)
            if self.on_json_response:
                self.on_json_response(content)
            return self._parse_json_content(content), content, first_token_ms
        except (TypeError, ValueError) as exc:
            raise LLMError("LLM returned invalid streamed JSON: {}".format(exc), code="invalid_json")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        with self._trace_span(
            "llm.embedding",
            kind="embedding",
            attributes={
                "model": self.embedding_model,
                "provider": self.base_url,
                "batch_size": len(texts),
            },
            input_value={"texts": texts},
        ) as span:
            vectors, usage = self._create_embeddings_impl(texts)
            if span:
                span.output_summary = "{} vectors, dimension {}".format(
                    len(vectors), len(vectors[0]) if vectors else 0
                )
                span.usage = usage
            return vectors

    def _create_embeddings_impl(self, texts: List[str]) -> Any:
        if not self.enabled:
            raise LLMError("LLM_API_KEY or DEEPSEEK_API_KEY is not configured")
        payload = self._post_json(
            "/embeddings",
            {"model": self.embedding_model, "input": texts},
        )
        try:
            ordered = sorted(payload["data"], key=lambda item: item["index"])
            vectors = [
                [float(value) for value in item["embedding"]]
                for item in ordered
            ]
            return vectors, self._usage(payload, "\n".join(texts), "")
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError("Embedding response is invalid: {}".format(exc))

    def _trace_span(self, name: str, **kwargs: Any):
        if not self.tracer:
            return nullcontext(None)
        return self.tracer.span(name, **kwargs)

    @staticmethod
    def _usage(
        payload: Dict[str, Any], input_text: str, output_text: str
    ) -> Dict[str, Any]:
        usage = payload.get("usage") or {}
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        estimated = input_tokens is None or output_tokens is None
        if input_tokens is None:
            input_tokens = max(1, len(input_text) // 4)
        if output_tokens is None:
            output_tokens = max(0, len(output_text) // 4)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "estimated": estimated,
        }

    def _post_json(
        self, endpoint: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                # Non-stream responses have no token events; record received
                # body chunks without pretending they are generated tokens.
                tracked = bool(self.last_call_diagnostics) and endpoint == "/chat/completions"
                if tracked:
                    self.last_call_diagnostics.update(connected_ms=self._elapsed_ms(), phase="awaiting_body")
                chunks = []
                read = getattr(response, "read1", response.read)
                while True:
                    chunk = read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if tracked:
                        self._received()
                if tracked:
                    self.last_call_diagnostics["phase"] = "parsing"
                return json.loads(b"".join(chunks).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMError(
                "LLM request failed with HTTP {}: {}".format(exc.code, detail)
            )
        except TimeoutError as exc:
            raise LLMError("LLM read timed out (timeout={}s)".format(self.timeout), code="timeout") from exc
        except (urllib.error.URLError, ValueError) as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise LLMError("LLM connection timed out", code="timeout") from exc
            raise LLMError("LLM request failed: {}".format(exc))

    @staticmethod
    def _parse_json_content(content: Any) -> Dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("message content is not text")
        value = content.strip()
        fence = chr(96) * 3
        if value.startswith(fence):
            lines = value.splitlines()
            if lines and lines[0].strip().lower() in {fence, fence + "json"}:
                lines = lines[1:]
            value = "\n".join(lines).strip()
        # Decode a complete value first. Only tolerate an isolated closing
        # marker (including the two-backtick variant observed in GLM output).
        # Never repair truncated JSON or silently discard a second answer.
        result, end = json.JSONDecoder().raw_decode(value)
        suffix = value[end:]
        if suffix.strip() and not re.fullmatch(r"[ \t]*\r?\n[ \t]*`{2,3}", suffix):
            raise json.JSONDecodeError("Extra data", value, end)
        if not isinstance(result, dict):
            raise ValueError("JSON response must be an object")
        return result
