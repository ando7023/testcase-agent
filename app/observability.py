import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .models import TraceEvent, TraceRun, TraceSpan, utc_now_iso


_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|token|secret)(\s*[:=]\s*)([^,}\]\s]+)"
)


def safe_summary(value: Any, limit: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
    rendered = _SECRET_PATTERN.sub(r"\1\2<redacted>", rendered)
    rendered = " ".join(rendered.split())
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


class TraceManager:
    """Lightweight hierarchical tracing with ContextVar propagation.

    Runs are persisted independently from project JSON so detailed spans do not
    inflate every project read. The manager intentionally has no dependency on
    the application store to avoid tracing its own trace writes.
    """

    def __init__(self, data_root: Path, max_runs: int = 500) -> None:
        self.trace_dir = data_root / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.max_runs = max(20, max_runs)
        self._lock = threading.RLock()
        self._run: ContextVar[Optional[TraceRun]] = ContextVar(
            "caseforge_trace_run", default=None
        )
        self._span_id: ContextVar[str] = ContextVar(
            "caseforge_trace_span", default=""
        )
        self._started: Dict[str, float] = {}

    @property
    def current_run(self) -> Optional[TraceRun]:
        return self._run.get()

    @property
    def current_trace_id(self) -> str:
        run = self.current_run
        return run.trace_id if run else ""

    @contextmanager
    def run(
        self,
        operation: str,
        *,
        project_id: str = "",
        transport: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[TraceRun]:
        existing = self.current_run
        if existing is not None:
            with self.span(
                operation,
                kind="workflow",
                attributes=attributes,
            ):
                yield existing
            return

        trace_id = "TR-" + uuid.uuid4().hex
        root_id = "SP-" + uuid.uuid4().hex[:16]
        run = TraceRun(
            trace_id=trace_id,
            operation=operation,
            project_id=project_id,
            transport=transport,
            root_span_id=root_id,
            attributes=self._clean_attributes(attributes or {}),
        )
        root = TraceSpan(
            id=root_id,
            trace_id=trace_id,
            name=operation,
            kind="request" if transport in {"http", "websocket"} else "workflow",
            attributes=self._clean_attributes(attributes or {}),
        )
        run.spans.append(root)
        self._started[root.id] = time.perf_counter()
        run_token = self._run.set(run)
        span_token = self._span_id.set(root.id)
        try:
            yield run
            if root.status == "running":
                root.status = "success"
            run.status = root.status
        except Exception as exc:
            root.status = "error"
            root.error = safe_summary(exc, 1000)
            run.status = "error"
            raise
        finally:
            self._finish_span(root)
            run.ended_at = root.ended_at
            run.duration_ms = root.duration_ms
            self._span_id.reset(span_token)
            self._run.reset(run_token)
            self._persist(run)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = "internal",
        attributes: Optional[Dict[str, Any]] = None,
        input_value: Any = None,
    ) -> Iterator[Optional[TraceSpan]]:
        run = self.current_run
        if run is None:
            yield None
            return
        span = TraceSpan(
            id="SP-" + uuid.uuid4().hex[:16],
            trace_id=run.trace_id,
            parent_span_id=self._span_id.get(),
            name=name,
            kind=kind,
            input_summary=safe_summary(input_value),
            attributes=self._clean_attributes(attributes or {}),
        )
        run.spans.append(span)
        self._started[span.id] = time.perf_counter()
        token = self._span_id.set(span.id)
        try:
            yield span
            if span.status == "running":
                span.status = "success"
        except Exception as exc:
            span.status = "error"
            span.error = safe_summary(exc, 1000)
            raise
        finally:
            self._finish_span(span)
            self._span_id.reset(token)

    def event(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        run = self.current_run
        if run is None:
            return
        span_id = self._span_id.get() or run.root_span_id
        span = next((item for item in run.spans if item.id == span_id), None)
        if span:
            span.events.append(
                TraceEvent(
                    name=name,
                    attributes=self._clean_attributes(attributes or {}),
                )
            )

    def read(self, trace_id: str) -> Optional[TraceRun]:
        if not re.fullmatch(r"TR-[a-f0-9]{32}", trace_id):
            return None
        path = self.trace_dir / (trace_id + ".json")
        if not path.exists():
            return None
        try:
            return TraceRun.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            return None

    def list_runs(
        self, project_id: str = "", limit: int = 50
    ) -> List[TraceRun]:
        runs: List[TraceRun] = []
        paths = sorted(
            self.trace_dir.glob("TR-*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            try:
                run = TraceRun.model_validate(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue
            if project_id and run.project_id != project_id:
                continue
            runs.append(run)
            if len(runs) >= max(1, min(limit, 200)):
                break
        return runs

    def _finish_span(self, span: TraceSpan) -> None:
        started = self._started.pop(span.id, time.perf_counter())
        span.duration_ms = max(0, int((time.perf_counter() - started) * 1000))
        span.ended_at = utc_now_iso()

    def _persist(self, run: TraceRun) -> None:
        path = self.trace_dir / (run.trace_id + ".json")
        temporary = path.with_suffix(".tmp")
        with self._lock:
            temporary.write_text(
                run.model_dump_json(indent=2), encoding="utf-8"
            )
            temporary.replace(path)
            self._prune()

    def _prune(self) -> None:
        paths = sorted(
            self.trace_dir.glob("TR-*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in paths[self.max_runs :]:
            try:
                path.unlink()
            except OSError:
                pass

    @staticmethod
    def _clean_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in attributes.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("key", "secret", "token", "authorization")):
                cleaned[str(key)] = "<redacted>"
            elif isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[str(key)] = safe_summary(value, 500)
            else:
                cleaned[str(key)] = safe_summary(value, 1000)
        return cleaned
