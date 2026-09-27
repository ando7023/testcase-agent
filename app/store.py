import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    AdaptiveMemoryRecord,
    KnowledgeDocument,
    ProjectState,
    ScenarioRule,
    ScenarioTemplate,
    utc_now_iso,
)


class JsonStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.projects_dir = root / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_file = root / "knowledge.json"
        self.knowledge_index_file = root / "knowledge_index.sqlite3"
        self.memory_file = root / "memory.json"
        self.adaptive_memory_file = root / "adaptive_memory.json"
        self.memory_index_file = root / "memory_index.sqlite3"
        self.memory_migration_file = root / "memory_migration.json"
        self.badcase_file = root / "badcases.json"
        self.scenario_rules_file = root / "scenario_rules.json"
        self.scenario_templates_file = root / "scenario_templates.json"
        self._lock = threading.RLock()
        self._project_leases = {}
        self.tracer = None

    @contextmanager
    def project_lease(self, project_id):
        """Serialize supervised runs per project in this single-process service."""
        with self._lock:
            lease = self._project_leases.setdefault(project_id, threading.Lock())
        if not lease.acquire(blocking=False):
            raise ValueError("This project is already running; wait for the current operation")
        try:
            yield
        finally:
            lease.release()

    def save_agent_run(self, run):
        from .supervisor_models import SupervisorRun
        run = SupervisorRun.model_validate(run)
        run.updated_at = utc_now_iso()
        with self._lock:
            folder = self.root / "agent_runs"
            folder.mkdir(exist_ok=True)
            self._write_json(folder / (run.id + ".json"), run.model_dump())

    def get_agent_run(self, project_id, run_id):
        from .supervisor_models import SupervisorRun
        if not re.fullmatch(r"AR-[a-f0-9]{32}", run_id):
            return None
        payload = self._read_json(self.root / "agent_runs" / (run_id + ".json"), None)
        if not payload or payload.get("project_id") != project_id:
            return None
        return SupervisorRun.model_validate(payload)

    def list_agent_runs(self, project_id):
        from .supervisor_models import SupervisorRun
        records = []
        for path in (self.root / "agent_runs").glob("AR-*.json"):
            payload = self._read_json(path, {})
            if payload.get("project_id") == project_id:
                records.append(SupervisorRun.model_validate(payload))
        return sorted(records, key=lambda item: (item.created_at, item.id), reverse=True)

    def set_tracer(self, tracer: Any) -> None:
        self.tracer = tracer

    def create_project(
        self,
        title: str,
        requirement: str,
        context: str = "",
        source_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> ProjectState:
        project = ProjectState(
            id=uuid.uuid4().hex[:12],
            title=title,
            requirement=requirement,
            context=context,
            source_documents=source_documents or [],
        )
        self.save_project(project)
        return project

    def save_project(self, project: ProjectState) -> None:
        project.updated_at = utc_now_iso()
        span_context = (
            self.tracer.span(
                "store.save_project",
                kind="storage",
                attributes={"project_id": project.id, "phase": project.phase},
            )
            if self.tracer
            else nullcontext(None)
        )
        with span_context as span:
            with self._lock:
                self._write_json(
                    self.projects_dir / (project.id + ".json"),
                    project.model_dump(),
                )
            if span:
                span.output_summary = "project persisted"

    def attach_trace_run(self, project_id: str, trace_id: str) -> None:
        with self._lock:
            project = self.get_project(project_id)
            if not project or trace_id in project.trace_run_ids:
                return
            project.trace_run_ids.append(trace_id)
            if len(project.trace_run_ids) > 100:
                project.trace_run_ids = project.trace_run_ids[-100:]
            self.save_project(project)

    def get_project(self, project_id: str) -> Optional[ProjectState]:
        path = self.projects_dir / (project_id + ".json")
        if not path.exists():
            return None
        return ProjectState.model_validate(self._read_json(path, {}))

    def list_projects(self) -> List[ProjectState]:
        projects = [ProjectState.model_validate(self._read_json(path, {})) for path in self.projects_dir.glob("*.json")]
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def list_knowledge(self) -> List[KnowledgeDocument]:
        payload = self._read_json(self.knowledge_file, [])
        documents = []
        for item in payload:
            normalized = dict(item)
            metadata = dict(normalized.get("metadata") or {})
            if normalized.get("id", "").startswith("KB-") and not metadata.get("ticket_type"):
                metadata.update({"domain": "ticket", "ticket_type": "COMMON"})
                normalized["metadata"] = metadata
            documents.append(KnowledgeDocument.model_validate(normalized))
        return documents

    def add_knowledge(
        self,
        title: str,
        content: str,
        doc_type: str,
        tags: List[str],
        ticket_type: str = "COMMON",
        source: str = "",
        section: str = "",
        page: Optional[int] = None,
        version: str = "1.0",
        effective_at: str = "",
        expires_at: str = "",
    ) -> KnowledgeDocument:
        documents = self.list_knowledge()
        document = KnowledgeDocument(
            id="KB-" + uuid.uuid4().hex[:8],
            title=title,
            content=content,
            doc_type=doc_type,
            tags=tags,
            source=source,
            section=section,
            page=page,
            version=version,
            effective_at=effective_at,
            expires_at=expires_at,
            metadata={"domain": "ticket", "ticket_type": ticket_type},
        )
        documents.append(document)
        with self._lock:
            self._write_json(self.knowledge_file, [item.model_dump() for item in documents])
        return document

    def get_knowledge(self, document_id: str) -> Optional[KnowledgeDocument]:
        return next(
            (item for item in self.list_knowledge() if item.id == document_id), None
        )

    @staticmethod
    def _same_knowledge(
        previous: KnowledgeDocument, candidate: KnowledgeDocument
    ) -> bool:
        previous_payload = previous.model_dump()
        candidate_payload = candidate.model_dump()
        previous_payload.pop("updated_at", None)
        candidate_payload.pop("updated_at", None)
        return previous_payload == candidate_payload

    def upsert_knowledge(self, payloads: List[Dict[str, Any]]) -> int:
        documents = {document.id: document for document in self.list_knowledge()}
        changed = 0
        for payload in payloads:
            normalized = dict(payload)
            previous = documents.get(str(normalized.get("id", "")))
            if previous:
                normalized["created_at"] = previous.created_at
                candidate = KnowledgeDocument.model_validate(normalized)
                if self._same_knowledge(previous, candidate):
                    continue
                candidate.updated_at = utc_now_iso()
            else:
                candidate = KnowledgeDocument.model_validate(normalized)
            documents[candidate.id] = candidate
            changed += 1
        if changed:
            self._write_knowledge(documents.values())
        return changed

    def replace_source_knowledge(
        self, source_id: str, payloads: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Replace one source snapshot and retain removed chunks as inactive history."""
        if not source_id:
            raise ValueError("source_id is required")
        incoming = [
            KnowledgeDocument.model_validate(payload) for payload in payloads
        ]
        if any(item.source_id != source_id for item in incoming):
            raise ValueError("All incoming chunks must belong to the same source")
        with self._lock:
            documents = {
                document.id: document for document in self.list_knowledge()
            }
            incoming_ids = {item.id for item in incoming}
            previous_active = [
                item
                for item in documents.values()
                if item.source_id == source_id
                and item.status == "active"
                and item.id not in incoming_ids
            ]
            for candidate in incoming:
                if candidate.id in documents:
                    continue
                predecessors = [
                    item
                    for item in previous_active
                    if item.chunk_level == candidate.chunk_level
                    and item.section == candidate.section
                    and item.page == candidate.page
                ]
                if predecessors:
                    candidate.supersedes = [item.id for item in predecessors]
                    candidate.chunk_version = (
                        max(item.chunk_version for item in predecessors) + 1
                    )

            invalidated_ids: List[str] = []
            changed = 0

            for document in list(documents.values()):
                if (
                    document.source_id == source_id
                    and document.status == "active"
                    and document.id not in incoming_ids
                ):
                    replacements = [
                        item.id
                        for item in incoming
                        if item.chunk_level == document.chunk_level
                        and item.section == document.section
                        and item.page == document.page
                    ]
                    document.status = "inactive"
                    document.superseded_by = replacements
                    document.updated_at = utc_now_iso()
                    document.metadata = dict(document.metadata)
                    document.metadata["invalidation_reason"] = "source_rechunked"
                    documents[document.id] = document
                    invalidated_ids.append(document.id)
                    changed += 1

            for candidate in incoming:
                previous = documents.get(candidate.id)
                if previous:
                    candidate.created_at = previous.created_at
                    if self._same_knowledge(previous, candidate):
                        continue
                    candidate.updated_at = utc_now_iso()
                documents[candidate.id] = candidate
                changed += 1

            if changed:
                self._write_knowledge(documents.values())
                self._delete_knowledge_vectors(invalidated_ids)
        return {
            "changed": changed,
            "invalidated": len(invalidated_ids),
            "invalidated_ids": invalidated_ids,
        }

    def apply_knowledge_mutation(
        self,
        payloads: List[Dict[str, Any]],
        invalidate_vector_ids: List[str],
    ) -> int:
        with self._lock:
            documents = {
                document.id: document for document in self.list_knowledge()
            }
            changed = 0
            for payload in payloads:
                candidate = KnowledgeDocument.model_validate(payload)
                previous = documents.get(candidate.id)
                if previous:
                    candidate.created_at = previous.created_at
                    if self._same_knowledge(previous, candidate):
                        continue
                candidate.updated_at = utc_now_iso()
                documents[candidate.id] = candidate
                changed += 1
            if changed:
                self._write_knowledge(documents.values())
                self._delete_knowledge_vectors(invalidate_vector_ids)
            return changed

    def _write_knowledge(self, documents) -> None:
        ordered = sorted(
            documents,
            key=lambda item: (
                item.source_id or item.source,
                0 if item.chunk_level == "parent" else 1,
                item.chunk_index,
                item.id,
            ),
        )
        self._write_json(
            self.knowledge_file, [item.model_dump() for item in ordered]
        )

    def _delete_knowledge_vectors(self, document_ids: List[str]) -> None:
        if not document_ids or not self.knowledge_index_file.exists():
            return
        with sqlite3.connect(str(self.knowledge_index_file), timeout=10) as connection:
            placeholders = ",".join("?" for _ in document_ids)
            connection.execute(
                "DELETE FROM knowledge_vectors WHERE document_id IN ({})".format(
                    placeholders
                ),
                document_ids,
            )

    def add_badcase(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record.setdefault("created_at", utc_now_iso())
        with self._lock:
            records = self._read_json(self.badcase_file, [])
            records.append(record)
            self._write_json(self.badcase_file, records)
        return record

    def list_badcases(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        records = self._read_json(self.badcase_file, [])
        if project_id:
            records = [record for record in records if record.get("project_id") == project_id]
        return records

    def list_scenario_rules(self) -> List[ScenarioRule]:
        return [
            ScenarioRule.model_validate(item)
            for item in self._read_json(self.scenario_rules_file, [])
        ]

    def upsert_scenario_rules(self, incoming: List[ScenarioRule]) -> List[ScenarioRule]:
        rules = {rule.id: rule for rule in self.list_scenario_rules()}
        for candidate in incoming:
            previous = rules.get(candidate.id)
            if previous:
                sources = list(dict.fromkeys(previous.sources or [previous.source]))
                if candidate.source not in sources:
                    sources.append(candidate.source)
                previous.sources = sources
                previous.support_count = len(sources)
                rules[candidate.id] = previous
            else:
                candidate.sources = [candidate.source]
                candidate.support_count = 1
                rules[candidate.id] = candidate
        ordered = sorted(rules.values(), key=lambda item: (item.ticket_type, item.case_type, item.id))
        with self._lock:
            self._write_json(self.scenario_rules_file, [item.model_dump() for item in ordered])
        return ordered

    def list_scenario_templates(self) -> List[ScenarioTemplate]:
        return [
            ScenarioTemplate.model_validate(item)
            for item in self._read_json(self.scenario_templates_file, [])
        ]

    def save_scenario_templates(self, templates: List[ScenarioTemplate]) -> None:
        with self._lock:
            self._write_json(
                self.scenario_templates_file,
                [item.model_dump() for item in templates],
            )

    def get_memory(self) -> Dict[str, Any]:
        return self._read_json(self.memory_file, {"preferences": {}, "learned_rules": []})

    def list_memory_records(self, include_inactive: bool = False) -> List[AdaptiveMemoryRecord]:
        records = [
            AdaptiveMemoryRecord.model_validate(item)
            for item in self._read_json(self.adaptive_memory_file, [])
        ]
        return records if include_inactive else [item for item in records if item.status == "active"]

    @contextmanager
    def memory_transaction(self):
        """Serialize memory read/modify/write operations within this store."""
        with self._lock:
            yield

    def save_memory_records(self, records: List[AdaptiveMemoryRecord]) -> None:
        with self._lock:
            self._write_json(self.adaptive_memory_file, [item.model_dump() for item in records])

    @staticmethod
    def memory_scope(record: AdaptiveMemoryRecord):
        return tuple(getattr(record, key) for key in (
            "user_id", "project_id", "agent_id", "run_id", "ticket_type", "memory_type"
        ))

    def add_memory_records(self, incoming: List[AdaptiveMemoryRecord]) -> Dict[str, Any]:
        with self._lock:
            records = self.list_memory_records(include_inactive=True)
            hashes = {item.content_hash for item in records if item.content_hash and item.status == "active"}
            added: List[AdaptiveMemoryRecord] = []
            duplicate_ids: List[str] = []
            for candidate in incoming:
                if candidate.content_hash and candidate.content_hash in hashes:
                    duplicate = next(
                        item for item in records if item.content_hash == candidate.content_hash and item.status == "active"
                    )
                    duplicate_ids.append(duplicate.id)
                    continue
                if candidate.fact_key and any(
                    item.fact_key == candidate.fact_key
                    and self.memory_scope(item) == self.memory_scope(candidate)
                    and item.status == "active" for item in records
                ):
                    candidate.status = "pending_conflict"
                    candidate.invalidation_reason = "Explicit revision required for this fact_key"
                records.append(candidate)
                added.append(candidate)
                if candidate.content_hash and candidate.status == "active":
                    hashes.add(candidate.content_hash)
            if added:
                self._write_json(
                    self.adaptive_memory_file,
                    [item.model_dump() for item in records],
                )
        return {
            "added": len(added),
            "records": added,
            "duplicate_ids": duplicate_ids,
        }

    def touch_memory_records(self, memory_ids: List[str]) -> None:
        if not memory_ids:
            return
        with self._lock:
            records = self.list_memory_records(include_inactive=True)
            changed = False
            for record in records:
                if record.id not in memory_ids:
                    continue
                record.access_count += 1
                record.last_accessed_at = utc_now_iso()
                record.updated_at = record.last_accessed_at
                changed = True
            if changed:
                self._write_json(
                    self.adaptive_memory_file,
                    [item.model_dump() for item in records],
                )

    def add_memory_rule(self, rule: str, ticket_type: str = "COMMON") -> Dict[str, Any]:
        memory = self.get_memory()
        scoped_rules = memory.setdefault("scoped_rules", [])
        entry = {"rule": rule, "ticket_type": ticket_type}
        if not any(
            item.get("rule") == rule and item.get("ticket_type", "COMMON") == ticket_type
            for item in scoped_rules
            if isinstance(item, dict)
        ):
            scoped_rules.append(entry)
        with self._lock:
            self._write_json(self.memory_file, memory)
        return memory

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)
