import hashlib
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .knowledge import (
    HashingEmbeddingProvider,
    KnowledgeBase,
    OpenAIEmbeddingProvider,
)
from .llm import LLMError, OpenAICompatibleClient
from .models import (
    AdaptiveMemoryContext,
    AdaptiveMemoryHit,
    AdaptiveMemoryRecord,
    KnowledgeDocument,
    utc_now_iso,
)
from .store import JsonStore


ENTITY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_./-]{2,}|[\u4e00-\u9fff]{2,12}")


class AdaptiveMemory:
    """Versioned memory with scoped hybrid retrieval and explicit revisions."""

    def __init__(
        self, store: JsonStore, llm: OpenAICompatibleClient, tracer: Any = None
    ) -> None:
        self.store = store
        self.llm = llm
        self.tracer = tracer
        self.default_top_k = max(1, int(os.getenv("MEMORY_TOP_K", "5")))
        self.default_token_budget = max(
            200, int(os.getenv("MEMORY_TOKEN_BUDGET", "1000"))
        )
        self.llm_extraction = os.getenv(
            "MEMORY_LLM_EXTRACTION", "true"
        ).lower() in {"1", "true", "yes", "enabled"}

    def migrate_legacy(self) -> Dict[str, Any]:
        """Idempotent compatibility migration, with quarantine of widened scopes."""
        with self.store.memory_transaction():
            payload = self.store.get_memory()
            fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            marker = self.store._read_json(self.store.memory_migration_file, {})
            if marker.get("version") == 2 and marker.get("legacy_fingerprint") == fingerprint:
                return {"migrated": 0, "quarantined": 0, "converted": 0}
            records = self.store.list_memory_records(include_inactive=True)
            scoped = {
                (" ".join(item.content.split()).lower(), item.ticket_type)
                for item in records
                if any((item.user_id, item.project_id, item.agent_id, item.run_id))
            }
            now = utc_now_iso()
            quarantined = converted = 0
            for item in records:
                identity = (" ".join(item.content.split()).lower(), item.ticket_type)
                if (item.source == "legacy_memory" and item.status == "active"
                        and not any((item.user_id, item.project_id, item.agent_id, item.run_id))
                        and identity in scoped):
                    item.status = "inactive"
                    item.invalidation_reason = "legacy_scope_ambiguous"
                    item.updated_at = now
                    quarantined += 1
                if (item.source in {"module_conversation", "case_conversation"}
                        and item.run_id.startswith(("MV-", "CV-"))
                        and item.source_ref == item.run_id
                        and not item.source_run_id and not item.source_version_id):
                    item.source_run_id = item.source_run_id or item.run_id
                    item.source_version_id = item.source_version_id or item.source_ref or item.run_id
                    item.run_id = ""
                    item.updated_at = now
                    converted += 1
                item.content_hash = self._digest(item.content, item.model_dump())
            entries = []
            for rule in payload.get("learned_rules", []):
                if not isinstance(rule, str) or not rule.strip():
                    continue
                match = re.match(r"^\[badcase沉淀\]\[([^]]+)\]\s*(.*)$", rule.strip())
                entries.append((match.group(2), match.group(1)) if match else (rule.strip(), "COMMON"))
            for item in payload.get("scoped_rules", []):
                if isinstance(item, dict) and str(item.get("rule", "")).strip():
                    entries.append((str(item["rule"]), str(item.get("ticket_type") or "COMMON")))
            # Include inactive hashes so revoked legacy facts are never resurrected.
            hashes = {item.content_hash for item in records}
            migrated = 0
            for content, ticket_type in entries:
                if (" ".join(content.split()).lower(), ticket_type) in scoped:
                    continue
                candidate = self._record(content, memory_type="team_rule", ticket_type=ticket_type,
                                         source="legacy_memory", importance=0.8)
                if candidate.content_hash not in hashes:
                    records.append(candidate)
                    hashes.add(candidate.content_hash)
                    migrated += 1
            self.store.save_memory_records(records)
            self.store._write_json(self.store.memory_migration_file, {
                "version": 2, "legacy_fingerprint": fingerprint, "completed_at": now,
                "quarantined": quarantined, "converted": converted,
            })
            return {"migrated": migrated, "quarantined": quarantined, "converted": converted}

    def remember(
        self,
        text: str,
        *,
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        ticket_type: str = "COMMON",
        source: str = "conversation",
        source_ref: str = "",
        source_run_id: str = "",
        source_version_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> Dict[str, Any]:
        if not self.tracer:
            return self._remember(
                text, user_id=user_id, project_id=project_id,
                agent_id=agent_id, run_id=run_id, ticket_type=ticket_type,
                source=source, source_ref=source_ref, source_run_id=source_run_id, source_version_id=source_version_id, metadata=metadata,
                infer=infer,
            )
        with self.tracer.span(
            "memory.remember",
            kind="memory",
            attributes={
                "project_id": project_id,
                "agent_id": agent_id,
                "ticket_type": ticket_type,
                "source": source,
                "infer": infer,
            },
            input_value=text,
        ) as span:
            result = self._remember(
                text, user_id=user_id, project_id=project_id,
                agent_id=agent_id, run_id=run_id, ticket_type=ticket_type,
                source=source, source_ref=source_ref, source_run_id=source_run_id, source_version_id=source_version_id, metadata=metadata,
                infer=infer,
            )
            if span:
                span.output_summary = "added={}, extracted={}, duplicates={}".format(
                    result.get("added", 0), result.get("extracted", 0),
                    len(result.get("duplicate_ids", [])),
                )
            return result

    def _remember(
        self,
        text: str,
        *,
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        ticket_type: str = "COMMON",
        source: str = "conversation",
        source_ref: str = "",
        source_run_id: str = "",
        source_version_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"added": 0, "records": [], "duplicate_ids": []}
        facts = self._extract(text) if infer else [self._fallback_fact(text)]
        records = [
            self._record(
                fact.get("content", ""),
                memory_type=str(fact.get("memory_type") or "semantic"),
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
                run_id=run_id,
                ticket_type=ticket_type,
                source=source,
                source_ref=source_ref, source_run_id=source_run_id, source_version_id=source_version_id,
                importance=float(fact.get("importance", 0.6)),
                entities=[str(item) for item in fact.get("entities", [])],
                metadata=metadata or {},
            )
            for fact in facts
            if str(fact.get("content", "")).strip()
        ]
        result = self.store.add_memory_records(records)
        result["extracted"] = len(records)
        return result

    def add_fact(
        self,
        content: str,
        *,
        memory_type: str = "team_rule",
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        ticket_type: str = "COMMON",
        source: str = "manual",
        source_ref: str = "",
        source_run_id: str = "",
        source_version_id: str = "",
        importance: float = 0.8,
        fact_key: str = "",
        valid_from: str = "",
        valid_to: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.tracer:
            return self._add_fact(
                content, memory_type=memory_type, user_id=user_id,
                project_id=project_id, agent_id=agent_id, run_id=run_id,
                ticket_type=ticket_type, source=source, source_ref=source_ref, source_run_id=source_run_id, source_version_id=source_version_id,
                importance=importance, fact_key=fact_key, valid_from=valid_from, valid_to=valid_to, metadata=metadata,
            )
        with self.tracer.span(
            "memory.add_fact",
            kind="memory",
            attributes={
                "memory_type": memory_type,
                "project_id": project_id,
                "agent_id": agent_id,
                "ticket_type": ticket_type,
                "source": source,
            },
            input_value=content,
        ) as span:
            result = self._add_fact(
                content, memory_type=memory_type, user_id=user_id,
                project_id=project_id, agent_id=agent_id, run_id=run_id,
                ticket_type=ticket_type, source=source, source_ref=source_ref, source_run_id=source_run_id, source_version_id=source_version_id,
                importance=importance, fact_key=fact_key, valid_from=valid_from, valid_to=valid_to, metadata=metadata,
            )
            if span:
                span.output_summary = "added={}, duplicates={}".format(
                    result.get("added", 0), len(result.get("duplicate_ids", []))
                )
            return result

    def _add_fact(
        self,
        content: str,
        *,
        memory_type: str = "team_rule",
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        ticket_type: str = "COMMON",
        source: str = "manual",
        source_ref: str = "",
        source_run_id: str = "",
        source_version_id: str = "",
        importance: float = 0.8,
        fact_key: str = "",
        valid_from: str = "",
        valid_to: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self._record(
            content,
            memory_type=memory_type,
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            run_id=run_id,
            ticket_type=ticket_type,
            source=source,
            source_ref=source_ref, source_run_id=source_run_id, source_version_id=source_version_id,
            importance=importance, fact_key=fact_key, valid_from=valid_from, valid_to=valid_to,
            entities=self._entities(content),
            metadata=metadata or {},
        )
        result = self.store.add_memory_records([record])
        result["extracted"] = 1
        return result

    def search(
        self,
        query: str,
        *,
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        ticket_type: str = "COMMON",
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> AdaptiveMemoryContext:
        if not self.tracer:
            return self._search(
                query, user_id=user_id, project_id=project_id,
                agent_id=agent_id, run_id=run_id, ticket_type=ticket_type,
                top_k=top_k, token_budget=token_budget,
            )
        with self.tracer.span(
            "memory.search",
            kind="memory",
            attributes={
                "project_id": project_id,
                "agent_id": agent_id,
                "ticket_type": ticket_type,
                "top_k": top_k or self.default_top_k,
                "token_budget": token_budget or self.default_token_budget,
            },
            input_value=query,
        ) as span:
            result = self._search(
                query, user_id=user_id, project_id=project_id,
                agent_id=agent_id, run_id=run_id, ticket_type=ticket_type,
                top_k=top_k, token_budget=token_budget,
            )
            if span:
                span.output_summary = "{} of {} memories, {} tokens".format(
                    result.selected_count, result.candidate_count,
                    result.selected_tokens,
                )
                span.attributes.update({
                    "selected_count": result.selected_count,
                    "candidate_count": result.candidate_count,
                    "selected_tokens": result.selected_tokens,
                    "token_saving_ratio": result.token_saving_ratio,
                    "embedding_provider": result.embedding_provider,
                })
            return result

    def _search(
        self,
        query: str,
        *,
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
        ticket_type: str = "COMMON",
        top_k: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> AdaptiveMemoryContext:
        started = time.time()
        top_k = top_k or self.default_top_k
        token_budget = token_budget or self.default_token_budget
        records = [
            record
            for record in self.store.list_memory_records()
            if self._is_effective(record) and self._scope_matches(
                record,
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
                run_id=run_id,
                ticket_type=ticket_type,
            )
        ]
        full_tokens = sum(
            self._estimate_tokens(self._format_line(item)) for item in records
        )
        documents = [self._as_document(item) for item in records]
        provider_name = os.getenv("MEMORY_EMBEDDING_PROVIDER", "hashing").lower()
        if provider_name in {"openai", "remote"} and self.llm.enabled:
            provider = OpenAIEmbeddingProvider(self.llm)
        else:
            provider = HashingEmbeddingProvider()
        knowledge_base = KnowledgeBase(
            documents,
            embedding_provider=provider,
            index_path=self.store.memory_index_file,
        )
        ranked = knowledge_base.search_detailed(query, limit=max(top_k * 3, 12))
        by_id = {item.id: item for item in records}
        hits: List[AdaptiveMemoryHit] = []
        for item in ranked:
            record = by_id[item.document.id]
            recency = self._recency_score(record)
            score = min(
                1.0,
                0.75 * item.score + 0.20 * record.importance + 0.05 * recency,
            )
            hits.append(
                AdaptiveMemoryHit(
                    memory=record,
                    score=round(score, 6),
                    lexical_score=item.lexical_score,
                    vector_score=item.vector_score,
                    importance_score=record.importance,
                    recency_score=round(recency, 6),
                    reasons=item.reasons
                    + ["importance {:.2f}".format(record.importance)],
                )
            )
        if not hits and records:
            hits = [
                AdaptiveMemoryHit(
                    memory=item,
                    score=round(0.8 * item.importance, 6),
                    importance_score=item.importance,
                    recency_score=round(self._recency_score(item), 6),
                    reasons=["scope fallback", "importance {:.2f}".format(item.importance)],
                )
                for item in sorted(
                    records,
                    key=lambda record: (record.importance, record.created_at),
                    reverse=True,
                )[:top_k]
            ]
        hits = sorted(hits, key=lambda item: item.score, reverse=True)[:top_k]
        selected: List[AdaptiveMemoryHit] = []
        lines: List[str] = []
        used_tokens = 0
        for hit in hits:
            line = self._format_line(hit.memory)
            line_tokens = self._estimate_tokens(line)
            if selected and used_tokens + line_tokens > token_budget:
                continue
            if not selected and line_tokens > token_budget:
                line = self._truncate_to_tokens(line, token_budget)
                line_tokens = self._estimate_tokens(line)
            selected.append(hit)
            lines.append(line)
            used_tokens += line_tokens
        self.store.touch_memory_records([item.memory.id for item in selected])
        saving = 1.0 - used_tokens / float(max(1, full_tokens))
        return AdaptiveMemoryContext(
            query=query[:500],
            context="\n".join("- " + line for line in lines)
            or "No relevant long-term memory.",
            hits=selected,
            candidate_count=len(records),
            selected_count=len(selected),
            latency_ms=int((time.time() - started) * 1000),
            estimated_full_tokens=full_tokens,
            selected_tokens=used_tokens,
            token_saving_ratio=round(max(0.0, saving), 4),
            embedding_provider=knowledge_base.last_trace.get(
                "embedding_provider", provider.name
            ),
        )

    def stats(self) -> Dict[str, Any]:
        records = self.store.list_memory_records()
        by_type: Dict[str, int] = {}
        for record in records:
            by_type[record.memory_type] = by_type.get(record.memory_type, 0) + 1
        return {"total": len(records), "by_type": by_type}

    @staticmethod
    def _digest(content: str, scope: Dict[str, Any]) -> str:
        values = [str(scope.get(key, "")) for key in (
            "user_id", "project_id", "agent_id", "run_id", "ticket_type", "memory_type", "fact_key"
        )]
        values.append(" ".join(content.split()).lower())
        return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Memory validity timestamps must include a timezone")
        return parsed

    @classmethod
    def _validate_validity(cls, start: str, end: str) -> None:
        lower = cls._timestamp(start) if start else None
        upper = cls._timestamp(end) if end else None
        if lower and upper and lower >= upper:
            raise ValueError("valid_to must be after valid_from")

    @classmethod
    def _is_effective(cls, record: AdaptiveMemoryRecord) -> bool:
        now = datetime.now(timezone.utc)
        try:
            return (not record.valid_from or cls._timestamp(record.valid_from) <= now) and (
                not record.valid_to or now < cls._timestamp(record.valid_to)
            )
        except ValueError:
            return False

    def revise(self, memory_id: str, content: str, reason: str) -> AdaptiveMemoryRecord:
        """Replace an explicitly selected current fact, inheriting its scope."""
        if not content.strip() or not reason.strip():
            raise ValueError("Revision content and reason are required")
        with self.store.memory_transaction():
            records = self.store.list_memory_records(include_inactive=True)
            previous = next((item for item in records if item.id == memory_id), None)
            if previous is None:
                raise ValueError("Memory not found")
            if previous.status != "active":
                # A retry of the exact completed revision must not create another version.
                successor = next((item for item in records if item.id == previous.superseded_by), None)
                if successor and successor.status == "active" and successor.content == " ".join(content.split()):
                    return successor
                raise ValueError("Memory is no longer active; reload its current version")
            if previous.content == " ".join(content.split()):
                return previous
            now = utc_now_iso()
            candidate = self._record(
                content, **{
                    key: getattr(previous, key) for key in (
                        "memory_type", "user_id", "project_id", "agent_id", "run_id",
                        "ticket_type", "source", "source_ref", "source_run_id",
                        "source_version_id", "importance", "fact_key"
                    )
                }, valid_from=now, supersedes=previous.id,
                metadata={**previous.metadata, "revision_reason": reason},
            )
            previous.status = "superseded"
            previous.metadata = {**previous.metadata, "revision_previous_valid_to": previous.valid_to}
            previous.superseded_by = candidate.id
            previous.valid_to = now
            previous.updated_at = now
            previous.invalidation_reason = reason
            records.append(candidate)
            # Resolve candidates for this explicitly revised fact without widening scope.
            for item in records:
                if (item.status == "pending_conflict" and candidate.fact_key
                        and item.fact_key == candidate.fact_key
                        and self.store.memory_scope(item) == self.store.memory_scope(candidate)):
                    item.status = "superseded"
                    item.superseded_by = candidate.id
                    item.updated_at = now
            self.store.save_memory_records(records)
            return candidate

    def invalidate(self, memory_id: str, reason: str) -> AdaptiveMemoryRecord:
        if not reason.strip():
            raise ValueError("Invalidation reason is required")
        with self.store.memory_transaction():
            records = self.store.list_memory_records(include_inactive=True)
            record = next((item for item in records if item.id == memory_id), None)
            if record is None:
                raise ValueError("Memory not found")
            if record.status == "inactive":
                return record
            if record.status not in {"active", "pending_conflict"}:
                raise ValueError("Memory is no longer current; reload its current version")
            record.status = "inactive"
            record.valid_to = utc_now_iso()
            record.updated_at = record.valid_to
            record.invalidation_reason = reason
            self.store.save_memory_records(records)
            return record

    def history(self, memory_id):
        """Return related revisions, without combining different facts or scopes."""
        with self.store.memory_transaction():
            records = self.store.list_memory_records(True)
            anchor = next((m for m in records if m.id == memory_id), None)
            if not anchor:
                raise ValueError("Memory not found")
            candidates = [m for m in records if self.store.memory_scope(m) == self.store.memory_scope(anchor)
                          and m.fact_key == anchor.fact_key]
            ids = {anchor.id}
            while True:
                expanded = ids | {m.id for m in candidates if m.supersedes in ids or m.superseded_by in ids}
                expanded |= {v for m in candidates if m.id in ids for v in (m.supersedes, m.superseded_by) if v}
                if expanded == ids:
                    break
                ids = expanded
            versions = [m for m in candidates if m.id in ids]
            active = [m for m in versions if m.status == "active"]
            current = active[0] if len(active) == 1 else anchor if not active and anchor.status == "inactive" else None
            return {"current_id": current.id if current else "", "versions": [m.model_dump() for m in versions]}

    def rollback(self, memory_id, target_id, expected_current_id, reason="人工回滚"):
        """Rollback creates a new revision; the selected historical record is immutable."""
        if not reason.strip():
            raise ValueError("Rollback reason is required")
        with self.store.memory_transaction():
            records = self.store.list_memory_records(True)
            by_id = {m.id: m for m in records}
            current, target = by_id.get(memory_id), by_id.get(target_id)
            if (not current or not target or memory_id != expected_current_id or
                    self.history(memory_id)["current_id"] != memory_id or
                    current.status not in {"active", "inactive"}):
                raise ValueError("当前版本已变化或不可回滚，请刷新版本历史")
            ancestors, cursor = set(), current
            while cursor and cursor.id not in ancestors:
                if self.store.memory_scope(cursor) != self.store.memory_scope(current):
                    raise ValueError("Memory lineage crosses scopes")
                ancestors.add(cursor.id)
                cursor = by_id.get(cursor.supersedes)
            if target_id not in ancestors or target.status == "pending_conflict":
                raise ValueError("只能回滚到同一事实版本链上的历史版本")
            now = utc_now_iso()
            if target.status == "superseded" and "revision_previous_valid_to" not in target.metadata:
                raise ValueError("旧版本缺少原有效期记录，请通过显式修订确认有效期")
            expires = target.metadata.get("revision_previous_valid_to", "") if target.status == "superseded" else target.valid_to
            if expires and self._timestamp(expires) <= self._timestamp(now):
                raise ValueError("历史事实有效期已结束，需明确修订后重新启用")
            if target.id == current.id and current.status == "active":
                return current
            candidate = self._record(target.content, **{
                key: getattr(target, key) for key in ("memory_type", "user_id", "project_id", "agent_id", "run_id",
                    "ticket_type", "source", "source_ref", "source_run_id", "source_version_id", "importance", "fact_key")},
                valid_from=target.valid_from if target.valid_from and self._timestamp(target.valid_from) > self._timestamp(now) else now,
                valid_to=expires, supersedes=current.id,
                metadata={**target.metadata, "rollback_target_id": target.id, "revision_reason": reason})
            current.status = "superseded"
            current.metadata = {**current.metadata, "revision_previous_valid_to": current.valid_to}
            current.superseded_by = candidate.id
            current.valid_to, current.updated_at, current.invalidation_reason = now, now, reason
            records.append(candidate)
            for item in records:
                if (item.status == "pending_conflict" and candidate.fact_key and item.fact_key == candidate.fact_key
                        and self.store.memory_scope(item) == self.store.memory_scope(candidate)):
                    item.status, item.superseded_by = "superseded", candidate.id
            self.store.save_memory_records(records)
            return candidate

    def _extract(self, text: str) -> List[Dict[str, Any]]:
        if not self.llm_extraction or not self.llm.enabled:
            return [self._fallback_fact(text)]
        schema = {
            "type": "object",
            "properties": {
                "memories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "memory_type": {"type": "string"},
                            "importance": {"type": "number"},
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["content", "memory_type", "importance", "entities"],
                    },
                }
            },
            "required": ["memories"],
        }
        try:
            payload = self.llm.generate_json(
                """Extract only durable facts useful in future QA work. Keep explicit
preferences, confirmed project decisions, reusable test rules, and review lessons.
Ignore greetings, transient progress, and facts already phrased as uncertain.
Use memory_type from user_preference, project_decision, team_rule, review_feedback,
agent_experience. Return short standalone facts and importance in [0,1].""",
                text,
                schema,
            )
            memories = payload.get("memories", [])
            return memories if isinstance(memories, list) else []
        except (LLMError, ValueError, TypeError):
            return [self._fallback_fact(text)]

    def _record(self, content: str, **kwargs: Any) -> AdaptiveMemoryRecord:
        normalized = " ".join(content.split()).strip()
        kwargs["importance"] = max(0.0, min(1.0, float(kwargs["importance"])))
        kwargs["entities"] = list(dict.fromkeys(kwargs.get("entities") or self._entities(normalized)))[:16]
        self._validate_validity(kwargs.get("valid_from", ""), kwargs.get("valid_to", ""))
        digest = self._digest(normalized, kwargs)
        return AdaptiveMemoryRecord(
            id="MEM-" + uuid.uuid4().hex[:12],
            content=normalized,
            content_hash=digest,
            **kwargs,
        )

    def _fallback_fact(self, text: str) -> Dict[str, Any]:
        content = " ".join(text.split())
        return {
            "content": content,
            "memory_type": "semantic",
            "importance": 0.6,
            "entities": self._entities(content),
        }

    @staticmethod
    def _entities(text: str) -> List[str]:
        values = [item.strip() for item in ENTITY_PATTERN.findall(text)]
        return list(dict.fromkeys(item for item in values if len(item) >= 2))[:16]

    @staticmethod
    def _scope_matches(record: AdaptiveMemoryRecord, **scope: str) -> bool:
        for key in ["user_id", "project_id", "agent_id", "run_id"]:
            value = getattr(record, key)
            requested = scope.get(key, "")
            if value and value != requested:
                return False
        requested_ticket = scope.get("ticket_type", "COMMON") or "COMMON"
        return record.ticket_type in {"COMMON", requested_ticket}

    @staticmethod
    def _as_document(record: AdaptiveMemoryRecord) -> KnowledgeDocument:
        return KnowledgeDocument(
            id=record.id,
            title="{} memory".format(record.memory_type),
            content=record.content,
            doc_type=record.memory_type,
            tags=record.entities + [record.ticket_type, record.agent_id],
            source=record.source,
            metadata={"memory_id": record.id, "ticket_type": record.ticket_type},
        )

    @staticmethod
    def _recency_score(record: AdaptiveMemoryRecord) -> float:
        try:
            created = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
            days = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400.0)
            return 1.0 / (1.0 + days / 30.0)
        except ValueError:
            return 0.5

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        other = max(0, len(text) - cjk)
        return max(1, cjk + int(math.ceil(other / 4.0)))

    @classmethod
    def _truncate_to_tokens(cls, text: str, token_budget: int) -> str:
        if cls._estimate_tokens(text) <= token_budget:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if cls._estimate_tokens(text[:middle]) <= token_budget:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip()

    @classmethod
    def _format_line(cls, record: AdaptiveMemoryRecord) -> str:
        return "[{} | {} | {}] {}".format(
            record.memory_type,
            cls._scope_label(record),
            record.source,
            record.content,
        )

    @staticmethod
    def _scope_label(record: AdaptiveMemoryRecord) -> str:
        values = []
        for key in ["user_id", "project_id", "agent_id", "run_id"]:
            value = getattr(record, key)
            if value:
                values.append("{}={}".format(key[:-3], value))
        values.append("ticket={}".format(record.ticket_type))
        return ",".join(values)
