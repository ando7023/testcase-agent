import hashlib
from typing import Any, Dict, List

from .models import KnowledgeDocument, utc_now_iso
from .store import JsonStore
from .text_chunking import suggest_two_parts


def mutation_id(action: str, document_ids: List[str], content: str) -> str:
    key = "{}|{}|{}".format(action, "|".join(document_ids), content)
    return "CHK-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class KnowledgeChunkManager:
    """Versioned manual split/merge operations with immutable lineage."""

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def convert_to_parent_child(self, document_id: str) -> Dict[str, Any]:
        source = self._active_child(document_id)
        if source.source_id and source.parent_id:
            raise ValueError("Knowledge chunk already has a parent-child structure")

        now = utc_now_iso()
        source_key = "{}|{}|{}".format(
            source.id, source.source, source.metadata.get("ticket_type", "COMMON")
        )
        source_id = source.source_id or (
            "SRC-" + hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
        )
        parent_key = "{}|{}|{}".format(source_id, source.title, source.content)
        parent_id = (
            "PAR-" + hashlib.sha1(parent_key.encode("utf-8")).hexdigest()[:12]
        )

        parent = source.model_copy(deep=True)
        parent.id = parent_id
        parent.title = "{} / parent".format(source.title)
        parent.source_id = source_id
        parent.parent_id = ""
        parent.chunk_level = "parent"
        parent.chunking_version = "manual-parent-v1"
        parent.chunk_version = 1
        parent.status = "active"
        parent.supersedes = []
        parent.superseded_by = []
        parent.created_at = now
        parent.updated_at = now
        parent.metadata = dict(source.metadata)
        parent.metadata.update(
            {
                "mutation": "manual_parent_conversion",
                "indexable": False,
                "child_count": 1,
            }
        )

        child = source.model_copy(deep=True)
        child.id = mutation_id("convert", [source.id], source.content)
        child.source_id = source_id
        child.parent_id = parent_id
        child.chunk_level = "child"
        child.chunking_version = "manual-parent-v1"
        child.chunk_version = source.chunk_version + 1
        child.status = "active"
        child.supersedes = [source.id]
        child.superseded_by = []
        child.created_at = now
        child.updated_at = now
        child.metadata = dict(source.metadata)
        child.metadata.update(
            {"mutation": "manual_parent_conversion", "indexable": True}
        )

        source.status = "inactive"
        source.superseded_by = [child.id]
        source.updated_at = now
        source.metadata = dict(source.metadata)
        source.metadata["invalidation_reason"] = "manual_parent_conversion"

        changed = self.store.apply_knowledge_mutation(
            [source.model_dump(), parent.model_dump(), child.model_dump()],
            [source.id],
        )
        return {
            "operation": "convert_to_parent_child",
            "changed": changed,
            "invalidated_ids": [source.id],
            "parent": parent.model_dump(),
            "documents": [child.model_dump()],
        }
    def suggest_split(self, document_id: str) -> Dict[str, Any]:
        source = self._active_child(document_id)
        parts = suggest_two_parts(source.content)
        if len(parts) < 2:
            raise ValueError("Knowledge chunk is too short to split")
        return {
            "document_id": source.id,
            "strategy": "paragraph_sentence_clause_v1",
            "parts": parts,
        }
    def split(self, document_id: str, parts: List[str]) -> Dict[str, Any]:
        source = self._active_child(document_id)
        normalized_parts = [part.strip() for part in parts if part.strip()]
        if len(normalized_parts) < 2:
            raise ValueError("Split requires at least two non-empty parts")
        if "".join(normalized_parts) == "":
            raise ValueError("Split content cannot be empty")

        created: List[KnowledgeDocument] = []
        for index, content in enumerate(normalized_parts, 1):
            candidate = source.model_copy(deep=True)
            candidate.id = mutation_id(
                "split-{}".format(index), [source.id], content
            )
            candidate.title = "{} / part {}".format(source.title, index)
            candidate.content = content
            candidate.chunk_index = source.chunk_index * 100 + index
            candidate.chunk_version = source.chunk_version + 1
            candidate.status = "active"
            candidate.supersedes = [source.id]
            candidate.superseded_by = []
            candidate.checksum = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()
            candidate.created_at = utc_now_iso()
            candidate.updated_at = candidate.created_at
            candidate.metadata = dict(source.metadata)
            candidate.metadata.update(
                {"mutation": "manual_split", "indexable": True}
            )
            created.append(candidate)

        source.status = "inactive"
        source.superseded_by = [item.id for item in created]
        source.updated_at = utc_now_iso()
        source.metadata = dict(source.metadata)
        source.metadata["invalidation_reason"] = "manual_split"
        changed = self.store.apply_knowledge_mutation(
            [source.model_dump()] + [item.model_dump() for item in created],
            [source.id],
        )
        return {
            "operation": "split",
            "changed": changed,
            "invalidated_ids": [source.id],
            "documents": [item.model_dump() for item in created],
        }

    def merge(
        self, document_ids: List[str], title: str = ""
    ) -> Dict[str, Any]:
        unique_ids = list(dict.fromkeys(document_ids))
        if len(unique_ids) < 2:
            raise ValueError("Merge requires at least two chunks")
        sources = [self._active_child(document_id) for document_id in unique_ids]
        source_ids = {item.source_id for item in sources}
        parent_ids = {item.parent_id for item in sources}
        if len(source_ids) != 1 or len(parent_ids) != 1:
            raise ValueError("Only sibling chunks from the same source can be merged")

        sources.sort(key=lambda item: (item.page or 0, item.chunk_index, item.id))
        content = "\n\n".join(item.content.strip() for item in sources)
        candidate = sources[0].model_copy(deep=True)
        candidate.id = mutation_id("merge", [item.id for item in sources], content)
        candidate.title = title.strip() or sources[0].title.rsplit(" / part", 1)[0]
        candidate.content = content
        candidate.chunk_index = min(item.chunk_index for item in sources)
        candidate.chunk_version = max(item.chunk_version for item in sources) + 1
        candidate.status = "active"
        candidate.supersedes = [item.id for item in sources]
        candidate.superseded_by = []
        candidate.checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        candidate.created_at = utc_now_iso()
        candidate.updated_at = candidate.created_at
        candidate.metadata = dict(candidate.metadata)
        candidate.metadata.update({"mutation": "manual_merge", "indexable": True})

        for source in sources:
            source.status = "inactive"
            source.superseded_by = [candidate.id]
            source.updated_at = utc_now_iso()
            source.metadata = dict(source.metadata)
            source.metadata["invalidation_reason"] = "manual_merge"

        changed = self.store.apply_knowledge_mutation(
            [item.model_dump() for item in sources] + [candidate.model_dump()],
            [item.id for item in sources],
        )
        return {
            "operation": "merge",
            "changed": changed,
            "invalidated_ids": [item.id for item in sources],
            "documents": [candidate.model_dump()],
        }

    def _active_child(self, document_id: str) -> KnowledgeDocument:
        document = self.store.get_knowledge(document_id)
        if not document:
            raise ValueError("Knowledge chunk not found: {}".format(document_id))
        if document.status != "active":
            raise ValueError("Knowledge chunk is already inactive: {}".format(document_id))
        if document.chunk_level != "child":
            raise ValueError("Only child chunks can be edited")
        return document