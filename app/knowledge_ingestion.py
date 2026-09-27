import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from .models import DocumentChunk, KnowledgeDocument, ParsedDocument


DOCUMENT_TYPE_MAP = {
    "technical_design": "technical_design",
    "prd": "business_rule",
    "api_spec": "api_contract",
    "test_plan": "test_standard",
    "general_document": "business_rule",
}


def stable_digest(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


class KnowledgeIngestionPipeline:
    """Build section parents and indexable child chunks from parsed evidence."""

    def build(
        self,
        document: ParsedDocument,
        ticket_type: str,
        doc_type: str = "",
        version: str = "1.0",
        effective_at: str = "",
        expires_at: str = "",
        chunking_version: str = "section-child-v1",
    ) -> List[KnowledgeDocument]:
        resolved_type = doc_type or DOCUMENT_TYPE_MAP.get(
            document.document_type, "business_rule"
        )
        source_key = "{}|{}".format(ticket_type, document.filename.lower())
        source_digest = stable_digest(source_key)
        source_id = "SRC-" + source_digest
        stem = Path(document.filename).stem

        grouped: "OrderedDict[Tuple[int, str], List[DocumentChunk]]" = OrderedDict()
        for chunk in document.chunks:
            key = (chunk.page or 0, chunk.section or "正文")
            grouped.setdefault(key, []).append(chunk)

        documents: List[KnowledgeDocument] = []
        child_index = 0
        for parent_index, ((page_number, section), chunks) in enumerate(
            grouped.items(), 1
        ):
            parent_content = "\n\n".join(
                chunk.content.strip() for chunk in chunks if chunk.content.strip()
            )
            parent_key = "{}|{}|{}|{}".format(
                source_id, page_number, section, parent_content
            )
            parent_id = "PAR-" + stable_digest(parent_key)
            if not parent_content:
                continue
            common = {
                "doc_type": resolved_type,
                "tags": [
                    tag
                    for tag in dict.fromkeys(
                        [ticket_type, stem, section, document.document_type]
                    )
                    if tag
                ],
                "source": document.filename,
                "section": section,
                "page": page_number or None,
                "source_id": source_id,
                "chunking_version": chunking_version,
                "version": version,
                "effective_at": effective_at,
                "expires_at": expires_at,
                "status": "active",
                "section_path": [section] if section else [],
            }
            documents.append(
                KnowledgeDocument(
                    id=parent_id,
                    title="{} / {}".format(stem, section),
                    content=parent_content,
                    chunk_index=parent_index,
                    parent_id="",
                    chunk_level="parent",
                    checksum=hashlib.sha256(
                        parent_content.encode("utf-8")
                    ).hexdigest(),
                    metadata={
                        "domain": "ticket",
                        "ticket_type": ticket_type,
                        "document_type": document.document_type,
                        "indexable": False,
                        "child_count": len(chunks),
                    },
                    **common
                )
            )
            for chunk in chunks:
                content = chunk.content.strip()
                if not content:
                    continue
                child_index += 1
                child_key = "{}|{}|{}|{}".format(
                    source_id, page_number, section, content
                )
                documents.append(
                    KnowledgeDocument(
                        id="CHK-" + stable_digest(child_key),
                        title="{} / {}".format(stem, section),
                        content=content,
                        chunk_index=child_index,
                        parent_id=parent_id,
                        chunk_level="child",
                        checksum=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        metadata={
                            "domain": "ticket",
                            "ticket_type": ticket_type,
                            "document_type": document.document_type,
                            "extraction_method": chunk.extraction_method,
                            "ocr_confidence": chunk.ocr_confidence,
                            "indexable": True,
                        },
                        **common
                    )
                )
        return documents