import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from .models import ParsedDocument
from .text_chunking import split_text_naturally


TEST_FACTS = re.compile(
    r"(?i)(api|接口|字段|参数|状态|权限|角色|事件|kafka|消息|幂等|重试|异常|失败|"
    r"超时|回滚|灰度|兼容|验收|约束|必须|不得|只能|审核|流程|风控|安全|日志|监控)"
)
NOISE_TITLES = re.compile(
    r"(?i)^(目录|修订记录|修改记录|版本记录|文档信息|作者信息|联系人|项目成员|"
    r"会议纪要|参考资料|table of contents|revision history|contacts?)$"
)
ANCHORS = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:[A-Z_][A-Za-z0-9_]*)+|"
    r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_.]+|[A-Z]{2,}[A-Z0-9_-]*|"
    r"\d+(?:\.\d+)*(?:ms|s|min|h|%|MB|GB)?)\b"
)


class DocumentCompressor:
    """Create a compact, evidence-preserving test brief from extracted text."""

    SYSTEM = """You are a QA document parsing agent. Convert PRDs and technical designs
into compact Markdown for downstream test module planning. For every section decide
keep, compress, or drop. Drop only revision history, contacts, meeting notes, repeated
background, and content unrelated to implementation or testing. Preserve exact API and
field names, values, numbers, states, roles, permissions, events, errors, retries,
idempotency, gray release, compatibility, rollback, acceptance criteria, and ambiguity.
Never invent behavior. Return JSON only."""

    MAP_SCHEMA = {
        "type": "object",
        "properties": {
            "document_summary": {"type": "string"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["keep", "compress", "drop"]},
                        "reason": {"type": "string"},
                        "markdown": {"type": "string"},
                    },
                    "required": ["section_id", "decision", "reason", "markdown"],
                },
            },
        },
        "required": ["document_summary", "sections"],
    }
    REDUCE_SCHEMA = {
        "type": "object",
        "properties": {"markdown": {"type": "string"}},
        "required": ["markdown"],
    }

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.batch_chars = max(4000, int(os.getenv("DOCUMENT_MAP_BATCH_CHARS", "18000")))
        self.section_chars = max(2000, int(os.getenv("DOCUMENT_SECTION_CHARS", "7000")))
        self.reduce_chars = max(6000, int(os.getenv("DOCUMENT_REDUCE_BATCH_CHARS", "26000")))
        self.target_chars = max(3000, int(os.getenv("DOCUMENT_BRIEF_TARGET_CHARS", "12000")))

    def compress(self, document: ParsedDocument, force_demo: bool = False) -> ParsedDocument:
        raw = document.raw_text or document.normalized_text
        sections = self._sections(document)
        if self.llm.enabled and not force_demo:
            markdown, decisions, map_calls, reduce_calls = self._llm(document, sections)
            mode, llm_used = "llm_hierarchical", True
        else:
            markdown, decisions = self._fallback(document, sections)
            map_calls = reduce_calls = 0
            mode, llm_used = "deterministic_fallback", False

        markdown = markdown.strip() or raw.strip()
        excluded = [item for item in decisions if item["decision"] == "drop"]
        summary = {
            "mode": mode,
            "llm_used": llm_used,
            "input_chars": len(raw),
            "output_chars": len(markdown),
            "compression_ratio": round(len(markdown) / len(raw), 4) if raw else 1.0,
            "section_count": len(sections),
            "kept_sections": sum(item["decision"] == "keep" for item in decisions),
            "compressed_sections": sum(item["decision"] == "compress" for item in decisions),
            "excluded_sections": len(excluded),
            "map_calls": map_calls,
            "reduce_calls": reduce_calls,
            "target_chars": self.target_chars,
        }
        document.raw_text = raw
        document.normalized_text = markdown
        document.section_decisions = decisions
        document.excluded_sections = excluded
        document.compression_summary = summary
        document.extraction_summary = dict(document.extraction_summary)
        document.extraction_summary["compression"] = dict(summary)
        return document

    def _llm(
        self, document: ParsedDocument, sections: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]], int, int]:
        rendered, decisions, summaries = [], [], []
        map_calls = 0
        for batch in self._batches(sections, self.batch_chars):
            result = self.llm.generate_json(
                self.SYSTEM, self._prompt(document, batch), self.MAP_SCHEMA
            )
            map_calls += 1
            if result.get("document_summary"):
                summaries.append(str(result["document_summary"]).strip())
            answers = {
                str(item.get("section_id")): item
                for item in result.get("sections", [])
                if isinstance(item, dict)
            }
            for section in batch:
                answer = answers.get(section["id"], {})
                decision = str(answer.get("decision", "compress")).lower()
                if decision not in {"keep", "compress", "drop"}:
                    decision = "compress"
                reason = str(answer.get("reason", "")).strip()
                body = str(answer.get("markdown", "")).strip()
                section_text = section["title"] + "\n" + section["content"]
                known_noise = NOISE_TITLES.match(section["title"].strip())
                if known_noise and not TEST_FACTS.search(section_text):
                    decision = "drop"
                    reason = "Known maintenance section."
                    body = ""
                elif decision == "drop" and TEST_FACTS.search(section_text):
                    decision = "compress"
                    reason = "Drop overridden: test-critical facts detected."
                if decision != "drop":
                    body = body or self._extract(section["content"], self.section_chars)
                    body = self._restore_anchors(body, section["content"])
                    rendered.append(self._render(section, body))
                decisions.append(self._decision(section, decision, reason, len(body)))

        result = "# {}".format(document.filename)
        if summaries:
            result += "\n\n## 文档概览\n\n" + self._dedupe("\n".join(summaries))
        if rendered:
            result += "\n\n" + "\n\n".join(rendered)
        result, reduce_calls = self._reduce(result)
        return result, decisions, map_calls, reduce_calls

    def _reduce(self, markdown: str) -> Tuple[str, int]:
        current, calls = markdown.strip(), 0
        for _ in range(3):
            if len(current) <= self.target_chars:
                break
            parts = split_text_naturally(current, self.reduce_chars)
            reduced = []
            for index, part in enumerate(parts, 1):
                prompt = """Further compress this test brief. Preserve evidence markers,
exact identifiers, states, fields, numbers, constraints, and Markdown headings.
Remove repetition, not test facts. Do not invent requirements.
Target about {} characters.

PART {}/{}:
{}""".format(
                    max(1800, self.target_chars // max(1, len(parts))),
                    index, len(parts), part,
                )
                answer = self.llm.generate_json(self.SYSTEM, prompt, self.REDUCE_SCHEMA)
                calls += 1
                body = str(answer.get("markdown", "")).strip() or part
                reduced.append(self._restore_anchors(body, part))
            candidate = "\n\n".join(reduced)
            if len(candidate) >= len(current):
                break
            current = candidate
        return current, calls

    def _fallback(
        self, document: ParsedDocument, sections: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        rendered, decisions = ["# {}".format(document.filename)], []
        total = sum(len(section["content"]) for section in sections)
        limit = max(1000, self.target_chars // max(1, len(sections)))
        for section in sections:
            noise = NOISE_TITLES.match(section["title"].strip())
            if noise and not TEST_FACTS.search(section["content"]):
                decisions.append(self._decision(
                    section, "drop", "Known maintenance section.", 0
                ))
                continue
            if total <= self.target_chars:
                body, decision = section["content"], "keep"
                reason = "Document fits the context budget."
            else:
                body, decision = self._extract(section["content"], limit), "compress"
                reason = "Extractive fallback used for context budget."
            rendered.append(self._render(section, body))
            decisions.append(self._decision(section, decision, reason, len(body)))
        return "\n\n".join(rendered), decisions

    def _sections(self, document: ParsedDocument) -> List[Dict[str, Any]]:
        grouped: "OrderedDict[Tuple[str, int], List[Any]]" = OrderedDict()
        parts: Dict[str, int] = {}
        for chunk in document.chunks:
            title = (chunk.section or "正文").strip()
            part = parts.get(title, 1)
            key = (title, part)
            size = sum(len(item.content) for item in grouped.get(key, []))
            if size and size + len(chunk.content) > self.section_chars:
                part += 1
                parts[title], key = part, (title, part)
            else:
                parts.setdefault(title, part)
            grouped.setdefault(key, []).append(chunk)

        result = []
        for index, ((title, part), chunks) in enumerate(grouped.items(), 1):
            display = title if parts[title] == 1 else "{}（第 {} 部分）".format(title, part)
            result.append({
                "id": "SEC-{:03d}".format(index),
                "title": display,
                "content": "\n\n".join(item.content.strip() for item in chunks if item.content.strip()),
                "pages": sorted({item.page for item in chunks if item.page}),
                "evidence_ids": [item.id for item in chunks],
            })
        return result

    @staticmethod
    def _batches(sections: List[Dict[str, Any]], limit: int) -> List[List[Dict[str, Any]]]:
        result, current, size = [], [], 0
        for section in sections:
            item_size = len(section["content"]) + len(section["title"]) + 80
            if current and size + item_size > limit:
                result.append(current)
                current, size = [], 0
            current.append(section)
            size += item_size
        if current:
            result.append(current)
        return result

    @staticmethod
    def _prompt(document: ParsedDocument, sections: List[Dict[str, Any]]) -> str:
        blocks = []
        for item in sections:
            blocks.append("""[SECTION {id}]
Title: {title}
Pages: {pages}
Evidence IDs: {evidence}
Content:
{content}""".format(
                id=item["id"], title=item["title"],
                pages=", ".join(str(page) for page in item["pages"]) or "?",
                evidence=", ".join(item["evidence_ids"]), content=item["content"],
            ))
        return """Document: {}
Type: {}

Analyze every section for a compact but complete test specification:
{}""".format(document.filename, document.document_type, "\n\n".join(blocks))

    @staticmethod
    def _render(section: Dict[str, Any], body: str) -> str:
        pages = " ".join(
            "[第{}页]".format(page) for page in section["pages"]
        ) or "[页码未知]"
        evidence = "、".join(section["evidence_ids"])
        return "## {}\n\n{}\n\n> 证据：{} {}".format(
            section["title"], body.strip(), pages, evidence
        )

    @staticmethod
    def _decision(
        section: Dict[str, Any], decision: str, reason: str, output_chars: int
    ) -> Dict[str, Any]:
        return {
            "section_id": section["id"], "title": section["title"],
            "pages": section["pages"], "evidence_ids": section["evidence_ids"],
            "decision": decision, "reason": reason,
            "input_chars": len(section["content"]), "output_chars": output_chars,
        }

    @staticmethod
    def _extract(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chosen = []
        for line in lines:
            if TEST_FACTS.search(line) or not chosen:
                chosen.append(line)
            if len("\n".join(chosen)) >= limit:
                break
        return ("\n".join(chosen) or text[:limit])[:limit].strip()

    @staticmethod
    def _restore_anchors(rendered: str, source: str) -> str:
        missing = [
            value for value in dict.fromkeys(ANCHORS.findall(source))
            if value not in rendered
        ]
        snippets = []
        for value in missing:
            start = source.find(value)
            if start < 0:
                continue
            left = max(0, start - 90)
            right = min(len(source), start + len(value) + 120)
            snippet = source[left:right].strip()
            if left:
                snippet = "..." + snippet
            if right < len(source):
                snippet += "..."
            snippets.append(snippet)
        snippets = list(dict.fromkeys(snippets))
        if not snippets:
            return rendered
        return "{}\n\n### 必须保留的原文事实\n\n{}".format(
            rendered.rstrip(), "\n".join("- " + item for item in snippets)
        )

    @staticmethod
    def _dedupe(text: str) -> str:
        seen, result = set(), []
        for line in text.splitlines():
            key = re.sub(r"\s+", "", line).lower()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            result.append(line)
        return "\n".join(result).strip()
