import hashlib
import io
import json
import zipfile
from typing import Any, Dict, List

from .models import MindMapDocument, MindMapNode


class XMindExportAdapter:
    """Serialize a mind-map document to the XMind JSON archive format."""

    name = "xmind_export"
    media_type = "application/vnd.xmind.workbook"

    def schema(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": "Export a module or test-case mind map as an editable .xmind workbook.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "view": {"type": "string", "enum": ["modules", "cases"]},
                },
                "required": ["project_id", "view"],
            },
            "side_effect": "none",
        }

    def export(self, document: MindMapDocument) -> bytes:
        sheet_id = self._stable_id("sheet", document.project_id, document.view)
        root_topic = self._topic(document.root, "root")
        content = [
            {
                "id": sheet_id,
                "class": "sheet",
                "title": self._sheet_title(document),
                "rootTopic": root_topic,
            }
        ]
        metadata = {
            "creator": {"name": "CaseForge"},
            "dataStructureVersion": "2",
        }
        manifest = {
            "file-entries": {
                "content.json": {},
                "metadata.json": {},
            }
        }

        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            self._write_json(archive, "content.json", content)
            self._write_json(archive, "metadata.json", metadata)
            self._write_json(archive, "manifest.json", manifest)
        return output.getvalue()

    def _topic(self, node: MindMapNode, path: str) -> Dict[str, Any]:
        topic: Dict[str, Any] = {
            "id": self._stable_id(path, node.id),
            "class": "topic",
            "title": node.label,
        }
        labels = self._labels(node)
        if labels:
            topic["labels"] = labels

        note = self._note(node)
        if note:
            topic["notes"] = {"plain": {"content": note + "\n"}}

        if node.children:
            topic["children"] = {
                "attached": [
                    self._topic(child, f"{path}/{index}")
                    for index, child in enumerate(node.children)
                ]
            }
        return topic

    @staticmethod
    def _labels(node: MindMapNode) -> List[str]:
        values = []
        for key in ("priority", "case_type", "module_id"):
            value = node.metadata.get(key)
            if value:
                values.append(str(value))
        return values

    @staticmethod
    def _note(node: MindMapNode) -> str:
        metadata = node.metadata
        lines = []
        scalar_fields = [
            ("Node type", node.node_type),
            ("Objective", metadata.get("objective")),
            ("Phase", metadata.get("phase")),
        ]
        for label, value in scalar_fields:
            if value:
                lines.append(f"{label}: {value}")

        list_fields = [
            ("Requirements", metadata.get("requirement_ids")),
            ("Risks", metadata.get("risks") or metadata.get("risk_tags")),
            ("Case types", metadata.get("case_types")),
        ]
        for label, values in list_fields:
            if values:
                lines.append(f"{label}: {', '.join(str(value) for value in values)}")
        return "\n".join(lines)

    @staticmethod
    def _sheet_title(document: MindMapDocument) -> str:
        suffix = "Test Cases" if document.view == "cases" else "Modules"
        return f"{document.root.label} - {suffix}"

    @staticmethod
    def _stable_id(*parts: str) -> str:
        raw = "|".join(parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:26]

    @staticmethod
    def _write_json(
        archive: zipfile.ZipFile,
        filename: str,
        value: Any,
    ) -> None:
        info = zipfile.ZipInfo(filename, date_time=(2020, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o600 << 16
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        archive.writestr(info, payload)
