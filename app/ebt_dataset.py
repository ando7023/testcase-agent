import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import KnowledgeDocument, RetrievalDataset, RetrievalEvalCase


EBT_DATASET_ID = "EBT-RAG-V1"
EBT_BASE_URL = "https://huggingface.co/datasets/thearod5/ebt/resolve/main/"
EBT_FILES = ("README.md", "artifacts.csv", "matrices.csv", "traces.csv", "train.csv")
EBT_MAX_FILE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class EBTArtifact:
    id: str
    content: str
    layer: str
    summary: str = ""


@dataclass(frozen=True)
class EBTTrace:
    requirement_id: str
    test_case_id: str


class EBTRepository:
    """Loads EBT without adding benchmark artifacts to production knowledge."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def ready(self) -> bool:
        return all((self.root / name).is_file() for name in EBT_FILES)

    def status(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "dataset_id": EBT_DATASET_ID,
            "ready": self.ready,
            "root": str(self.root),
            "source": "thearod5/ebt",
            "license": "MIT",
            "files": [
                {
                    "name": name,
                    "present": (self.root / name).is_file(),
                    "size": (self.root / name).stat().st_size
                    if (self.root / name).is_file()
                    else 0,
                }
                for name in EBT_FILES
            ],
        }
        if self.ready:
            artifacts, traces = self.load()
            counts: Dict[str, int] = {}
            for artifact in artifacts.values():
                counts[artifact.layer] = counts.get(artifact.layer, 0) + 1
            result.update(
                {
                    "artifact_count": len(artifacts),
                    "artifact_layers": counts,
                    "positive_trace_count": len(traces),
                    "retrieval_case_count": len(self.build_retrieval_dataset()[0].cases),
                }
            )
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            result["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return result

    def download(self, timeout: int = 60) -> Dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_files = []
        for name in EBT_FILES:
            target = self.root / name
            if target.is_file() and 0 < target.stat().st_size <= EBT_MAX_FILE_BYTES:
                payload = target.read_bytes()
            else:
                request = urllib.request.Request(
                    EBT_BASE_URL + name,
                    headers={"User-Agent": "CaseForge-EBT-Importer/1.0"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    declared_size = int(response.headers.get("Content-Length") or 0)
                    if declared_size > EBT_MAX_FILE_BYTES:
                        raise ValueError("EBT file exceeds size limit: {}".format(name))
                    payload = response.read(EBT_MAX_FILE_BYTES + 1)
                if len(payload) > EBT_MAX_FILE_BYTES:
                    raise ValueError("EBT file exceeds size limit: {}".format(name))
                self._atomic_write(target, payload)
            manifest_files.append(
                {
                    "name": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifest = {
            "dataset_id": EBT_DATASET_ID,
            "source": "thearod5/ebt",
            "base_url": EBT_BASE_URL,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "license": "MIT",
            "usage_note": "Source and license are pinned in this manifest; retain attribution when redistributing.",
            "files": manifest_files,
        }
        self._atomic_write(
            self.root / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return self.status()

    def load(self) -> Tuple[Dict[str, EBTArtifact], List[EBTTrace]]:
        if not self.ready:
            raise ValueError("EBT is not imported; run the EBT importer first")
        artifacts: Dict[str, EBTArtifact] = {}
        with (self.root / "artifacts.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                artifact_id = (row.get("id") or "").strip()
                content = (row.get("content") or "").strip()
                layer = (row.get("layer") or "").strip()
                if artifact_id and content and layer:
                    artifacts[artifact_id] = EBTArtifact(
                        id=artifact_id,
                        content=content,
                        layer=layer,
                        summary=(row.get("summary") or "").strip(),
                    )

        traces: List[EBTTrace] = []
        seen = set()
        with (self.root / "traces.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                if str(row.get("label", "1")).strip() not in {"1", "true", "True"}:
                    continue
                source = artifacts.get((row.get("s_id") or "").strip())
                target = artifacts.get((row.get("t_id") or "").strip())
                if not source or not target:
                    continue
                pair = self._normalize_trace(source, target)
                if pair and pair not in seen:
                    seen.add(pair)
                    traces.append(EBTTrace(requirement_id=pair[0], test_case_id=pair[1]))
        return artifacts, traces

    def build_retrieval_dataset(
        self, max_cases: int = 0
    ) -> Tuple[RetrievalDataset, List[KnowledgeDocument]]:
        artifacts, traces = self.load()
        tests_to_requirements: Dict[str, List[str]] = {}
        requirements_to_tests: Dict[str, List[str]] = {}
        for trace in traces:
            tests_to_requirements.setdefault(trace.test_case_id, []).append(
                trace.requirement_id
            )
            requirements_to_tests.setdefault(trace.requirement_id, []).append(
                trace.test_case_id
            )

        documents = [
            KnowledgeDocument(
                id="EBT-TEST-{}".format(test_id),
                title="EBT test case {}".format(test_id),
                content=artifacts[test_id].content,
                doc_type="case_example",
                tags=["EBT", "test case", "traceability"],
                source="thearod5/ebt",
                source_id="EBT-TEST-{}".format(test_id),
                metadata={
                    "domain": "external_benchmark",
                    "ticket_type": "COMMON",
                    "artifact_id": test_id,
                    "artifact_layer": artifacts[test_id].layer,
                    "requirement_ids": sorted(
                        set(tests_to_requirements[test_id]), key=self._sort_key
                    ),
                },
            )
            for test_id in sorted(tests_to_requirements, key=self._sort_key)
            if test_id in artifacts
        ]
        requirement_ids = sorted(requirements_to_tests, key=self._sort_key)
        if max_cases > 0:
            requirement_ids = requirement_ids[:max_cases]
        cases = [
            RetrievalEvalCase(
                id="EBT-RAG-{}".format(requirement_id),
                query=artifacts[requirement_id].content,
                ticket_type="COMMON",
                expected_document_ids=[
                    "EBT-TEST-{}".format(test_id)
                    for test_id in sorted(
                        set(requirements_to_tests[requirement_id]), key=self._sort_key
                    )
                ],
            )
            for requirement_id in requirement_ids
            if requirement_id in artifacts
        ]
        return (
            RetrievalDataset(
                id=EBT_DATASET_ID,
                name="EBT requirement-to-test trace retrieval benchmark",
                cases=cases,
            ),
            documents,
        )

    @staticmethod
    def _normalize_trace(
        source: EBTArtifact, target: EBTArtifact
    ) -> Optional[Tuple[str, str]]:
        source_layer = source.layer.lower()
        target_layer = target.layer.lower()
        if source_layer == "test" and target_layer == "requirement":
            return target.id, source.id
        if source_layer == "requirement" and target_layer == "test":
            return source.id, target.id
        return None

    @staticmethod
    def _sort_key(value: str) -> Tuple[int, object]:
        return (0, int(value)) if value.isdigit() else (1, value)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
