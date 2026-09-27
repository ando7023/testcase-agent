import csv
import hashlib
import io
import json
import re
import tempfile
import urllib.request
import uuid
import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional

from .ebt_dataset import EBTRepository
from .models import (
    AtomicRequirement,
    ModuleTree,
    RequirementAnalysis,
    TestCase,
    TestModule,
    TestStep,
)


STORYSEEK_DATASET_ID = "STORYSEEK-V1"
SRS_DATASET_ID = "PUBLIC-SRS-V1"
CRITIC_DATASET_ID = "CRITIC-MUTATION-V1"

STORYSEEK_BASE_URL = (
    "https://huggingface.co/datasets/SoftACE/StorySeek/resolve/main/"
)
STORYSEEK_FILES = (
    "1_250833_gitlab-runner.csv",
    "2_3828396_gitlab-charts.csv",
    "3_6206924_tildes.csv",
    "4_28644964_fpc.csv",
    "5_5261717_gitlab-vscode-extension.csv",
    "6_734943_gitlab-pages.csv",
    "7_28419588_lazarus.csv",
    "8_2009901_gitaly.csv",
    "9_14052249_mythic-table.csv",
    "10_12584701_stackgres.csv",
)
SRS_ARCHIVE_URL = (
    "https://zenodo.org/records/7897601/files/Dataset.zip?download=1"
)

TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff]", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "shall", "should", "that", "the", "to", "with",
    "test", "case", "system", "user",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tokenize(value: str) -> set:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value or "")
        if token.lower() not in STOPWORDS
    }


def token_recall(expected: str, actual: str) -> float:
    gold = tokenize(expected)
    if not gold:
        return 1.0
    return len(gold & tokenize(actual)) / float(len(gold))


def average(values: Iterable[float]) -> float:
    items = list(values)
    return round(mean(items), 4) if items else 0.0


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.name, dir=str(path.parent), delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def download_bytes(url: str, max_bytes: int, timeout: int = 120) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "CaseForge-Public-Benchmark/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > max_bytes:
            raise ValueError("Benchmark file exceeds size limit")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("Benchmark file exceeds size limit")
    return payload


class StorySeekRepository:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def ready(self) -> bool:
        return all((self.root / name).is_file() for name in STORYSEEK_FILES)

    def status(self) -> Dict[str, Any]:
        result = {
            "dataset_id": STORYSEEK_DATASET_ID,
            "name": "StorySeek public requirements benchmark",
            "ready": self.ready,
            "source": "SoftACE/StorySeek",
            "license": "MIT",
            "purpose": ["requirement_understanding", "module_planning"],
            "project_count": sum(
                1 for name in STORYSEEK_FILES if (self.root / name).is_file()
            ),
            "sample_count": 0,
        }
        if self.ready:
            result["sample_count"] = len(self.load(split="all"))
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            result["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        return result

    def download(self) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        files = []
        for name in STORYSEEK_FILES:
            target = self.root / name
            payload = target.read_bytes() if target.is_file() else download_bytes(
                STORYSEEK_BASE_URL + name, 3 * 1024 * 1024
            )
            if not payload:
                raise ValueError("StorySeek file is empty: {}".format(name))
            atomic_write(target, payload)
            files.append({
                "name": name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
        manifest = {
            "dataset_id": STORYSEEK_DATASET_ID,
            "source": "https://huggingface.co/datasets/SoftACE/StorySeek",
            "license": "MIT",
            "downloaded_at": utc_now(),
            "files": files,
        }
        atomic_write(
            self.root / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return self.status()

    def load(self, split: str = "test", limit: int = 0) -> List[Dict[str, str]]:
        if not self.ready:
            raise ValueError("StorySeek is not imported")
        split_files = {
            "development": STORYSEEK_FILES[:6],
            "validation": STORYSEEK_FILES[6:8],
            "test": STORYSEEK_FILES[8:],
            "all": STORYSEEK_FILES,
        }
        if split not in split_files:
            raise ValueError("Unknown StorySeek split: {}".format(split))
        records: List[Dict[str, str]] = []
        for filename in split_files[split]:
            with (self.root / filename).open(
                "r", encoding="utf-8-sig", newline="", errors="replace"
            ) as handle:
                for row in csv.DictReader(handle):
                    normalized = {
                        str(key or "").strip().lower(): str(value or "").strip()
                        for key, value in row.items()
                    }
                    actor = normalized.get("us_actor") or normalized.get("actor", "")
                    action = normalized.get("us_action", "")
                    outcome = normalized.get("us_expected_outcome", "")
                    if not any((action, outcome, normalized.get("goal"))):
                        continue
                    records.append({
                        "id": normalized.get("storyid") or "{}:{}".format(
                            filename, len(records) + 1
                        ),
                        "project": filename.rsplit(".", 1)[0],
                        "background": normalized.get("background", ""),
                        "problems": normalized.get("problems", ""),
                        "solutions": normalized.get("solutions", ""),
                        "goal": normalized.get("goal", ""),
                        "actor": actor,
                        "impact": normalized.get("impact", ""),
                        "deliverable": normalized.get("deliverable", ""),
                        "action": action,
                        "expected_outcome": outcome,
                    })
                    if limit and len(records) >= limit:
                        return records
        return records


class SRSRepository:
    SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt"}

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files_root = root / "files"
        self.archive = root / "Dataset.zip"

    @property
    def ready(self) -> bool:
        return self.files_root.is_dir() and any(self.files_root.rglob("*"))

    def status(self) -> Dict[str, Any]:
        files = [path for path in self.files_root.rglob("*") if path.is_file()] if self.ready else []
        pairs = self.pairs() if self.ready else []
        result = {
            "dataset_id": SRS_DATASET_ID,
            "name": "Software Requirements Data Set",
            "ready": self.ready,
            "source": "Zenodo 10.5281/zenodo.7897601",
            "license": "CC-BY-4.0",
            "purpose": ["document_parsing", "document_compression"],
            "file_count": len(files),
            "sample_count": len(pairs),
        }
        manifest = self.root / "manifest.json"
        if manifest.is_file():
            result["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        return result

    def download(self) -> Dict[str, Any]:
        payload = self.archive.read_bytes() if self.archive.is_file() else download_bytes(
            SRS_ARCHIVE_URL, 25 * 1024 * 1024, timeout=180
        )
        atomic_write(self.archive, payload)
        self._extract(payload)
        manifest = {
            "dataset_id": SRS_DATASET_ID,
            "source": "https://zenodo.org/records/7897601",
            "license": "CC-BY-4.0",
            "downloaded_at": utc_now(),
            "archive": {
                "name": self.archive.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        }
        atomic_write(
            self.root / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return self.status()

    def _extract(self, payload: bytes) -> None:
        self.files_root.mkdir(parents=True, exist_ok=True)
        total = 0
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) > 5000:
                raise ValueError("SRS archive contains too many files")
            for member in members:
                total += member.file_size
                if total > 200 * 1024 * 1024:
                    raise ValueError("SRS archive expands beyond size limit")
                target = (self.files_root / member.filename).resolve()
                try:
                    target.relative_to(self.files_root.resolve())
                except ValueError:
                    raise ValueError("Unsafe path in SRS archive")
                atomic_write(target, archive.read(member))

    def pairs(self, limit: int = 0) -> List[Dict[str, str]]:
        if not self.ready:
            return []
        files = [path for path in self.files_root.rglob("*") if path.is_file()]
        pairs: List[Dict[str, str]] = []
        folders = sorted({path.parent for path in files}, key=lambda path: str(path).lower())
        for folder in folders:
            siblings = [item for item in files if item.parent == folder]
            raw_files = sorted(
                (
                    item for item in siblings
                    if item.stem.lower().endswith("_raw")
                    and item.suffix.lower() in {".txt", ".md"}
                ),
                key=lambda item: (item.suffix.lower() != ".txt", item.name.lower()),
            )
            if not raw_files:
                continue
            gold = raw_files[0]
            derived_names = {
                "all", "allrelevant", "functionalrequirements", "usecases", "_source"
            }
            candidates = sorted(
                (
                    item for item in siblings
                    if item.suffix.lower() in self.SUPPORTED
                    and item not in raw_files
                    and item.stem.lower().replace(" ", "") not in derived_names
                    and not item.stem.lower().endswith("_raw")
                ),
                key=lambda item: (
                    {".pdf": 0, ".docx": 1, ".md": 2, ".markdown": 2, ".txt": 3}.get(
                        item.suffix.lower(), 9
                    ),
                    item.name.lower(),
                ),
            )
            if not candidates:
                continue
            relevant = next(
                (
                    item for item in siblings
                    if any(marker in item.stem.lower() for marker in (
                        "allrelevant", "functionalrequirements"
                    ))
                    and item.suffix.lower() in {".txt", ".md"}
                ),
                None,
            )
            pairs.append({
                "id": str(folder.relative_to(self.files_root)).replace("\\", "/"),
                "source_path": str(candidates[0]),
                "gold_path": str(gold),
                "relevant_path": str(relevant) if relevant else "",
            })
            if limit and len(pairs) >= limit:
                break
        return pairs


class PublicBenchmarkService:
    def __init__(self, data_root: Path, ebt: EBTRepository) -> None:
        self.data_root = data_root
        self.external_root = data_root / "external"
        self.report_root = data_root / "benchmark_reports"
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.ebt = ebt
        self.storyseek = StorySeekRepository(self.external_root / "storyseek")
        self.srs = SRSRepository(self.external_root / "public_srs")

    def catalog(self) -> Dict[str, Any]:
        ebt = self.ebt.status()
        ebt.update({
            "name": "EBT requirement-to-test trace benchmark",
            "purpose": ["rag", "case_generation"],
        })
        return {
            "datasets": [ebt, self.storyseek.status(), self.srs.status(), {
                "dataset_id": CRITIC_DATASET_ID,
                "name": "Programmatic critic mutation benchmark",
                "ready": self.ebt.ready,
                "source": "Derived from EBT without modifying source data",
                "license": "derived-evaluation",
                "purpose": ["quality_critic"],
                "sample_count": 4,
            }],
            "suites": [
                {"id": "ebt_generation", "dataset_id": "EBT-RAG-V1", "label": "EBT 用例生成"},
                {"id": "storyseek_pipeline", "dataset_id": STORYSEEK_DATASET_ID, "label": "StorySeek 需求与模块"},
                {"id": "srs_document", "dataset_id": SRS_DATASET_ID, "label": "SRS 文档解析"},
                {"id": "critic_mutation", "dataset_id": CRITIC_DATASET_ID, "label": "Critic 缺陷检测"},
            ],
            "reports": self.list_reports(limit=20),
        }

    def import_dataset(self, dataset_id: str) -> Dict[str, Any]:
        normalized = dataset_id.upper()
        if normalized in {"EBT", "EBT-RAG-V1"}:
            return self.ebt.download()
        if normalized in {"STORYSEEK", STORYSEEK_DATASET_ID}:
            return self.storyseek.download()
        if normalized in {"SRS", SRS_DATASET_ID}:
            return self.srs.download()
        raise ValueError("Unknown public dataset: {}".format(dataset_id))

    def list_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        reports = []
        for path in sorted(
            self.report_root.glob("BR-*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )[:limit]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            reports.append({
                key: payload.get(key)
                for key in (
                    "report_id", "suite", "dataset_id", "mode", "status",
                    "score", "sample_count", "created_at", "metrics",
                )
            })
        return reports

    def get_report(self, report_id: str) -> Dict[str, Any]:
        path = self.report_root / (report_id + ".json")
        if not path.is_file():
            raise ValueError("Benchmark report not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def run(
        self,
        suite: str,
        limit: int,
        split: str,
        mode: str,
        pipeline_factory: Callable[[Path], Any],
    ) -> Dict[str, Any]:
        limit = max(1, min(20, int(limit)))
        if mode not in {"offline", "live"}:
            raise ValueError("Benchmark mode must be offline or live")
        report_id = "BR-" + uuid.uuid4().hex[:12]
        workspace = self.data_root / "benchmark_runs" / report_id
        pipeline = pipeline_factory(workspace)
        if mode == "offline":
            pipeline.llm.api_key = ""
        runners = {
            "ebt_generation": lambda: self._run_ebt(pipeline, limit),
            "storyseek_pipeline": lambda: self._run_storyseek(
                pipeline, limit, split
            ),
            "srs_document": lambda: self._run_srs(pipeline, limit, mode),
            "critic_mutation": lambda: self._run_critic(pipeline),
        }
        if suite not in runners:
            raise ValueError("Unknown benchmark suite: {}".format(suite))
        payload = runners[suite]()
        payload.update({
            "report_id": report_id,
            "suite": suite,
            "mode": mode,
            "status": "completed",
            "created_at": utc_now(),
        })
        atomic_write(
            self.report_root / (report_id + ".json"),
            json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return payload

    def _run_ebt(self, pipeline: Any, limit: int) -> Dict[str, Any]:
        if not self.ebt.ready:
            raise ValueError("Import EBT before running this suite")
        artifacts, traces = self.ebt.load()
        links: Dict[str, List[str]] = {}
        for trace in traces:
            links.setdefault(trace.requirement_id, []).append(trace.test_case_id)
        samples = []
        for requirement_id in sorted(links, key=self.ebt._sort_key)[:limit]:
            if requirement_id not in artifacts:
                continue
            requirement = artifacts[requirement_id].content
            gold_tests = [
                artifacts[test_id].content
                for test_id in links[requirement_id]
                if test_id in artifacts
            ]
            try:
                project = pipeline.store.create_project(
                    "EBT requirement {}".format(requirement_id), requirement
                )
                project = pipeline.analyze(project)
                project = pipeline.plan_modules(project)
                project.module_tree.confirmed = True
                pipeline.store.save_project(project)
                project = pipeline.generate_cases(project)
                project = pipeline.review(project)
                generated = "\n".join(
                    case.title + "\n" + "\n".join(
                        step.action + " " + step.expected for step in case.steps
                    )
                    for case in project.cases
                )
                overlaps = [token_recall(test, generated) for test in gold_tests]
                samples.append({
                    "id": requirement_id,
                    "status": "passed",
                    "gold_test_count": len(gold_tests),
                    "generated_case_count": len(project.cases),
                    "linked_test_recall": average(
                        1.0 if score >= 0.35 else 0.0 for score in overlaps
                    ),
                    "mean_gold_token_recall": average(overlaps),
                    "traceability_ratio": average(
                        1.0 if case.requirement_ids and case.source_evidence else 0.0
                        for case in project.cases
                    ),
                    "assertion_ratio": average(
                        1.0 if step.expected.strip() else 0.0
                        for case in project.cases for step in case.steps
                    ),
                    "critic_score": project.review.score if project.review else 0,
                })
            except Exception as exc:
                samples.append({"id": requirement_id, "status": "failed", "error": str(exc)[:500]})
        passed = [item for item in samples if item["status"] == "passed"]
        metrics = {
            "task_success_rate": average(1.0 if item["status"] == "passed" else 0.0 for item in samples),
            "linked_test_recall": average(item["linked_test_recall"] for item in passed),
            "mean_gold_token_recall": average(item["mean_gold_token_recall"] for item in passed),
            "traceability_ratio": average(item["traceability_ratio"] for item in passed),
            "assertion_ratio": average(item["assertion_ratio"] for item in passed),
            "critic_score": average(item["critic_score"] / 100.0 for item in passed),
        }
        return self._report("EBT-RAG-V1", samples, metrics)

    def _run_storyseek(
        self, pipeline: Any, limit: int, split: str
    ) -> Dict[str, Any]:
        records = self.storyseek.load(split=split, limit=limit)
        samples = []
        for record in records:
            requirement = "\n".join(
                part for part in (
                    record["background"], record["problems"], record["solutions"],
                    "Goal: " + record["goal"],
                    "User story: As {}, I want {}, so that {}".format(
                        record["actor"], record["action"], record["expected_outcome"]
                    ),
                ) if part.strip(" :")
            )
            try:
                project = pipeline.store.create_project(
                    "StorySeek {}".format(record["id"]), requirement
                )
                project = pipeline.analyze(project)
                project = pipeline.plan_modules(project)
                analysis_text = project.analysis.model_dump_json()
                module_text = project.module_tree.model_dump_json()
                actor_tokens = tokenize(record["actor"])
                actual_actor_tokens = tokenize(" ".join(project.analysis.actors))
                samples.append({
                    "id": record["id"],
                    "project": record["project"],
                    "status": "passed",
                    "actor_accuracy": 1.0 if actor_tokens and actor_tokens <= actual_actor_tokens else 0.0,
                    "action_recall": token_recall(record["action"], analysis_text),
                    "outcome_recall": token_recall(record["expected_outcome"], analysis_text),
                    "goal_recall": token_recall(record["goal"], analysis_text),
                    "deliverable_module_recall": token_recall(record["deliverable"], module_text),
                    "module_count": len(project.module_tree.modules),
                })
            except Exception as exc:
                samples.append({"id": record["id"], "status": "failed", "error": str(exc)[:500]})
        passed = [item for item in samples if item["status"] == "passed"]
        metrics = {
            "task_success_rate": average(1.0 if item["status"] == "passed" else 0.0 for item in samples),
            "actor_accuracy": average(item["actor_accuracy"] for item in passed),
            "action_recall": average(item["action_recall"] for item in passed),
            "outcome_recall": average(item["outcome_recall"] for item in passed),
            "goal_recall": average(item["goal_recall"] for item in passed),
            "deliverable_module_recall": average(item["deliverable_module_recall"] for item in passed),
        }
        return self._report(STORYSEEK_DATASET_ID, samples, metrics, {"split": split})

    def _run_srs(self, pipeline: Any, limit: int, mode: str) -> Dict[str, Any]:
        if not self.srs.ready:
            raise ValueError("Import the public SRS dataset before running this suite")
        samples = []
        for pair in self.srs.pairs(limit=limit):
            try:
                source = Path(pair["source_path"])
                gold = Path(pair["gold_path"]).read_text(
                    encoding="utf-8", errors="replace"
                )
                document = pipeline.document_agent.run(
                    {"filename": source.name, "data": source.read_bytes()},
                    {"force_demo": mode == "offline"},
                )
                relevant = ""
                if pair["relevant_path"]:
                    relevant = Path(pair["relevant_path"]).read_text(
                        encoding="utf-8", errors="replace"
                    )
                samples.append({
                    "id": pair["id"],
                    "status": "passed",
                    "format": source.suffix.lower(),
                    "extraction_recall": token_recall(gold, document.raw_text),
                    "relevant_recall": token_recall(relevant, document.normalized_text) if relevant else None,
                    "compression_ratio": round(
                        len(document.normalized_text) / float(max(1, len(document.raw_text))), 4
                    ),
                    "warning_count": len(document.warnings),
                })
            except Exception as exc:
                samples.append({"id": pair["id"], "status": "failed", "error": str(exc)[:500]})
        passed = [item for item in samples if item["status"] == "passed"]
        relevant = [item for item in passed if item["relevant_recall"] is not None]
        metrics = {
            "task_success_rate": average(1.0 if item["status"] == "passed" else 0.0 for item in samples),
            "extraction_recall": average(item["extraction_recall"] for item in passed),
            "relevant_recall": average(item["relevant_recall"] for item in relevant),
            "compression_ratio": average(item["compression_ratio"] for item in passed),
            "warning_rate": average(1.0 if item["warning_count"] else 0.0 for item in passed),
        }
        return self._report(SRS_DATASET_ID, samples, metrics)

    def _run_critic(self, pipeline: Any) -> Dict[str, Any]:
        requirement_text = "A registered user submits a request and receives a visible result."
        if self.ebt.ready:
            artifacts, traces = self.ebt.load()
            if traces:
                requirement_text = artifacts[traces[0].requirement_id].content
        base_analysis = RequirementAnalysis(
            summary=requirement_text,
            atomic_requirements=[AtomicRequirement(
                id="REQ-001", statement=requirement_text, source_quote=requirement_text
            )],
        )
        base_tree = ModuleTree(modules=[TestModule(
            id="MOD-1", name="Core behavior", objective=requirement_text,
            requirement_ids=["REQ-001"], case_types=["functional", "boundary", "exception"],
        )], confirmed=True)
        base_cases = [
            TestCase(
                id="TC-{:03d}".format(index), module_id="MOD-1",
                title="{} scenario".format(case_type), case_type=case_type,
                requirement_ids=["REQ-001"], source_evidence=["EBT-REQ-001"],
                steps=[TestStep(action="Execute {} scenario".format(case_type), expected="The observable result matches the requirement")],
            )
            for index, case_type in enumerate(("functional", "boundary", "exception"), 1)
        ]

        scenarios = []
        mutations = []
        missing_assertion = deepcopy(base_cases)
        missing_assertion[0].steps[0].expected = ""
        mutations.append(("missing_assertion", base_analysis, base_tree, missing_assertion, "assertion"))
        missing_evidence = deepcopy(base_cases)
        missing_evidence[0].source_evidence = []
        mutations.append(("missing_evidence", base_analysis, base_tree, missing_evidence, "traceability"))
        uncovered = base_analysis.model_copy(deep=True)
        uncovered.atomic_requirements.append(AtomicRequirement(
            id="REQ-002", statement="A second requirement must be covered.", source_quote="A second requirement must be covered."
        ))
        mutations.append(("uncovered_requirement", uncovered, base_tree, deepcopy(base_cases), "requirement_coverage"))
        empty_tree = base_tree.model_copy(deep=True)
        empty_tree.modules.append(TestModule(
            id="MOD-2", name="Empty module", objective="Must contain cases", requirement_ids=[]
        ))
        mutations.append(("empty_module", base_analysis, empty_tree, deepcopy(base_cases), "module_coverage"))

        clean_result = self._critic_review(pipeline, base_analysis, base_tree, base_cases)
        for name, analysis, tree, cases, expected in mutations:
            result = self._critic_review(pipeline, analysis, tree, cases)
            categories = {item["category"] for item in result["review"]["findings"]}
            scenarios.append({
                "id": name,
                "status": "passed",
                "expected_category": expected,
                "detected": expected in categories,
                "detected_categories": sorted(categories),
            })
        metrics = {
            "defect_detection_recall": average(1.0 if item["detected"] else 0.0 for item in scenarios),
            "clean_false_positive_count": float(len(clean_result["review"]["findings"])),
            "clean_pass_rate": 1.0 if not clean_result["review"]["findings"] else 0.0,
        }
        return self._report(CRITIC_DATASET_ID, scenarios, metrics)

    @staticmethod
    def _critic_review(
        pipeline: Any,
        analysis: RequirementAnalysis,
        tree: ModuleTree,
        cases: List[TestCase],
    ) -> Dict[str, Any]:
        result, _ = pipeline.harness.execute(
            pipeline.quality_agent,
            {
                "target": "case",
                "analysis": analysis.model_dump(),
                "module_tree": tree.model_dump(),
                "cases": [case.model_dump() for case in cases],
            },
            {"force_demo": True},
        )
        return result

    @staticmethod
    def _report(
        dataset_id: str,
        samples: List[Dict[str, Any]],
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        score_metrics = [
            value for key, value in metrics.items()
            if 0.0 <= value <= 1.0
            and not key.endswith("_count")
            and key not in {"compression_ratio", "warning_rate"}
        ]
        payload = {
            "dataset_id": dataset_id,
            "sample_count": len(samples),
            "score": round(100 * mean(score_metrics)) if score_metrics else 0,
            "metrics": {key: round(value, 4) for key, value in metrics.items()},
            "samples": samples,
        }
        payload.update(extra or {})
        return payload
