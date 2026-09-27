import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .models import EvaluationReport, GoldenDataset, ProjectState


EQUIPMENT_BASELINE = GoldenDataset(
    id="EQUIPMENT-BASELINE-V1",
    name="设备借用工单核心质量基准",
    ticket_type="EQUIPMENT_BORROW",
    expected_interfaces=[
        "CreateBorrowRequest",
        "GetBorrowRequest",
        "CheckEquipmentAvailability",
        "UpdateBorrowNote",
    ],
    expected_case_types=[
        "api_contract",
        "realtime_data",
        "review_operation",
        "state_transition",
        "permission",
        "event_consistency",
        "exception",
        "compatibility",
    ],
    required_terms=[
        "request_id",
        "equipment_summary",
        "error.message",
        "borrow_id",
        "READY",
        "DECLINED",
        "灰度",
    ],
    forbidden_terms=[],
)


DATASETS: Dict[str, GoldenDataset] = {EQUIPMENT_BASELINE.id: EQUIPMENT_BASELINE}


def list_datasets() -> List[GoldenDataset]:
    return list(DATASETS.values())


def dataset_for(ticket_type: str) -> Optional[GoldenDataset]:
    return next(
        (dataset for dataset in DATASETS.values() if dataset.ticket_type == ticket_type),
        None,
    )


class OfflineEvaluationAgent:
    name = "offline_evaluation"

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> EvaluationReport:
        project = ProjectState.model_validate(payload["project"])
        dataset = GoldenDataset.model_validate(payload["dataset"])
        if not project.analysis:
            raise ValueError("Requirement analysis is required before evaluation")

        interface_names = {item.name for item in project.analysis.interfaces}
        case_types = {case.case_type for case in project.cases}
        corpus = project.analysis.model_dump_json() + "\n" + "\n".join(
            case.model_dump_json() for case in project.cases
        )
        interface_recall, missing_interfaces = self._recall(
            dataset.expected_interfaces, interface_names
        )
        type_recall, missing_types = self._recall(dataset.expected_case_types, case_types)
        term_hits = [term for term in dataset.required_terms if term.lower() in corpus.lower()]
        term_recall = (
            len(term_hits) / len(dataset.required_terms) if dataset.required_terms else 1.0
        )
        missing_terms = [term for term in dataset.required_terms if term not in term_hits]
        violations = [
            term for term in dataset.forbidden_terms if term.lower() in corpus.lower()
        ]

        evidence_ratio = (
            sum(1 for case in project.cases if case.source_evidence) / len(project.cases)
            if project.cases
            else 0.0
        )
        all_steps = [step for case in project.cases for step in case.steps]
        assertion_ratio = (
            sum(
                1
                for step in all_steps
                if step.expected.strip()
                and "需人工确认" not in step.expected
                and len(step.expected.strip()) >= 6
            )
            / len(all_steps)
            if all_steps
            else 0.0
        )
        normalized_titles = [
            re.sub(r"\s+", "", case.title).lower() for case in project.cases
        ]
        duplicate_count = sum(
            count - 1 for count in Counter(normalized_titles).values() if count > 1
        )
        duplicate_ratio = (
            duplicate_count / len(project.cases) if project.cases else 0.0
        )
        violation_ratio = (
            len(violations) / len(dataset.forbidden_terms)
            if dataset.forbidden_terms
            else 0.0
        )
        score = round(
            100
            * (
                interface_recall * 0.20
                + type_recall * 0.20
                + term_recall * 0.20
                + evidence_ratio * 0.15
                + assertion_ratio * 0.15
                + (1.0 - duplicate_ratio) * 0.05
                + (1.0 - violation_ratio) * 0.05
            )
        )
        missing_items = (
            ["interface:{}".format(item) for item in missing_interfaces]
            + ["case_type:{}".format(item) for item in missing_types]
            + ["term:{}".format(item) for item in missing_terms]
        )
        return EvaluationReport(
            dataset_id=dataset.id,
            score=max(0, min(100, score)),
            metrics={
                "interface_recall": round(interface_recall, 4),
                "case_type_recall": round(type_recall, 4),
                "required_term_recall": round(term_recall, 4),
                "evidence_ratio": round(evidence_ratio, 4),
                "assertion_ratio": round(assertion_ratio, 4),
                "duplicate_ratio": round(duplicate_ratio, 4),
                "forbidden_term_ratio": round(violation_ratio, 4),
            },
            missing_items=missing_items,
            violations=["forbidden_term:{}".format(item) for item in violations],
        )

    @staticmethod
    def _recall(expected: List[str], actual: set) -> Any:
        if not expected:
            return 1.0, []
        missing = [item for item in expected if item not in actual]
        return (len(expected) - len(missing)) / len(expected), missing
