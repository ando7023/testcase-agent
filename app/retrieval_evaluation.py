from statistics import mean
from typing import Dict, List

from .agents import KnowledgeRetrievalAgent
from .models import (
    KnowledgeDocument,
    RetrievalCaseResult,
    RetrievalDataset,
    RetrievalEvalCase,
    RetrievalEvaluationReport,
)


EQUIPMENT_RAG_DATASET = RetrievalDataset(
    id="EQUIPMENT-RAG-V1",
    name="设备借用 工单知识检索黄金集",
    cases=[
        RetrievalEvalCase(
            id="EQUIPMENT-RAG-001",
            query="设备借用 Check 和 Edit 分别调用什么接口，失败如何提示？",
            ticket_type="EQUIPMENT_BORROW",
            expected_document_ids=["DEMO-GUIDE-OPERATIONS"],
        ),
        RetrievalEvalCase(
            id="EQUIPMENT-RAG-002",
            query="EQUIPMENT_BORROW 工单创建接口有哪些关键字段和幂等约束？",
            ticket_type="EQUIPMENT_BORROW",
            expected_document_ids=["DEMO-GUIDE-CREATE"],
        ),
        RetrievalEvalCase(
            id="EQUIPMENT-RAG-003",
            query="设备借用 工单 READY DECLINED 后 Kafka topic、过滤事件和关联键是什么？",
            ticket_type="EQUIPMENT_BORROW",
            expected_document_ids=["DEMO-GUIDE-EVENTS"],
        ),
        RetrievalEvalCase(
            id="EQUIPMENT-RAG-004",
            query="设备借用 工单上线灰度、历史兼容和回滚需要验证什么？",
            ticket_type="EQUIPMENT_BORROW",
            expected_document_ids=["DEMO-GUIDE-RELEASE"],
        ),
        RetrievalEvalCase(
            id="EQUIPMENT-RAG-005",
            query="设备借用 工单详情每次打开如何获取最新信息，展示哪些区域？",
            ticket_type="EQUIPMENT_BORROW",
            expected_document_ids=["DEMO-GUIDE-DETAIL"],
        ),
        RetrievalEvalCase(
            id="EQUIPMENT-RAG-006",
            query="设备借用 工单 申请人提交、Librarian 批准或拒绝 的审核流转顺序是什么？",
            ticket_type="EQUIPMENT_BORROW",
            expected_document_ids=["DEMO-GUIDE-FLOW"],
        ),
    ],
)

RETRIEVAL_DATASETS: Dict[str, RetrievalDataset] = {
    EQUIPMENT_RAG_DATASET.id: EQUIPMENT_RAG_DATASET
}


class RetrievalEvaluationAgent:
    name = "retrieval_evaluation"

    def __init__(self, retrieval_agent: KnowledgeRetrievalAgent) -> None:
        self.retrieval_agent = retrieval_agent

    def run(
        self,
        documents: List[KnowledgeDocument],
        dataset_id: str = "EQUIPMENT-RAG-V1",
        k: int = 5,
    ) -> RetrievalEvaluationReport:
        dataset = RETRIEVAL_DATASETS.get(dataset_id)
        if dataset is None:
            raise ValueError("Unknown retrieval dataset: {}".format(dataset_id))
        return self.run_dataset(documents, dataset, k)

    def run_dataset(
        self,
        documents: List[KnowledgeDocument],
        dataset: RetrievalDataset,
        k: int = 5,
    ) -> RetrievalEvaluationReport:
        if not dataset.cases:
            raise ValueError("Retrieval dataset contains no evaluation cases")
        results: List[RetrievalCaseResult] = []
        total_hits = 0
        contamination = 0
        stale = 0
        for case in dataset.cases:
            context = self.retrieval_agent.run(
                case.query, {"documents": documents, "force_demo": True}
            )
            hits = context.hits[:k]
            retrieved_ids = [hit.document_id for hit in hits]
            expected = set(case.expected_document_ids)
            found = expected & set(retrieved_ids)
            recall = len(found) / float(max(1, len(expected)))
            reciprocal_rank = 0.0
            for rank, document_id in enumerate(retrieved_ids, 1):
                if document_id in expected:
                    reciprocal_rank = 1.0 / rank
                    break
            case_contamination = sum(
                1
                for hit in hits
                if hit.ticket_type not in {case.ticket_type, "COMMON"}
            )
            case_stale = 0
            total_hits += len(hits)
            contamination += case_contamination
            stale += case_stale
            results.append(
                RetrievalCaseResult(
                    case_id=case.id,
                    query=case.query,
                    expected_document_ids=case.expected_document_ids,
                    retrieved_document_ids=retrieved_ids,
                    recall_at_k=round(recall, 4),
                    reciprocal_rank=round(reciprocal_rank, 4),
                    contamination_count=case_contamination,
                    stale_count=case_stale,
                )
            )
        return RetrievalEvaluationReport(
            dataset_id=dataset.id,
            k=k,
            recall_at_k=round(mean(item.recall_at_k for item in results), 4),
            hit_rate=round(
                mean(1.0 if item.reciprocal_rank else 0.0 for item in results), 4
            ),
            mrr=round(mean(item.reciprocal_rank for item in results), 4),
            contamination_rate=round(contamination / float(max(1, total_hits)), 4),
            stale_hit_rate=round(stale / float(max(1, total_hits)), 4),
            cases=results,
        )
