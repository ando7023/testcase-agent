from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal

from pydantic import BaseModel, Field, field_validator, computed_field


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class RequirementInput(BaseModel):
    title: str = "Untitled requirement"
    content: str = Field(min_length=10)
    context: str = ""


class OCRSpan(BaseModel):
    text: str
    bbox: List[List[float]] = Field(default_factory=list)
    confidence: float = 0.0


class DocumentChunk(BaseModel):
    id: str
    page: Optional[int] = None
    section: str = ""
    content: str
    extraction_method: str = "native"
    ocr_confidence: Optional[float] = None
    ocr_spans: List[OCRSpan] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    filename: str
    document_type: str = "unknown"
    normalized_text: str
    raw_text: str = ""
    chunks: List[DocumentChunk] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    extraction_summary: Dict[str, Any] = Field(default_factory=dict)
    compression_summary: Dict[str, Any] = Field(default_factory=dict)
    section_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    excluded_sections: List[Dict[str, Any]] = Field(default_factory=list)


class AtomicRequirement(BaseModel):
    id: str
    statement: str
    category: str = "functional"
    actors: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    source_quote: str = ""


class InterfaceContract(BaseModel):
    name: str
    purpose: str
    direction: str = "internal"
    required_fields: List[str] = Field(default_factory=list)
    success_condition: str = ""
    failure_condition: str = ""
    evidence_ids: List[str] = Field(default_factory=list)


class StateTransition(BaseModel):
    from_state: str
    action: str
    to_state: str
    actor: str = ""
    terminal: bool = False
    evidence_ids: List[str] = Field(default_factory=list)


class EventContract(BaseModel):
    topic: str
    event: str
    statuses: List[str] = Field(default_factory=list)
    correlation_key: str = ""
    delivery_risks: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class RequirementAmbiguity(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    question: str = Field(min_length=1, max_length=1500)


class ClarificationItem(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["execution_detail", "out_of_scope", "behavior_blocker"]
    question: str = Field(min_length=1, max_length=1500)
    requirement_ids: List[str] = Field(default_factory=list)
    affected_scenarios: List[str] = Field(min_length=1, max_length=20)
    source_quote: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=1500)
    ambiguity_ids: List[str] = Field(default_factory=list)


class RequirementAnalysis(BaseModel):
    summary: str
    actors: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    business_rules: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    ambiguities: List[Union[str, RequirementAmbiguity]] = Field(default_factory=list)
    clarification_items: List[ClarificationItem] = Field(default_factory=list)
    atomic_requirements: List[AtomicRequirement] = Field(default_factory=list)
    risk_hints: List[str] = Field(default_factory=list)
    ticket_types: List[str] = Field(default_factory=list)
    interfaces: List[InterfaceContract] = Field(default_factory=list)
    state_transitions: List[StateTransition] = Field(default_factory=list)
    events: List[EventContract] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    retrieved_evidence_ids: List[str] = Field(default_factory=list)


class ModuleSpec(BaseModel):
    """Data-driven module planning spec; lives in a domain knowledge pack, not in agent code."""

    name: str
    keywords: List[str] = Field(default_factory=list)
    objective: str = ""
    case_types: List[str] = Field(default_factory=list)


class DomainFacts(BaseModel):
    """Structured domain contracts for one ticket type, loaded from its knowledge pack."""

    ticket_type: str
    extra_actors: List[str] = Field(default_factory=list)
    interfaces: List[InterfaceContract] = Field(default_factory=list)
    state_transitions: List[StateTransition] = Field(default_factory=list)
    events: List[EventContract] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    required_case_types: List[str] = Field(default_factory=list)
    module_specs: List[ModuleSpec] = Field(default_factory=list)


class TestModule(BaseModel):
    id: str
    name: str
    objective: str
    requirement_ids: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    case_types: List[str] = Field(default_factory=list)
    children: List["TestModule"] = Field(default_factory=list)


class ModuleTree(BaseModel):
    modules: List[TestModule]
    coverage_notes: List[str] = Field(default_factory=list)
    confirmed: bool = False  # Optional human provenance; not a generation prerequisite.


class ConversationDecision(BaseModel):
    key: str
    content: str
    kind: str = "decision"
    status: str = "active"
    source_message_id: str = ""
    source_version_id: str = ""
    target_module_id: str = ""


class ConversationDecisionState(BaseModel):
    processed_count: int = 0
    recent_start: int = 0
    entries: List[ConversationDecision] = Field(default_factory=list)


class ModuleConversationMessage(BaseModel):
    id: str
    role: str
    content: str
    mode: str = "chat"
    target_module_id: str = ""
    version_id: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class ModuleTreeVersion(BaseModel):
    id: str
    mode: str
    instruction: str = ""
    target_module_id: str = ""
    module_tree: ModuleTree
    memory_state: Optional[ConversationDecisionState] = None
    created_at: str = Field(default_factory=utc_now_iso)


class ModuleOperationRequest(BaseModel):
    mode: str = "full"
    instruction: str = ""
    target_module_id: str = ""


class MindMapNode(BaseModel):
    id: str
    label: str
    node_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    children: List["MindMapNode"] = Field(default_factory=list)


class MindMapDocument(BaseModel):
    version: str = "mindmap-v1"
    tool: str = "mindmap_conversion"
    project_id: str
    view: str
    root: MindMapNode
    stats: Dict[str, int] = Field(default_factory=dict)


class ModuleReviewFinding(BaseModel):
    severity: str
    category: str
    message: str
    module_id: Optional[str] = None
    detail: Optional[str] = None


class ModuleReviewReport(BaseModel):
    score: int = Field(ge=0, le=100)
    findings: List[ModuleReviewFinding] = Field(default_factory=list)
    rounds: int = 0


class TestStep(BaseModel):
    action: str
    expected: str


class TestCase(BaseModel):
    id: str
    module_id: str
    title: str
    priority: str = "P1"
    case_type: str = "functional"
    preconditions: List[str] = Field(default_factory=list)
    steps: List[TestStep]
    test_data: Dict[str, Any] = Field(default_factory=dict)
    risk_tags: List[str] = Field(default_factory=list)
    requirement_ids: List[str] = Field(default_factory=list)
    source_evidence: List[str] = Field(default_factory=list)
    automation_feasibility: str = "medium"
    ai_friendly: bool = True
    review_status: str = "pending"
    human_status: str = "pending"
    generated_by: str = "generation"

    @field_validator("case_type", mode="before")
    @classmethod
    def normalize_case_type(cls, value):
        if not isinstance(value, str):
            return value
        value = value.strip()
        aliases = {
            "正常": "functional", "功能": "functional", "normal": "functional",
            "happy_path": "functional", "边界": "boundary", "异常": "exception",
            "权限": "permission", "并发幂等": "state_transition",
            "concurrency_idempotency": "state_transition", "状态转换": "state_transition",
        }
        return aliases.get(value.lower(), value)


class CaseConversationMessage(BaseModel):
    id: str
    role: str
    content: str
    mode: str = "chat"
    target_module_id: str = ""
    version_id: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class CaseSetVersion(BaseModel):
    id: str
    mode: str
    instruction: str = ""
    target_module_id: str = ""
    cases: List[TestCase] = Field(default_factory=list)
    memory_state: Optional[ConversationDecisionState] = None
    created_at: str = Field(default_factory=utc_now_iso)


class ReviewFinding(BaseModel):
    severity: str
    category: str
    message: str
    module_id: Optional[str] = None
    case_id: Optional[str] = None
    detail: Optional[str] = None  # machine-readable payload for the revision loop
    disposition: Literal["legacy", "defect", "clarification", "suggestion"] = "legacy"
    issue_type: str = ""
    issue_id: str = ""
    evidence: str = ""
    requirement_ids: List[str] = Field(default_factory=list)
    clarification_kind: Literal["unspecified", "execution_detail", "out_of_scope", "behavior_blocker"] = "unspecified"
    clarification_reason: str = ""
    clarification_basis_verified: bool = False  # Set by runtime, never trusted from model JSON.


class ReviewReport(BaseModel):
    score: int = Field(ge=0, le=100)
    findings: List[ReviewFinding] = Field(default_factory=list)
    added_case_ids: List[str] = Field(default_factory=list)
    coverage_by_type: Dict[str, int] = Field(default_factory=dict)
    requirement_coverage: Dict[str, int] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    agent: str
    started_at: str
    duration_ms: int
    status: str
    mode: str
    input_summary: str
    output_summary: str
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    react_steps: int = 0


class TraceEvent(BaseModel):
    name: str
    timestamp: str = Field(default_factory=utc_now_iso)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class TraceSpan(BaseModel):
    id: str
    trace_id: str
    parent_span_id: str = ""
    name: str
    kind: str = "internal"
    status: str = "running"
    started_at: str = Field(default_factory=utc_now_iso)
    ended_at: str = ""
    duration_ms: int = 0
    input_summary: str = ""
    output_summary: str = ""
    error: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    usage: Dict[str, Any] = Field(default_factory=dict)
    events: List[TraceEvent] = Field(default_factory=list)


class TraceRun(BaseModel):
    trace_id: str
    operation: str
    project_id: str = ""
    transport: str = "internal"
    status: str = "running"
    started_at: str = Field(default_factory=utc_now_iso)
    ended_at: str = ""
    duration_ms: int = 0
    root_span_id: str = ""
    spans: List[TraceSpan] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class CaseFeedback(BaseModel):
    case_id: str
    action: str  # adopted | edited | rejected
    reason: str = ""
    category: str = ""  # coverage | quality | maintenance
    case_fingerprint: str = ""
    case_version_id: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class ProjectMetrics(BaseModel):
    total_cases: int = 0
    adopted: int = 0
    edited: int = 0
    rejected: int = 0
    pending: int = 0
    adoption_rate: float = 0.0
    generation_rate: float = 0.0
    modification_rate: float = 0.0
    badcase_by_category: Dict[str, int] = Field(default_factory=dict)


class GoldenDataset(BaseModel):
    id: str
    name: str
    ticket_type: str = "COMMON"
    expected_interfaces: List[str] = Field(default_factory=list)
    expected_case_types: List[str] = Field(default_factory=list)
    required_terms: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    dataset_id: str
    score: int = Field(ge=0, le=100)
    metrics: Dict[str, float] = Field(default_factory=dict)
    missing_items: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)


class ScenarioRule(BaseModel):
    id: str
    ticket_type: str
    scene: str
    rule: str
    case_type: str = "functional"
    source: str = "human_feedback"
    sources: List[str] = Field(default_factory=list)
    support_count: int = 1
    created_at: str = Field(default_factory=utc_now_iso)


class ScenarioTemplate(BaseModel):
    id: str
    ticket_type: str
    name: str
    rules: List[str] = Field(default_factory=list)
    common_terms: List[str] = Field(default_factory=list)
    support_count: int = 0
    version: int = 1
    status: str = "candidate"
    updated_at: str = Field(default_factory=utc_now_iso)


class AdaptiveMemoryRecord(BaseModel):
    id: str
    content: str = Field(min_length=1)
    memory_type: str = "semantic"
    user_id: str = ""
    project_id: str = ""
    agent_id: str = ""
    run_id: str = ""
    source_run_id: str = ""
    source_version_id: str = ""
    fact_key: str = ""
    supersedes: str = ""
    superseded_by: str = ""
    valid_from: str = ""
    valid_to: str = ""
    invalidation_reason: str = ""
    ticket_type: str = "COMMON"
    source: str = "manual"
    source_ref: str = ""
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    content_hash: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    access_count: int = 0
    last_accessed_at: str = ""
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class AdaptiveMemoryHit(BaseModel):
    memory: AdaptiveMemoryRecord
    score: float = 0.0
    lexical_score: float = 0.0
    vector_score: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class AdaptiveMemoryContext(BaseModel):
    query: str
    context: str = ""
    hits: List[AdaptiveMemoryHit] = Field(default_factory=list)
    candidate_count: int = 0
    selected_count: int = 0
    latency_ms: int = 0
    estimated_full_tokens: int = 0
    selected_tokens: int = 0
    token_saving_ratio: float = 0.0
    retrieval_mode: str = "bm25+vector+rrf+scope_rerank"
    embedding_provider: str = ""


class ProjectState(BaseModel):
    id: str
    title: str
    requirement: str
    context: str = ""
    clarification_policy: Literal["strict", "evidence_only"] = "strict"
    phase: str = "draft"
    memory_epoch: str = ""
    source_documents: List[ParsedDocument] = Field(default_factory=list)
    analysis: Optional[RequirementAnalysis] = None
    module_tree: Optional[ModuleTree] = None
    module_review: Optional[ModuleReviewReport] = None
    module_conversation: List[ModuleConversationMessage] = Field(default_factory=list)
    module_versions: List[ModuleTreeVersion] = Field(default_factory=list)
    module_memory_summary: str = ""
    module_decision_state: ConversationDecisionState = Field(default_factory=ConversationDecisionState)
    cases: List[TestCase] = Field(default_factory=list)
    case_conversation: List[CaseConversationMessage] = Field(default_factory=list)
    case_versions: List[CaseSetVersion] = Field(default_factory=list)
    case_memory_summary: str = ""
    case_decision_state: ConversationDecisionState = Field(default_factory=ConversationDecisionState)
    review: Optional[ReviewReport] = None
    evaluation: Optional[EvaluationReport] = None
    feedback: List[CaseFeedback] = Field(default_factory=list)
    traces: List[AgentTrace] = Field(default_factory=list)
    trace_run_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)

    @computed_field
    @property
    def human_acceptance(self) -> Dict[str, Any]:
        from .feedback import acceptance_summary
        return acceptance_summary(self)


class KnowledgeDocument(BaseModel):
    id: str
    title: str
    content: str
    doc_type: str = "business_rule"
    tags: List[str] = Field(default_factory=list)
    source: str = ""
    section: str = ""
    page: Optional[int] = None
    chunk_index: int = 0
    source_id: str = ""
    parent_id: str = ""
    chunk_level: str = "child"
    chunking_version: str = "legacy-v1"
    chunk_version: int = 1
    section_path: List[str] = Field(default_factory=list)
    supersedes: List[str] = Field(default_factory=list)
    superseded_by: List[str] = Field(default_factory=list)
    version: str = "1.0"
    effective_at: str = ""
    expires_at: str = ""
    status: str = "active"
    checksum: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class KnowledgeHit(BaseModel):
    document_id: str
    title: str
    content: str
    score: float
    doc_type: str = ""
    source: str = ""
    section: str = ""
    page: Optional[int] = None
    ticket_type: str = "COMMON"
    version: str = "1.0"
    parent_id: str = ""
    parent_title: str = ""
    parent_content: str = ""
    chunk_version: int = 1
    chunking_version: str = "legacy-v1"
    rank: int = 0
    lexical_score: float = 0.0
    vector_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    matched_terms: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


class KnowledgeContext(BaseModel):
    query: str
    domain: str = "general"
    ticket_type: str = "GENERAL_TICKET"
    has_specific_knowledge: bool = False
    expanded_terms: List[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid"
    candidate_count: int = 0
    filtered_count: int = 0
    trace: Dict[str, Any] = Field(default_factory=dict)
    hits: List[KnowledgeHit] = Field(default_factory=list)


class RetrievalEvalCase(BaseModel):
    id: str
    query: str
    ticket_type: str
    expected_document_ids: List[str] = Field(default_factory=list)


class RetrievalDataset(BaseModel):
    id: str
    name: str
    cases: List[RetrievalEvalCase] = Field(default_factory=list)


class RetrievalCaseResult(BaseModel):
    case_id: str
    query: str
    expected_document_ids: List[str] = Field(default_factory=list)
    retrieved_document_ids: List[str] = Field(default_factory=list)
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    contamination_count: int = 0
    stale_count: int = 0


class RetrievalEvaluationReport(BaseModel):
    dataset_id: str
    k: int = 5
    recall_at_k: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    contamination_rate: float = 0.0
    stale_hit_rate: float = 0.0
    cases: List[RetrievalCaseResult] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


TestModule.model_rebuild()
MindMapNode.model_rebuild()
