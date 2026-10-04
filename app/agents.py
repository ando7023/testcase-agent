from .clarification_policy import policy_system, validate_analysis, ambiguity_records
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import ValidationError

from .domain_registry import (
    GENERIC_TICKET_MODULE_SPECS,
    GENERIC_TICKET_REQUIRED_CASE_TYPES,
    detect_ticket_type,
    get_domain_facts,
)
from .knowledge import HashingEmbeddingProvider, KnowledgeBase, OpenAIEmbeddingProvider, SearchResult
from .llm import LLMError, OpenAICompatibleClient
from .tooling import ReActRuntime
from .models import (
    AgentTrace,
    AtomicRequirement,
    DomainFacts,
    KnowledgeContext,
    KnowledgeHit,
    ModuleReviewFinding,
    ModuleReviewReport,
    ModuleSpec,
    ModuleTree,
    RequirementAnalysis,
    RequirementInput,
    ReviewFinding,
    ReviewReport,
    TestCase,
    TestModule,
    TestStep,
    utc_now_iso,
)
from .skills import TestSkill, select_skills, resolve_skills
from .review_policy import FIXABLE_CATEGORIES, ISSUE_TYPES, is_auto_fixable, is_blocking, issue_key, contains_evidence


UNASSIGNED_MODULE_NAME = "未归类需求"

# Retrieval diversity: knowledge kinds and per-kind quotas (Kuaishou-style typed recall).
KIND_BY_DOC_TYPE = {
    "case_example": "example",
    "defect": "defect",
    "test_standard": "standard",
    "scenario_rule": "rule",
    "scenario_template": "template",
}
KIND_QUOTAS = {
    "fact": 8, "standard": 2, "defect": 3, "example": 3, "rule": 3, "template": 2
}
RETRIEVAL_LIMIT = 12


def compact(text: str, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?:\r?\n)+|[。；;!?！？]", text)
    return [part.strip(" -\t0123456789.、）)") for part in parts if len(part.strip()) >= 4]


class Agent(ABC):
    name = "agent"

    def __init__(self, llm: OpenAICompatibleClient) -> None:
        self.llm = llm

    @abstractmethod
    def run(self, payload: Any, context: Dict[str, Any]) -> Any:
        raise NotImplementedError


class AgentHarness:
    """Runs agents consistently and captures auditable execution traces."""

    def __init__(self, tracer: Any = None) -> None:
        self.tracer = tracer
        self.on_event = None

    def execute(self, agent: Agent, payload: Any, context: Dict[str, Any]) -> Tuple[Any, AgentTrace]:
        started = time.time()
        started_at = utc_now_iso()
        mode = "llm" if agent.llm.enabled else "demo"
        if self.on_event:
            self.on_event({"event": "agent_start", "agent": agent.name, "mode": mode})
        error: Optional[str] = None
        span_context = (
            self.tracer.span(
                "agent." + agent.name,
                kind="agent",
                attributes={"agent": agent.name, "mode": mode},
                input_value=payload,
            )
            if self.tracer
            else nullcontext(None)
        )
        with span_context as span:
            try:
                output = agent.run(payload, context)
                status = "success"
            except LLMError as exc:
                if getattr(self, "allow_fallback", True) is False:
                    if self.on_event:
                        self.on_event({"event": "agent_end", "agent": agent.name, "status": "error"})
                    raise
                error = str(exc)
                mode = "fallback"
                fallback_context = dict(context)
                fallback_context["force_demo"] = True
                output = agent.run(payload, fallback_context)
                status = "fallback_success"
            if span:
                span.status = "fallback" if mode == "fallback" else status
                span.error = error
                span.attributes.update({"mode": mode, "status": status})
                span.output_summary = compact(str(output), 2000)

        duration_ms = int((time.time() - started) * 1000)
        output_tool_calls = output.get("tool_calls", []) if isinstance(output, dict) else []
        output_react_steps = output.get("react_steps", 0) if isinstance(output, dict) else 0
        tool_calls = list(getattr(agent, "last_tool_calls", []) or output_tool_calls)
        react_steps = int(getattr(agent, "last_react_steps", 0) or output_react_steps)
        trace = AgentTrace(
            agent=agent.name,
            started_at=started_at,
            duration_ms=duration_ms,
            status=status,
            mode=mode,
            input_summary=compact(str(payload)),
            output_summary=compact(str(output)),
            error=error,
            tool_calls=tool_calls,
            react_steps=react_steps,
        )
        if self.on_event:
            self.on_event({"event": "agent_end", "agent": agent.name, "status": status})
        return output, trace


class ToolResearchAgent(Agent):
    """Bounded ReAct research node with read-only tool permissions."""

    def __init__(
        self,
        llm: OpenAICompatibleClient,
        runtime: ReActRuntime,
        name: str,
        profile: str,
    ) -> None:
        super().__init__(llm)
        self.runtime = runtime
        self.name = name
        self.profile = profile

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query", ""))
        ticket_type = str(payload.get("ticket_type", "GENERAL_TICKET"))
        fallback_plan = [
            ("search_knowledge", {"query": query}),
            ("get_domain_facts", {"ticket_type": ticket_type}),
            ("get_team_memory", {"ticket_type": ticket_type}),
        ]
        if self.profile == "case":
            fallback_plan.insert(
                1,
                (
                    "get_test_skills",
                    {"text": str(payload.get("skill_text", query))},
                ),
            )
        return self.runtime.run(
            self.name,
            policy_system(str(payload.get("objective") or query), context),
            fallback_plan,
            force_demo=bool(context.get("force_demo")),
        )

class KnowledgeRetrievalAgent(Agent):
    name = "knowledge_retrieval"

    def __init__(
        self, llm: OpenAICompatibleClient, index_path: Optional[Path] = None
    ) -> None:
        super().__init__(llm)
        self.index_path = index_path

    def run(self, payload: str, context: Dict[str, Any]) -> KnowledgeContext:
        all_documents = context.get("documents", [])
        parent_by_id = {
            document.id: document
            for document in all_documents
            if document.chunk_level == "parent" and document.status == "active"
        }
        definition = detect_ticket_type(payload)
        normalized = payload.lower()
        domain = "ticket" if any(
            token in normalized
            for token in ["工单", "ticket", "appeal", "设备借用", "p2p", "风控", "审核", "librarian", "equipment_borrow"]
        ) else "general"
        expanded_terms = definition.retrieval_terms if domain == "ticket" else []
        documents = list(all_documents)
        # Scope is supplied by the caller, never inferred from the model's query.
        project_id = context.get("project_id", "")
        documents = [d for d in documents if d.doc_type != "case_example" or
                     (project_id and d.metadata.get("scope") == "project" and
                      d.metadata.get("project_id") == project_id)]
        filters: Dict[str, Any] = {"status": "active"}
        if domain == "ticket":
            allowed_ticket_types = {definition.key, "COMMON"}
            filters["ticket_type"] = sorted(allowed_ticket_types)
            documents = [
                document
                for document in documents
                if document.metadata.get("ticket_type", "COMMON")
                in allowed_ticket_types
            ]

        before_temporal_filter = len(documents)
        now = utc_now_iso()
        documents = [
            document
            for document in documents
            if document.status == "active"
            and document.chunk_level != "parent"
            and document.metadata.get("indexable", True) is not False
            and document.metadata.get("status", "active") == "active"
            and (not document.effective_at or document.effective_at <= now)
            and (not document.expires_at or document.expires_at > now)
        ]
        filtered_count = len(all_documents) - len(documents)
        filters["temporal_filtered"] = before_temporal_filter - len(documents)

        query = payload + " " + " ".join(expanded_terms)
        provider_name = os.getenv("RAG_EMBEDDING_PROVIDER", "hashing").lower()
        if provider_name in {"openai", "remote"} and self.llm.enabled:
            provider = OpenAIEmbeddingProvider(self.llm)
        else:
            provider = HashingEmbeddingProvider()
        knowledge_base = KnowledgeBase(
            documents,
            embedding_provider=provider,
            index_path=self.index_path,
        )
        ranked = knowledge_base.search_detailed(query, limit=24)
        hits = self._diversify(ranked)
        trace = dict(knowledge_base.last_trace)
        trace["filters"] = filters
        trace["expanded_terms"] = expanded_terms
        return KnowledgeContext(
            query=compact(payload, 500),
            domain=domain,
            ticket_type=definition.key,
            has_specific_knowledge=domain == "ticket" and definition.has_specific_knowledge,
            expanded_terms=expanded_terms,
            retrieval_mode=trace.get("mode", "hybrid"),
            candidate_count=int(trace.get("candidate_count", 0)),
            filtered_count=filtered_count,
            trace=trace,
            hits=[
                KnowledgeHit(
                    document_id=item.document.id,
                    title=item.document.title,
                    content=item.document.content,
                    score=item.score,
                    doc_type=item.document.doc_type,
                    source=item.document.source,
                    section=item.document.section,
                    page=item.document.page,
                    ticket_type=item.document.metadata.get("ticket_type", "COMMON"),
                    version=item.document.version,
                    parent_id=item.document.parent_id,
                    parent_title=(
                        parent_by_id[item.document.parent_id].title
                        if item.document.parent_id in parent_by_id
                        else ""
                    ),
                    parent_content=(
                        parent_by_id[item.document.parent_id].content
                        if item.document.parent_id in parent_by_id
                        else ""
                    ),
                    chunk_version=item.document.chunk_version,
                    chunking_version=item.document.chunking_version,
                    rank=rank,
                    lexical_score=item.lexical_score,
                    vector_score=item.vector_score,
                    fusion_score=item.fusion_score,
                    rerank_score=item.rerank_score,
                    matched_terms=item.matched_terms,
                    reasons=item.reasons,
                )
                for rank, item in enumerate(hits, 1)
            ],
        )

    @staticmethod
    def _diversify(ranked: List[SearchResult]) -> List[SearchResult]:
        """Apply typed quotas after hybrid ranking so one knowledge kind cannot
        consume the complete prompt context."""
        selected: List[SearchResult] = []
        counts: Dict[str, int] = {}
        for item in ranked:
            kind = KIND_BY_DOC_TYPE.get(item.document.doc_type, "fact")
            if counts.get(kind, 0) >= KIND_QUOTAS[kind]:
                continue
            counts[kind] = counts.get(kind, 0) + 1
            selected.append(item)
            if len(selected) >= RETRIEVAL_LIMIT:
                break
        return selected


class RequirementUnderstandingAgent(Agent):
    name = "requirement_understanding"

    SYSTEM = """You are a senior QA requirement analyst. Decompose requirements without inventing behavior.
Identify actors, goals, explicit business rules, constraints, ambiguity, atomic testable requirements, and risks.
Extract API contracts, ticket states, events, correlation keys, and permissions when present.
Retrieved knowledge must be cited by evidence ID and must not be presented as an explicit PRD fact without a citation.
Every atomic requirement must preserve a source quote. Give atomic requirements local IDs (e.g. AR-001).
clarification_items.requirement_ids references these atomic IDs, never source document IDs.
Source document IDs belong in evidence_ids/retrieved_evidence_ids; ambiguity_ids references ambiguities[].id.
Return JSON only."""

    def run(self, payload: RequirementInput, context: Dict[str, Any]) -> RequirementAnalysis:
        if self.llm.enabled and not context.get("force_demo"):
            system = policy_system(self.SYSTEM, context)
            prompt = "Title: {}\nContext: {}\nRequirement:\n{}\nRetrieved knowledge:\n{}\nTeam memory:\n{}\nDynamic tool research:\n{}".format(
                    payload.title,
                    payload.context,
                    payload.content,
                    context.get("knowledge", ""),
                    context.get("memory", ""),
                    context.get("react_research", ""),
                )
            required_ambiguities = []
            for attempt in range(2):
                result = None
                analysis = None
                try:
                    result = self.llm.generate_json(system, prompt, RequirementAnalysis.model_json_schema())
                    analysis = RequirementAnalysis.model_validate(result)
                    if context.get("clarification_policy") == "evidence_only":
                        validate_analysis(analysis, payload.content, context.get("input_evidence"))
                        if not {a["id"] for a in required_ambiguities} <= {a["id"] for a in ambiguity_records(analysis)}:
                            raise LLMError("Correction dropped unresolved ambiguity IDs", code="invalid_scope")
                    return analysis
                except ValidationError as exc:
                    # Do not put provider data or Pydantic input values into feedback.
                    failure = LLMError("Requirement analysis does not match its schema", code="invalid_schema")
                    feedback = "; ".join("{}: {}".format(".".join(map(str, e["loc"])), e["type"])
                                         for e in exc.errors()[:8])
                except LLMError as exc:
                    if exc.code not in {"invalid_json", "invalid_scope"}:
                        raise  # Timeout, auth and transport failures are not format repairs.
                    failure = exc
                    feedback = ("Return exactly one JSON object, without trailing text or additional objects."
                                if exc.code == "invalid_json" else str(exc))
                if attempt:
                    raise failure
                if analysis is not None and context.get("clarification_policy") == "evidence_only":
                    required_ambiguities = ambiguity_records(analysis)
                prompt += ("\nValidation feedback (one correction attempt): {}. {}\n"
                           "Return a complete corrected analysis matching the schema. Keep the original input scope, "
                           "evidence and every unresolved ambiguity; classify gaps instead of deleting them to pass. "
                           "Do not invent business answers.\n").format(failure.code, feedback)
                if isinstance(result, dict):
                    prompt += "Previous analysis (untrusted task data):\n" + json.dumps(result, ensure_ascii=False)
                if analysis is not None and context.get("clarification_policy") == "evidence_only":
                    prompt += ("\nID namespaces: requirement_ids must reference atomic_requirements[].id; "
                               "evidence_ids/retrieved_evidence_ids must reference supplied document IDs. "
                               "These namespaces are distinct; do not put document IDs in requirement_ids. "
                               "If an atomic ID is itself a source document ID, assign a local atomic ID and remap its gaps.\n" +
                               json.dumps({"allowed_requirement_ids": [r.id for r in analysis.atomic_requirements
                                                                         if r.id not in context.get("input_evidence", {})],
                                           "allowed_evidence_ids": list(context.get("input_evidence", {}))}, ensure_ascii=False))
                    quotes = [("atomic_requirements", index, r.source_quote) for index, r in enumerate(analysis.atomic_requirements)]
                    quotes += [("clarification_items", index, g.source_quote) for index, g in enumerate(analysis.clarification_items)]
                    prompt += "\nExact supplied-content quote matches (cite the document in evidence_ids, not requirement_ids):\n" + json.dumps([
                        {"field": "{}[{}].source_quote".format(kind, index), "matching_evidence_ids": [
                            doc_id for doc_id, doc in context.get("input_evidence", {}).items() if quote.strip() and quote in doc["content"]]}
                        for kind, index, quote in quotes], ensure_ascii=False)
                if required_ambiguities:
                    prompt += "\nPreserve these ambiguity IDs and classify them via ambiguity_ids:\n" + json.dumps(required_ambiguities, ensure_ascii=False)
        analysis = self._demo(payload, context)
        if context.get("clarification_policy") == "evidence_only":
            # Demo heuristics (no digits / no failure words) are not business contradictions.
            analysis.ambiguities = []
        return analysis

    def _demo(self, payload: RequirementInput, context: Dict[str, Any]) -> RequirementAnalysis:
        sentences = sentence_split(payload.content)
        actors = []
        for actor in ["用户", "管理员", "访客", "商家", "审核员", "系统", "客服"]:
            if actor in payload.content:
                actors.append(actor)
        if not actors:
            actors = ["用户"]
        if any(token in payload.content.lower() for token in ["工单", "审核", "设备借用", "librarian", "equipment_borrow", "p2p", "风控"]):
            actors = list(dict.fromkeys(actors + ["业务系统", "审核员"]))

        rule_words = ["必须", "仅", "不能", "允许", "最多", "至少", "需要", "只有", "不得"]
        risk_words = ["金额", "权限", "状态", "并发", "重复", "超时", "库存", "支付", "删除", "审核"]
        rules = [sentence for sentence in sentences if any(word in sentence for word in rule_words)]
        atomic = []
        for index, sentence in enumerate(sentences[:16], 1):
            category = "business_rule" if any(word in sentence for word in rule_words) else "functional"
            atomic.append(
                AtomicRequirement(
                    id="REQ-{:03d}".format(index),
                    statement=sentence,
                    category=category,
                    actors=[actor for actor in actors if actor in sentence] or actors[:1],
                    source_quote=sentence,
                )
            )
        raw_knowledge = context.get("knowledge_context") or {"query": payload.content}
        knowledge_context = KnowledgeContext.model_validate(raw_knowledge)
        for hit in knowledge_context.hits:
            if len(atomic) >= 24:
                break
            if hit.doc_type == "case_example":
                continue
            atomic.append(
                AtomicRequirement(
                    id="REQ-{:03d}".format(len(atomic) + 1),
                    statement=hit.title + "：" + hit.content,
                    category="retrieved_" + next(
                        (kind for kind in ["event_contract", "api_contract", "workflow", "release_rule"] if kind in hit.title.lower() or kind in hit.section.lower()),
                        "domain_rule",
                    ),
                    actors=["系统"],
                    source_quote="[{} | 第{}页] {}".format(hit.document_id, hit.page or "?", hit.content),
                )
            )

        ambiguities = []
        if not any(char.isdigit() for char in payload.content):
            ambiguities.append("关键数量、时限或边界值未明确，需要产品确认。")
        if "失败" not in payload.content and "异常" not in payload.content:
            ambiguities.append("失败场景及恢复策略未说明。")
        if len(actors) == 1:
            ambiguities.append("角色与权限差异未完整说明。")

        risks = ["{}相关规则可能导致高影响缺陷".format(word) for word in risk_words if word in payload.content]
        if not risks:
            risks = ["核心流程中断后的数据一致性", "输入边界与重复提交"]
        evidence_ids = [hit.document_id for hit in knowledge_context.hits]
        ticket_types = [knowledge_context.ticket_type] if knowledge_context.domain == "ticket" else []
        if knowledge_context.domain == "ticket" and not knowledge_context.has_specific_knowledge:
            ambiguities.append(
                "当前未加载该工单类型的专属知识包，结果仅依据当前需求与通用工单规则。"
            )

        # Domain contracts come from the ticket type's knowledge pack, never from agent code.
        facts = get_domain_facts(knowledge_context.ticket_type) if knowledge_context.domain == "ticket" else None
        interfaces = []
        transitions = []
        events = []
        permissions: List[str] = []
        if facts:
            actors = list(dict.fromkeys(actors + facts.extra_actors))
            interfaces = [item.model_copy(deep=True) for item in facts.interfaces]
            transitions = [item.model_copy(deep=True) for item in facts.state_transitions]
            events = [item.model_copy(deep=True) for item in facts.events]
            permissions = list(facts.permissions)
        return RequirementAnalysis(
            summary=compact(payload.content, 240),
            actors=actors,
            goals=[item.statement for item in atomic[:4]],
            business_rules=rules,
            constraints=[sentence for sentence in sentences if any(x in sentence for x in ["秒", "分钟", "小时", "兼容", "性能", "灰度"])],
            ambiguities=ambiguities,
            atomic_requirements=atomic,
            risk_hints=risks,
            ticket_types=ticket_types,
            interfaces=interfaces,
            state_transitions=transitions,
            events=events,
            permissions=permissions,
            retrieved_evidence_ids=evidence_ids,
        )


class ModulePlanningAgent(Agent):
    name = "module_planning"

    SYSTEM = """You are a senior test architect operating an interactive module-tree editor.
Build modules from parsed atomic requirements, not from UI page names. Preserve requirement
traceability, interfaces, state flows, events, permissions, risks, and test strategies.
The requested operation mode is authoritative:
- full: create a complete module tree.
- regenerate: replace the existing tree with a new complete tree.
- continue: retain every existing module and add only missing concerns.
- targeted: modify only the requested target module or its descendants.
- chat: apply the user's conversational instruction to the existing tree.
Use conversation memory to resolve references such as 'the previous module' or 'split it'.
Never generate test cases. Never silently drop a requirement ID. Return JSON only."""

    def run(self, payload: Any, context: Dict[str, Any]) -> ModuleTree:
        if isinstance(payload, RequirementAnalysis):
            analysis = payload
            mode = "full"
            instruction = "Generate the complete test module tree."
            target_module_id = ""
            existing = None
        else:
            analysis = RequirementAnalysis.model_validate(payload["analysis"])
            mode = str(payload.get("mode", "full"))
            instruction = str(payload.get("instruction", "")).strip()
            target_module_id = str(payload.get("target_module_id", "")).strip()
            existing_payload = payload.get("existing_tree")
            existing = (
                ModuleTree.model_validate(existing_payload)
                if existing_payload else None
            )
        if mode not in {"full", "regenerate", "continue", "targeted", "chat"}:
            raise ValueError("Unsupported module generation mode: {}".format(mode))
        if mode in {"continue", "targeted", "chat"} and not existing:
            raise ValueError("{} mode requires an existing module tree".format(mode))
        if mode == "targeted" and not target_module_id:
            raise ValueError("targeted mode requires target_module_id")
        if target_module_id and existing and not self._find(existing.modules, target_module_id):
            raise ValueError("Target module not found: {}".format(target_module_id))

        if self.llm.enabled and not context.get("force_demo"):
            prompt = self._prompt(
                analysis, mode, instruction, target_module_id, existing, context
            )
            callback = context.get("stream_callback")
            if callback:
                result = self.llm.generate_json_stream(
                    policy_system(self.SYSTEM, context), prompt, ModuleTree.model_json_schema(), callback
                )
            else:
                result = self.llm.generate_json(
                    policy_system(self.SYSTEM, context), prompt, ModuleTree.model_json_schema()
                )
            candidate = ModuleTree.model_validate(result)
            candidate.confirmed = False
            tree = self._apply_mode(
                mode, existing, candidate, target_module_id, instruction
            )
            return self._ensure_coverage(tree, analysis)
        tree = self._demo_operation(
            analysis, mode, instruction, target_module_id, existing
        )
        return self._ensure_coverage(tree, analysis)

    @staticmethod
    def _prompt(
        analysis: RequirementAnalysis,
        mode: str,
        instruction: str,
        target_module_id: str,
        existing: Optional[ModuleTree],
        context: Dict[str, Any],
    ) -> str:
        return """Operation mode: {mode}
User instruction: {instruction}
Target module ID: {target}

Requirement analysis:
{analysis}

Existing module tree:
{existing}

Module conversation memory:
{history}

Retrieved knowledge and module examples:
{knowledge}

Team memory:
{memory}

Return the complete resulting ModuleTree. For continue mode, include the existing modules
unchanged plus additions. For targeted mode, include the complete tree but change only the
target subtree. Keep stable IDs for unchanged modules.""".format(
            mode=mode,
            instruction=instruction or "Follow the mode contract.",
            target=target_module_id or "None",
            analysis=analysis.model_dump_json(),
            existing=existing.model_dump_json() if existing else "None",
            history=context.get("module_conversation", "None"),
            knowledge=context.get("knowledge", "None"),
            memory=context.get("memory", "None"),
        )

    def _apply_mode(
        self,
        mode: str,
        existing: Optional[ModuleTree],
        candidate: ModuleTree,
        target_module_id: str,
        instruction: str,
    ) -> ModuleTree:
        if mode in {"full", "regenerate", "chat"} or not existing:
            candidate.confirmed = False
            return candidate
        if mode == "continue":
            current_ids = {item.id for item in self._walk(existing.modules)}
            additions = [
                item.model_copy(deep=True)
                for item in candidate.modules
                if item.id not in current_ids
            ]
            return ModuleTree(
                modules=[item.model_copy(deep=True) for item in existing.modules] + additions,
                coverage_notes=list(dict.fromkeys(
                    existing.coverage_notes
                    + candidate.coverage_notes
                    + ["继续生成：保留原模块并补充 {} 个模块。".format(len(additions))]
                )),
                confirmed=False,
            )
        replacement = self._find(candidate.modules, target_module_id)
        if not replacement:
            raise ValueError(
                "LLM result did not contain target module {}".format(target_module_id)
            )
        modules = self._replace(
            existing.modules, target_module_id, replacement.model_copy(deep=True)
        )
        return ModuleTree(
            modules=modules,
            coverage_notes=list(dict.fromkeys(
                existing.coverage_notes
                + ["指定模块 {} 已更新：{}".format(target_module_id, instruction)]
            )),
            confirmed=False,
        )

    def _demo_operation(
        self,
        analysis: RequirementAnalysis,
        mode: str,
        instruction: str,
        target_module_id: str,
        existing: Optional[ModuleTree],
    ) -> ModuleTree:
        if mode in {"full", "regenerate"} or not existing:
            return self._demo(analysis)
        tree = existing.model_copy(deep=True)
        tree.confirmed = False
        if mode == "continue":
            used = {item.id for item in self._walk(tree.modules)}
            index = 1
            while "MOD-C{:02d}".format(index) in used:
                index += 1
            tree.modules.append(TestModule(
                id="MOD-C{:02d}".format(index),
                name="补充测试关注点",
                objective=instruction or "补充现有模块树尚未覆盖的测试关注点",
                requirement_ids=[item.id for item in analysis.atomic_requirements[:3]],
                risks=analysis.risk_hints[:2],
                case_types=["functional", "exception"],
            ))
            tree.coverage_notes.append("本地回退继续生成已追加一个模块，供后续生成使用。")
            return tree
        target = self._find(tree.modules, target_module_id) if target_module_id else None
        if target:
            target.objective = instruction or target.objective
            tree.coverage_notes.append("本地回退已更新目标模块 {}。".format(target.id))
        else:
            tree.coverage_notes.append("对话请求已记录，需在 LLM 可用后应用：{}".format(instruction))
        return tree

    def _demo(self, analysis: RequirementAnalysis) -> ModuleTree:
        requirements = analysis.atomic_requirements

        def match_ids(keywords: List[str], allow_fallback: bool) -> List[str]:
            matched = [
                requirement.id
                for requirement in requirements
                if any(keyword.lower() in requirement.statement.lower() for keyword in keywords)
            ]
            if matched or not allow_fallback:
                return matched
            retrieved = [item.id for item in requirements if item.category.startswith("retrieved_")]
            return retrieved[:4] or [item.id for item in requirements[:4]]

        facts = get_domain_facts(analysis.ticket_types[0]) if analysis.ticket_types else None
        if facts and facts.module_specs:
            specs = facts.module_specs
            notes = [
                "模块树来自结构化接口、状态、事件和权限事实，可直接用于生成，支持人工调整。",
                "RAG 规则必须保留证据 ID 和 PDF 页码；歧义项不作为已确认规则。",
            ]
            allow_fallback = True
        elif analysis.ticket_types:
            specs = GENERIC_TICKET_MODULE_SPECS
            notes = [
                "当前使用通用工单模块模板；接入该类型 PRD 后可扩展专属接口、状态和事件模块。",
                "不同工单类型的知识检索相互隔离，未命中的规则保留为待确认项。",
            ]
            allow_fallback = True
        else:
            specs = [
                ModuleSpec(name="核心业务流程", keywords=["新增", "创建", "提交", "查询", "登录", "支付", "保存", "编辑"], objective="验证主要用户目标能够闭环完成"),
                ModuleSpec(name="业务规则与状态", keywords=["必须", "仅", "状态", "审核", "取消", "完成", "发布"], objective="验证业务限制和状态迁移正确"),
                ModuleSpec(name="输入与边界", keywords=["数量", "金额", "长度", "最多", "至少", "范围", "时间"], objective="验证输入域及临界值处理"),
                ModuleSpec(name="权限与安全", keywords=["权限", "角色", "管理员", "授权", "登录"], objective="验证不同身份的数据与操作隔离"),
                ModuleSpec(name="异常与恢复", keywords=["失败", "异常", "超时", "重试", "网络", "重复"], objective="验证依赖异常时的反馈和数据一致性"),
            ]
            notes = ["模块树规划后可直接生成用例，支持人工调整。"]
            allow_fallback = False

        risky_types = {"state_transition", "event_consistency", "exception"}
        modules = []
        for index, spec in enumerate(specs, 1):
            matched = match_ids(spec.keywords, allow_fallback)
            if not matched and not allow_fallback:
                continue
            risks = analysis.risk_hints[:3] if analysis.ticket_types else analysis.risk_hints[:2]
            if set(spec.case_types) & risky_types:
                risks = risks + ["跨系统状态不一致"]
            modules.append(TestModule(
                id="MOD-{:02d}".format(index),
                name=spec.name,
                objective=spec.objective,
                requirement_ids=matched,
                risks=risks,
                case_types=list(spec.case_types),
            ))

        assigned_ids = {
            requirement_id
            for module in modules
            for requirement_id in module.requirement_ids
        }
        unassigned_ids = [
            requirement.id for requirement in requirements
            if requirement.id not in assigned_ids
        ]
        if unassigned_ids:
            modules.append(TestModule(
                id="MOD-UN",
                name=UNASSIGNED_MODULE_NAME,
                objective="以下原子需求未匹配到任何测试模块，确认模块树时需人工归类或确认删除",
                requirement_ids=unassigned_ids,
                risks=["覆盖缺口"],
                case_types=["functional"],
            ))
            notes = notes + ["存在未归类原子需求，已单独列出，确认模块树时请人工归类。"]
        return ModuleTree(modules=modules, coverage_notes=notes, confirmed=False)

    @classmethod
    def _ensure_coverage(
        cls, tree: ModuleTree, analysis: RequirementAnalysis
    ) -> ModuleTree:
        required = [item.id for item in analysis.atomic_requirements]
        covered = {
            requirement_id
            for module in cls._walk(tree.modules)
            for requirement_id in module.requirement_ids
        }
        missing = [item for item in required if item not in covered]
        if not missing:
            return tree
        unassigned = cls._find(tree.modules, "MOD-UN")
        if unassigned:
            unassigned.requirement_ids = list(dict.fromkeys(
                unassigned.requirement_ids + missing
            ))
        else:
            tree.modules.append(TestModule(
                id="MOD-UN",
                name=UNASSIGNED_MODULE_NAME,
                objective="对话修改后未被其他模块承接的需求，必须由 QA 重新归类",
                requirement_ids=missing,
                risks=["覆盖缺口"],
                case_types=["functional"],
            ))
        tree.coverage_notes.append(
            "覆盖保护：{} 条需求在修改后进入未归类模块。".format(len(missing))
        )
        return tree

    @classmethod
    def _walk(cls, modules: List[TestModule]) -> List[TestModule]:
        result: List[TestModule] = []
        for module in modules:
            result.append(module)
            result.extend(cls._walk(module.children))
        return result

    @classmethod
    def _find(
        cls, modules: List[TestModule], module_id: str
    ) -> Optional[TestModule]:
        return next((item for item in cls._walk(modules) if item.id == module_id), None)

    @classmethod
    def _replace(
        cls,
        modules: List[TestModule],
        module_id: str,
        replacement: TestModule,
    ) -> List[TestModule]:
        result: List[TestModule] = []
        for module in modules:
            if module.id == module_id:
                result.append(replacement)
                continue
            copied = module.model_copy(deep=True)
            copied.children = cls._replace(copied.children, module_id, replacement)
            result.append(copied)
        return result

MODULE_FIXABLE_CATEGORIES = {"duplicate_id", "duplicate_name", "empty_module"}


class ModuleCriticAgent(Agent):
    name = "module_critic"

    SYSTEM = """You are an independent senior test architecture critic. Review a test module tree for
semantic overlap, missing test concerns, vague objectives, UI-page-oriented grouping, and modules that are too broad or too narrow.
Do not invent requirements. Return JSON: {"findings": [{"severity": "high|medium|low", "category": "semantic", "module_id": "...", "message": "..."}]}"""

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> ModuleReviewReport:
        analysis = RequirementAnalysis.model_validate(payload["analysis"])
        tree = ModuleTree.model_validate(payload["module_tree"])
        findings: List[ModuleReviewFinding] = []
        modules = ModulePlanningAgent._walk(tree.modules)
        ids = Counter(module.id for module in modules)
        names = Counter(module.name for module in modules)
        for module in modules:
            if ids[module.id] > 1:
                findings.append(ModuleReviewFinding(
                    severity="high", category="duplicate_id", module_id=module.id,
                    message="模块 ID 重复：{}".format(module.id), detail=module.id,
                ))
            if names[module.name] > 1:
                findings.append(ModuleReviewFinding(
                    severity="medium", category="duplicate_name", module_id=module.id,
                    message="模块名称重复：{}".format(module.name), detail=module.name,
                ))
            if not module.requirement_ids:
                findings.append(ModuleReviewFinding(
                    severity="high", category="empty_module", module_id=module.id,
                    message="模块没有关联任何原子需求", detail=module.id,
                ))
            if len(module.requirement_ids) > 10:
                findings.append(ModuleReviewFinding(
                    severity="medium", category="module_granularity", module_id=module.id,
                    message="模块关联 {} 条需求，粒度可能过宽".format(len(module.requirement_ids)),
                ))
            if analysis.ticket_types and not module.case_types:
                findings.append(ModuleReviewFinding(
                    severity="medium", category="missing_strategy", module_id=module.id,
                    message="工单模块未声明测试类型策略",
                ))
            if module.name == UNASSIGNED_MODULE_NAME and module.requirement_ids:
                findings.append(ModuleReviewFinding(
                    severity="medium", category="unassigned_requirement", module_id=module.id,
                    message="存在 {} 条未归类需求，需要 QA 确认归属".format(len(module.requirement_ids)),
                ))

        covered = {requirement_id for module in modules for requirement_id in module.requirement_ids}
        for requirement in analysis.atomic_requirements:
            if requirement.id not in covered:
                findings.append(ModuleReviewFinding(
                    severity="high", category="requirement_coverage",
                    message="原子需求 {} 未进入任何模块".format(requirement.id),
                    detail=requirement.id,
                ))
        if self.llm.enabled and not context.get("force_demo"):
            try:
                critique = self.llm.generate_json(
                    policy_system(self.SYSTEM, context),
                    "Original requirement:\n{}\nRequirement analysis:\n{}\nModule tree:\n{}".format(
                        context.get("raw_requirement", ""), analysis.model_dump_json(), tree.model_dump_json()
                    ),
                )
                for item in critique.get("findings", [])[:15]:
                    severity = item.get("severity", "medium")
                    findings.append(ModuleReviewFinding(
                        severity=severity if severity in {"high", "medium", "low"} else "medium",
                        category="semantic",
                        module_id=item.get("module_id"),
                        message=str(item.get("message", ""))[:300],
                    ))
            except LLMError:
                if getattr(self.llm, "strict_review_failures", False):
                    raise
        penalty = sum({"high": 15, "medium": 6}.get(item.severity, 2) for item in findings)
        return ModuleReviewReport(score=max(0, 100 - penalty), findings=findings, rounds=0)


class ModuleRevisionAgent(Agent):
    name = "module_revision"

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> ModuleTree:
        tree = ModuleTree.model_validate(payload["module_tree"])
        report = ModuleReviewReport.model_validate(payload["module_review"])
        if not any(item.category in MODULE_FIXABLE_CATEGORIES for item in report.findings):
            return tree

        merged: List[TestModule] = []
        by_name: Dict[str, TestModule] = {}
        used_ids = set()
        for source in tree.modules:
            if not source.requirement_ids:
                continue
            module = source.model_copy(deep=True)
            if module.name in by_name:
                target = by_name[module.name]
                target.requirement_ids = list(dict.fromkeys(target.requirement_ids + module.requirement_ids))
                target.risks = list(dict.fromkeys(target.risks + module.risks))
                target.case_types = list(dict.fromkeys(target.case_types + module.case_types))
                continue
            base_id = module.id
            suffix = 2
            while module.id in used_ids:
                module.id = "{}-{}".format(base_id, suffix)
                suffix += 1
            used_ids.add(module.id)
            by_name[module.name] = module
            merged.append(module)
        return ModuleTree(
            modules=merged,
            coverage_notes=tree.coverage_notes + ["模块 Critique 已自动处理重复或空模块。"],
            confirmed=False,
        )


class CaseGenerationAgent(Agent):
    name = "case_generation"

    SYSTEM = """You are a meticulous senior test case writer. Generate atomic, reviewable, executable cases.
Each step must pair one action with an observable expected result. Cite requirement IDs and source evidence.
Use retrieved knowledge as reference, never as a replacement for current requirements.
When adopted reference cases are provided, follow their granularity, wording style, and assertion style.
The requested operation mode is authoritative. Use case conversation memory to resolve
references such as "the previous case" or "that module". Return a JSON object with a cases array."""

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> List[TestCase]:
        analysis = RequirementAnalysis.model_validate(payload["analysis"])
        tree = ModuleTree.model_validate(payload["module_tree"])
        mode = str(payload.get("mode", "full"))
        instruction = str(payload.get("instruction", "")).strip()
        target_module_id = str(payload.get("target_module_id", "")).strip()
        existing = [
            TestCase.model_validate(item)
            for item in payload.get("existing_cases", [])
        ]
        if not tree.modules:
            raise ValueError("A nonempty module tree is required before case generation")
        if mode not in {"full", "regenerate", "continue", "targeted", "chat"}:
            raise ValueError("Unsupported case generation mode: {}".format(mode))
        if mode in {"continue", "targeted", "chat"} and not existing:
            raise ValueError("{} mode requires existing test cases".format(mode))
        if mode == "targeted" and not target_module_id:
            raise ValueError("targeted mode requires target_module_id")
        if self.llm.enabled and not context.get("force_demo"):
            case_schema = {
                "type": "object",
                "properties": {"cases": {"type": "array", "items": TestCase.model_json_schema()}},
                "required": ["cases"],
            }
            selected = resolve_skills(json.dumps(payload, ensure_ascii=False), context.get("selected_skills"))
            prompt = """Operation mode: {mode}
User instruction: {instruction}
Target module ID: {target}

Input:
{payload}

Case conversation memory:
{history}

Testing skills:
{skills}

Adopted reference cases (few-shot):
{few_shot}

Knowledge:
{knowledge}

Long-term memory:
{memory}

Dynamic tool research:
{research}

Mode contract:
- full/regenerate: return the complete new case set.
- continue: return only genuinely new cases; do not repeat existing cases.
- targeted: return the complete replacement cases for the target module subtree only.
- chat: return the complete resulting case set after applying the conversational instruction.
Keep stable IDs for unchanged cases and never invent a module ID.""".format(
                    mode=mode,
                    instruction=instruction or "Follow the mode contract.",
                    target=target_module_id or "None",
                    payload=json.dumps(payload, ensure_ascii=False),
                    history=context.get("case_conversation", "None"),
                    skills="\n".join(skill.instruction for skill in selected),
                    few_shot=self._few_shot(context),
                    knowledge=context.get("knowledge", ""),
                    memory=context.get("memory", ""),
                    research=context.get("react_research", ""),
                )
            callback = context.get("stream_callback")
            if callback:
                result = self.llm.generate_json_stream(
                    policy_system(self.SYSTEM, context), prompt, case_schema, callback
                )
            else:
                result = self.llm.generate_json(
                    policy_system(self.SYSTEM, context),
                    prompt,
                    case_schema,
                )
            return [TestCase.model_validate(item) for item in result.get("cases", [])]
        return self._demo_operation(
            analysis, tree, mode, instruction, target_module_id, existing, context.get("selected_skills")
        )

    def _demo_operation(
        self,
        analysis: RequirementAnalysis,
        tree: ModuleTree,
        mode: str,
        instruction: str,
        target_module_id: str,
        existing: List[TestCase],
        selected_skills=None,
    ) -> List[TestCase]:
        if mode in {"full", "regenerate", "targeted"}:
            return self._demo(analysis, tree, selected_skills)
        if mode == "continue":
            req_map = {item.id: item for item in analysis.atomic_requirements}
            existing_types = {
                (item.module_id, item.case_type) for item in existing
            }
            preferred = [
                "boundary",
                "exception",
                "permission",
                "state_transition",
                "event_consistency",
                "compatibility",
            ]
            if selected_skills is not None:
                preferred = list(dict.fromkeys(
                    kind for skill in resolve_skills("", selected_skills) for kind in skill.case_types
                ))
            for module in ModulePlanningAgent._walk(tree.modules):
                for case_type in preferred:
                    if (module.id, case_type) in existing_types:
                        continue
                    evidence = [
                        req_map[rid].source_quote
                        for rid in module.requirement_ids
                        if rid in req_map
                    ][:3] or [analysis.summary]
                    case = self.build_case(
                        case_id=self._next_case_id(existing),
                        module=module,
                        case_type=case_type,
                        source=evidence[0],
                        evidence=evidence,
                        analysis=analysis,
                        first_module_id=tree.modules[0].id,
                        generated_by="conversation",
                    )
                    if instruction:
                        case.risk_tags = list(dict.fromkeys(
                            case.risk_tags + [instruction[:80]]
                        ))
                    return [case]
            return []
        cases = [item.model_copy(deep=True) for item in existing]
        candidates = [
            item for item in cases
            if not target_module_id or item.module_id == target_module_id
        ]
        if candidates and instruction:
            target = candidates[0]
            target.title = "{}：{}".format(instruction[:80], target.title)
            target.risk_tags = list(dict.fromkeys(
                target.risk_tags + ["conversation_adjusted"]
            ))
            target.generated_by = "conversation"
            target.review_status = "pending"
            target.human_status = "pending"
        return cases

    @staticmethod
    def _next_case_id(existing: List[TestCase]) -> str:
        values = []
        for item in existing:
            match = re.search(r"(\d+)$", item.id)
            if match:
                values.append(int(match.group(1)))
        return "TC-{:04d}".format(max(values or [0]) + 1)

    @staticmethod
    def _few_shot(context: Dict[str, Any], limit: int = 3) -> str:
        """Adopted historical cases (kind=example) retrieved via RAG become few-shot samples."""
        knowledge_context = context.get("knowledge_context") or {}
        examples = [
            hit for hit in knowledge_context.get("hits", [])
            if hit.get("doc_type") == "case_example"
        ][:limit]
        if not examples:
            return "None yet."
        return "\n\n".join(
            "[{}] {}\n{}".format(hit.get("document_id"), hit.get("title"), hit.get("content"))
            for hit in examples
        )

    def _demo(self, analysis: RequirementAnalysis, tree: ModuleTree, selected_skills=None) -> List[TestCase]:
        req_map = {req.id: req for req in analysis.atomic_requirements}
        cases = []
        case_index = 1
        for module in ModulePlanningAgent._walk(tree.modules):
            evidence = [req_map[rid].source_quote for rid in module.requirement_ids if rid in req_map]
            source = evidence[0] if evidence else analysis.summary
            evidence = evidence[:3] or [analysis.summary]
            skills = resolve_skills(module.name + " " + " ".join(evidence), selected_skills)
            selected_types = self._case_types(module, skills)
            for case_type in selected_types:
                cases.append(self.build_case(
                    case_id="TC-{:04d}".format(case_index),
                    module=module,
                    case_type=case_type,
                    source=source,
                    evidence=evidence,
                    analysis=analysis,
                    first_module_id=tree.modules[0].id,
                ))
                case_index += 1
        return cases

    def build_case(
        self,
        case_id: str,
        module: TestModule,
        case_type: str,
        source: str,
        evidence: List[str],
        analysis: RequirementAnalysis,
        first_module_id: str,
        generated_by: str = "generation",
    ) -> TestCase:
        title, steps, data = self._template(module, case_type, source, analysis)
        return TestCase(
            id=case_id,
            module_id=module.id,
            title=title,
            priority="P0" if case_type == "functional" and module.id == first_module_id else "P1",
            case_type=case_type,
            preconditions=["测试账号及基础数据可用", "系统处于可执行该流程的初始状态"],
            steps=steps,
            test_data=data,
            risk_tags=module.risks[:2],
            requirement_ids=module.requirement_ids,
            source_evidence=evidence,
            automation_feasibility="high" if case_type in ["functional", "boundary"] else "medium",
            generated_by=generated_by,
        )

    def _case_types(self, module: TestModule, skills: List[TestSkill]) -> List[str]:
        if module.case_types:
            return list(dict.fromkeys(module.case_types))
        rules = [
            ("创建", ["api_contract", "exception"]),
            ("详情", ["realtime_data", "exception"]),
            ("实时", ["realtime_data", "exception"]),
            ("Check", ["review_operation", "permission"]),
            ("审核操作", ["review_operation"]),
            ("状态流转", ["state_transition"]),
            ("状态机", ["state_transition"]),
            ("角色", ["permission"]),
            ("权限", ["permission"]),
            ("回调", ["event_consistency"]),
            ("通知", ["event_consistency"]),
            ("Kafka", ["event_consistency"]),
            ("异常", ["exception"]),
            ("灰度", ["compatibility"]),
        ]
        domain_types = []
        for keyword, case_types in rules:
            if keyword in module.name:
                domain_types.extend(case_types)
        if domain_types:
            return list(dict.fromkeys(domain_types))
        types = []
        for skill in skills:
            types.extend(skill.case_types)
        if "边界" in module.name:
            types.append("boundary")
        return list(dict.fromkeys(types))[:3] or ["functional"]

    @staticmethod
    def _find_interface(analysis: RequirementAnalysis, keywords: List[str]):
        for item in analysis.interfaces:
            text = (item.name + " " + item.purpose).lower()
            if any(keyword.lower() in text for keyword in keywords):
                return item
        return None

    def _template(
        self,
        module: TestModule,
        case_type: str,
        source: str,
        analysis: RequirementAnalysis,
    ) -> Tuple[str, List[TestStep], Dict[str, Any]]:
        """Templates are parameterized by the structured facts inside the analysis
        (interfaces, states, events, permissions). Any ticket type whose knowledge
        pack provides those facts gets the same template quality — no per-domain code."""
        ticket_type = analysis.ticket_types[0] if analysis.ticket_types else "general"
        if case_type == "api_contract":
            creator = self._find_interface(analysis, ["创建", "create"]) or (
                analysis.interfaces[0] if analysis.interfaces else None
            )
            if creator:
                return (
                    "{}：{} 契约、路由与幂等正确".format(module.name, creator.name),
                    [
                        TestStep(
                            action="调用 {}，携带全部必填字段：{}".format(creator.name, "、".join(creator.required_fields) or "按契约声明"),
                            expected=creator.success_condition or "接口成功返回且数据正确落库",
                        ),
                        TestStep(
                            action="使用相同唯一请求标识重复请求，并分别缺失关键字段后重试",
                            expected=creator.failure_condition or "重复请求不产生重复工单；非法请求明确失败且不留下半成品数据",
                        ),
                    ],
                    {"interface": creator.name, "required_fields": creator.required_fields, "idempotency_probe": ["request-1", "request-1"]},
                )
            return (
                "{}：类型路由、必填字段与幂等正确".format(module.name),
                [
                    TestStep(action="使用当前需求声明的工单类型、来源、业务主键和唯一请求标识创建工单", expected="仅创建一张目标类型工单，返回唯一工单 ID、初始状态和正确路由"),
                    TestStep(action="重复提交相同请求标识，并分别缺失类型、业务主键和必要业务字段", expected="重复请求不产生重复工单；非法请求明确失败且不留下半成品数据"),
                ],
                {"ticket_type": ticket_type, "request_id": ["request-1", "request-1"]},
            )
        if case_type == "realtime_data":
            detail = self._find_interface(analysis, ["详情", "实时", "detail", "query", "info"])
            if detail:
                return (
                    "{}：重复打开详情始终读取最新数据".format(module.name),
                    [
                        TestStep(
                            action="打开工单详情，触发 {}（{}）".format(detail.name, detail.purpose),
                            expected=detail.success_condition or "只展示该工单类型允许的最新业务信息",
                        ),
                        TestStep(
                            action="在业务侧更新数据后再次打开详情，并模拟查询超时或失败",
                            expected=detail.failure_condition or "展示更新后的数据，不复用旧缓存；失败时给出明确反馈",
                        ),
                    ],
                    {"interface": detail.name, "checks": ["freshness", "field_scope", "failure_feedback"]},
                )
            return (
                "{}：业务详情保持最新且字段隔离".format(module.name),
                [
                    TestStep(action="打开目标类型工单详情并记录业务信息、材料和当前状态", expected="只展示该工单类型允许的字段，数据与业务主键正确关联"),
                    TestStep(action="更新上游业务信息后重新打开详情，并模拟查询失败", expected="按需求刷新最新数据；失败时明确反馈且不把过期数据伪装为最新结果"),
                ],
                {"ticket_type": ticket_type, "checks": ["freshness", "field_scope", "failure_feedback"]},
            )
        if case_type == "review_operation":
            operations = [
                item for item in analysis.interfaces
                if any(keyword in (item.name + " " + item.purpose).lower() for keyword in ["check", "edit", "审核", "verify", "处置"])
            ]
            if operations:
                op_names = "、".join(item.name for item in operations)
                success_text = "；".join(item.success_condition for item in operations if item.success_condition)
                failure_text = "；".join(item.failure_condition for item in operations if item.failure_condition)
                guard = analysis.permissions[0] if analysis.permissions else "只有当前处理人可执行操作"
                return (
                    "{}：{} 遵守操作边界".format(module.name, op_names),
                    [
                        TestStep(
                            action="由当前审核人员依次执行 {}，并分别模拟成功与失败响应".format(op_names),
                            expected=success_text or "操作成功且反馈明确；失败时展示错误信息，业务状态不被意外改变",
                        ),
                        TestStep(
                            action="尝试越过操作边界：篡改受保护字段、跳过前置状态或由非当前处理人执行（约束：{}）".format(guard),
                            expected=failure_text or "服务端拒绝非法操作，状态和业务数据保持不变",
                        ),
                    ],
                    {"operations": [item.name for item in operations], "permission_rules": analysis.permissions[:3]},
                )
            return (
                "{}：审核操作遵守角色、字段和状态边界".format(module.name),
                [
                    TestStep(action="由当前需求允许的审核角色在合法前置状态执行通过、拒绝或补充材料操作", expected="操作成功并进入需求声明的下一状态，审计记录包含操作者、时间和理由"),
                    TestStep(action="使用错误角色、缺失必要理由或篡改只读业务字段执行同一操作", expected="服务端拒绝非法操作，状态和业务数据保持不变"),
                ],
                {"ticket_type": ticket_type, "operations": ["approve", "reject", "request_more_info"]},
            )
        if case_type == "event_consistency":
            if analysis.events:
                event = analysis.events[0]
                terminal_states = [status for status in event.statuses if status in ["PASS", "REFUSE", "SUCCESS", "FAIL"]] or event.statuses[-2:]
                return (
                    "{}：终态事件可关联且重复乱序安全".format(module.name),
                    [
                        TestStep(
                            action="分别完成 {} 终态并消费 {}".format("、".join(terminal_states), event.topic),
                            expected="仅 event={} 被处理，并通过 {} 关联正确工单，状态映射符合需求".format(event.event, event.correlation_key or "工单业务主键"),
                        ),
                        TestStep(
                            action="重复投递、先终态后中间态投递，并模拟消费失败重试（风险：{}）".format("、".join(event.delivery_risks) or "重复、乱序、重试"),
                            expected="消费幂等，终态不回退，重试后最终一致且可通过 trace 追踪",
                        ),
                    ],
                    {"topic": event.topic, "event": event.event, "statuses": event.statuses, "correlation_key": event.correlation_key},
                )
            return (
                "{}：业务结果可关联且重复乱序安全".format(module.name),
                [
                    TestStep(action="触发需求声明的审核结果回调或通知", expected="消息通过工单 ID 或业务主键关联正确工单，状态映射符合当前需求"),
                    TestStep(action="重复、乱序和延迟投递同一结果，并模拟消费失败后重试", expected="处理幂等，终态不回退，重试后工单与业务侧最终一致"),
                ],
                {"ticket_type": ticket_type, "delivery": ["duplicate", "out_of_order", "retry"]},
            )
        if case_type == "compatibility":
            return (
                "{}：灰度期间新老链路兼容且可回滚".format(module.name),
                [
                    TestStep(action="按灰度比例同时创建和审核新老版本工单", expected="业务无感知，服务可用，数据和通知状态映射一致"),
                    TestStep(action="调整发布顺序并执行回滚", expected="历史工单与历史客户端仍可用，监控可发现异常且回滚不丢数据"),
                ],
                {"coexistence": ["old", "new"], "checks": ["availability", "data", "event", "rollback"]},
            )
        if case_type == "boundary":
            return (
                "{}：临界值输入处理正确".format(module.name),
                [
                    TestStep(action="进入{}并准备有效基础数据".format(module.name), expected="页面或接口处于可操作状态"),
                    TestStep(action="分别输入边界内、边界值和越界值后提交", expected="合法值成功；越界值被明确拒绝且不产生脏数据"),
                ],
                {"values": ["min-1", "min", "max", "max+1"]},
            )
        if case_type == "exception":
            return (
                "{}：依赖异常时可恢复且数据一致".format(module.name),
                [
                    TestStep(action="制造网络超时或下游失败后执行操作", expected="系统给出可理解的失败反馈"),
                    TestStep(action="恢复依赖并重试同一操作", expected="操作可完成且不会重复写入或产生中间状态"),
                ],
                {"fault": "timeout", "retry": 1},
            )
        if case_type == "permission":
            rules_text = "；".join(analysis.permissions[:3]) if analysis.permissions else ""
            return (
                "{}：无权限角色无法越权操作".format(module.name),
                [
                    TestStep(
                        action="使用无权限账号访问目标功能" + ("（权限规则：{}）".format(rules_text) if rules_text else ""),
                        expected="入口不可见或请求被拒绝",
                    ),
                    TestStep(action="直接构造目标请求绕过前端", expected="服务端拒绝请求且无数据变化，审计记录保留操作痕迹"),
                ],
                {"role": "unauthorized_user", "permission_rules": analysis.permissions[:3]},
            )
        if case_type == "state_transition":
            if analysis.state_transitions:
                happy = [item for item in analysis.state_transitions if not any(word in item.action for word in ["拒绝", "驳回", "reject"])]
                terminal_states = list(dict.fromkeys(item.to_state for item in analysis.state_transitions if item.terminal))
                path = " → ".join([happy[0].from_state] + [item.to_state for item in happy]) if happy else "按需求声明的状态顺序"
                return (
                    "{}：合法状态迁移并阻止非法或重复操作".format(module.name),
                    [
                        TestStep(
                            action="按声明路径执行完整审核：{}".format(path),
                            expected="每一步只进入允许的下一状态，最终进入终态（{}）并同步业务方".format("、".join(terminal_states) or "需求声明终态"),
                        ),
                        TestStep(
                            action="从错误角色、错误前置状态、重复及并发执行通过或拒绝",
                            expected="非法迁移被拒绝；合法拒绝进入声明终态；终态不可回退",
                        ),
                    ],
                    {"declared_transitions": ["{} --{}--> {}".format(item.from_state, item.action, item.to_state) for item in analysis.state_transitions], "terminal_states": terminal_states},
                )
            return (
                "{}：合法状态迁移并阻止非法或重复操作".format(module.name),
                [
                    TestStep(action="按当前需求声明的角色和状态顺序执行一条完整通过或关闭路径", expected="每一步只进入允许的下一状态，终态与业务处理结果一致"),
                    TestStep(action="从错误角色、错误前置状态、重复及并发执行通过或拒绝", expected="非法迁移被拒绝；合法拒绝进入声明终态；终态不可回退"),
                ],
                {"ticket_type": ticket_type, "declared_transitions": ["pending-confirmation"]},
            )
        return (
            "{}：主流程成功完成".format(module.name),
            [
                TestStep(action="按需求准备有效数据并进入目标功能", expected="前置数据正确展示且操作可用"),
                TestStep(action="按照主流程完成操作并提交", expected="操作成功，结果、状态和持久化数据与需求一致"),
            ],
            {"source_rule": compact(source, 80)},
        )


class CaseReviewAgent(Agent):
    name = "case_review"

    CRITIQUE_SYSTEM = """You are an independent senior test case critic. Review generated cases semantically:
unexecutable steps, unobservable expected results, wrong assertions versus the requirement, and duplicated cases.
Do not repeat structural issues (missing coverage counts are checked elsewhere).
Use the raw requirement, full analysis and user constraints as the acceptance basis.
Do not invent error codes, message keywords, retry semantics, caches or implementation mechanisms.
Separate defect (a demonstrated contradiction or execution failure), clarification (missing business
contract or evidence needed to decide), and suggestion (optional cleanup, duplication or readability).
Do not replace an undefined contract with invented proxy assertions. A suggestion is non-blocking;
never disguise a correctness defect as a suggestion. High severity always blocks delivery.
For each finding quote exact evidence from the supplied requirement or affected case, and identify
applicable atomic requirement IDs. Evidence supports the issue, not an invented replacement contract.
Review previous findings first: do not reintroduce an unsupported constraint that was removed.
Reopening or changing a previous recommendation requires concrete current evidence and explanation.
History and planner focus are reference data, not permission to waive genuine failures.
Output exactly one JSON object with a findings array and then stop. Do not output Markdown fences,
backticks, comments, summaries, or explanations before or after the JSON. Put every actionable
observation inside findings, with explanation inside message. Copy evidence verbatim from a single
input text field; do not add step labels, paraphrase it, or combine excerpts with ellipses.
Return JSON: {"findings": [{"severity": "high|medium|low", "case_id": "...",
"disposition": "defect|clarification|suggestion",
"issue_type": "state_setup|assertion_semantics|observability|data_isolation|requirement_conflict|redundancy|missing_contract|other",
"requirement_ids": [], "evidence": "exact source quote", "message": "issue and grounded action"}]}"""

    CRITIQUE_SCHEMA = {
        "type": "object", "additionalProperties": False, "required": ["findings"],
        "properties": {"findings": {"type": "array", "maxItems": 20, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["severity", "case_id", "disposition", "issue_type", "requirement_ids", "evidence", "message"],
            "properties": {
                "severity": {"type": "string", "enum": ["high", "critical", "error", "medium", "low"]},
                "case_id": {"type": "string"},
                "disposition": {"type": "string", "enum": ["defect", "clarification", "suggestion"]},
                "issue_type": {"type": "string", "enum": sorted(ISSUE_TYPES)},
                "requirement_ids": {"type": "array", "items": {"type": "string"}},
                "related_case_ids": {"type": "array", "items": {"type": "string"}},
                "evidence": {"type": "string", "minLength": 1, "maxLength": 800},
                "message": {"type": "string", "minLength": 1, "maxLength": 1500},
                "clarification_kind": {"type": "string", "enum": ["unspecified", "execution_detail", "out_of_scope", "behavior_blocker"]},
                "clarification_reason": {"type": "string", "maxLength": 1500},
            },
        }}},
    }

    def _generate_critique(self, user_prompt, context=None):
        # A malformed finding is a review protocol failure, not evidence that
        # the user owes us a new business contract. Retry the same complete
        # request once; never accept partial findings from the failed attempt.
        request = json.loads(user_prompt)
        feedback = ""
        for attempt in range(2):
            system = policy_system(self.CRITIQUE_SYSTEM, context or {})
            if attempt:
                system += ("\n" + feedback + " Regenerate the complete review from the same input. "
                           "Return one valid JSON object with all findings inside the array. "
                           "Quote exact text values, without JSON field names or added explanations. "
                           "Keep supported issues and their severity; do not invent business rules, "
                           "discard issues to pass validation, or turn formatting errors into business clarification.")
            try:
                result = self.llm.generate_json(system, user_prompt, self.CRITIQUE_SCHEMA)
                self._validate_critique_response(result, request)
                return result
            except LLMError as exc:
                if attempt or exc.code not in {"invalid_json", "invalid_schema"}:
                    raise
                feedback = ("The previous response failed JSON parsing." if exc.code == "invalid_json"
                            else "The previous response failed review validation: " + str(exc))

    @staticmethod
    def _validate_critique_response(result, request):
        def invalid(field):
            # Field paths and fixed explanations only, no provider content.
            raise LLMError("Review field " + field + " is invalid", code="invalid_schema")

        if not isinstance(result, dict) or not isinstance(result.get("findings"), list):
            invalid("findings (array required)")
        if len(result["findings"]) > 20:
            invalid("findings (maximum 20 per response)")
        screening = request.get("review_phase") == "cross_batch_screen"
        case_map = {c["id"]: c for c in request.get("case_cards" if screening else "cases", [])}
        known_requirements = {r["id"] for r in request.get("analysis", {}).get("atomic_requirements", [])}
        source = {key: request.get(key) for key in
                  ("raw_requirement", "input_evidence", "project_context", "analysis", "user_constraints")}
        for index, item in enumerate(result["findings"]):
            prefix = "findings[{}].".format(index)
            if not isinstance(item, dict):
                invalid(prefix + "object")
            if not isinstance(item.get("case_id"), str) or item["case_id"] not in case_map:
                invalid(prefix + "case_id (must belong to this batch)")
            if not isinstance(item.get("message"), str) or not item["message"].strip():
                invalid(prefix + "message")
            for field, allowed in [("severity", {"low", "medium", "high", "critical", "error"}),
                                   ("disposition", {"defect", "clarification", "suggestion"}),
                                   ("issue_type", ISSUE_TYPES)]:
                if not isinstance(item.get(field), str) or item[field] not in allowed:
                    invalid(prefix + field)
            ids = item.get("requirement_ids")
            if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids) or not set(ids) <= known_requirements:
                invalid(prefix + "requirement_ids (unknown requirement)")
            quote = item.get("evidence")
            if (not isinstance(quote, str) or not quote.strip() or
                    (not screening and not contains_evidence([source, case_map[item["case_id"]]], quote))):
                invalid(prefix + "evidence (exact supplied text value required)")
            if screening:
                related = item.get("related_case_ids")
                if (not isinstance(related, list) or any(not isinstance(i, str) for i in related)
                        or not 2 <= len(set(related)) <= 12 or item["case_id"] not in related
                        or not set(related) <= set(case_map)):
                    invalid(prefix + "related_case_ids")

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = RequirementAnalysis.model_validate(payload["analysis"])
        tree = ModuleTree.model_validate(payload["module_tree"])
        cases = [TestCase.model_validate(item) for item in payload["cases"]]
        findings: List[ReviewFinding] = []

        coverage = Counter(case.case_type for case in cases)
        requirement_coverage = Counter(
            requirement_id for case in cases for requirement_id in case.requirement_ids
        )
        for requirement in analysis.atomic_requirements:
            if not requirement_coverage[requirement.id]:
                findings.append(
                    ReviewFinding(
                        severity="high",
                        category="requirement_coverage",
                        message="原子需求 {} 尚无用例覆盖".format(requirement.id),
                        detail=requirement.id,
                    )
                )

        module_counts = Counter(case.module_id for case in cases)
        # Grouping modules inherit coverage from their descendants. A case on
        # the parent never substitutes for an empty child branch.
        def covered_count(module):
            count = module_counts[module.id] + sum(covered_count(child) for child in module.children)
            module_counts[module.id] = count
            return count

        for root in tree.modules:
            covered_count(root)
        for module in ModulePlanningAgent._walk(tree.modules):
            if not module_counts[module.id]:
                findings.append(
                    ReviewFinding(
                        severity="high",
                        category="module_coverage",
                        module_id=module.id,
                        message="模块没有任何测试用例",
                        detail=module.id,
                    )
                )
            if module.name == UNASSIGNED_MODULE_NAME and module.requirement_ids:
                findings.append(
                    ReviewFinding(
                        severity="medium",
                        category="coverage_gap",
                        module_id=module.id,
                        message="存在 {} 条未归类原子需求，模块规划未能显式覆盖，需 QA 人工归类".format(len(module.requirement_ids)),
                    )
                )

        for case in cases:
            if not case.source_evidence:
                findings.append(ReviewFinding(severity="medium", category="traceability", case_id=case.id, message="缺少需求证据"))
            if any(not step.expected.strip() for step in case.steps):
                findings.append(ReviewFinding(severity="high", category="assertion", case_id=case.id, message="存在无预期结果的操作步骤", detail=case.id))

        facts = get_domain_facts(analysis.ticket_types[0]) if analysis.ticket_types else None
        if facts and facts.required_case_types:
            important_types = facts.required_case_types
        elif analysis.ticket_types:
            important_types = GENERIC_TICKET_REQUIRED_CASE_TYPES
        else:
            important_types = ["functional", "boundary", "exception"]
        if context.get("clarification_policy") == "evidence_only":
            # Actual atomic requirements still require coverage; generic categories
            # cannot force new behavior into a short requirement.
            important_types = []
            for case in cases:
                if not case.requirement_ids or not set(case.requirement_ids) <= {r.id for r in analysis.atomic_requirements}:
                    findings.append(ReviewFinding(severity="high", category="requirement_coverage", case_id=case.id,
                                                  message="行为级用例必须关联有效的原文需求", detail=case.id))
        for important_type in important_types:
            if not coverage[important_type]:
                findings.append(
                    ReviewFinding(
                        severity="high" if analysis.ticket_types else "medium",
                        category="case_type",
                        message="缺少 {} 类型用例".format(important_type),
                        detail=important_type,
                    )
                )

        # Domain evidence traceability: when a knowledge pack exists, every case
        # should trace back to at least one retrieved PRD evidence chunk.
        if facts:
            for case in cases:
                if not any(
                    evidence_id in evidence
                    for evidence in case.source_evidence
                    for evidence_id in analysis.retrieved_evidence_ids
                ):
                    findings.append(
                        ReviewFinding(
                            severity="medium",
                            category="rag_traceability",
                            case_id=case.id,
                            message="工单领域用例缺少 PRD 知识证据 ID 或页码",
                        )
                    )

        # Semantic critique on top of structural checks (Generator-Critic pattern).
        if self.llm.enabled and not context.get("force_demo"):
            try:
                from .semantic_review import review_cases
                critique = review_cases(lambda prompt: self._generate_critique(prompt, context), {
                        "raw_requirement": payload.get("requirement", ""),
                        "input_evidence": context.get("input_evidence", {}),
                        "project_context": payload.get("project_context", ""),
                        "analysis": analysis.model_dump(),
                        "user_constraints": payload.get("review_constraints", ""),
                        "review_focus": payload.get("review_focus", ""),
                        "previous_reviews": payload.get("review_history", []),
                        "issue_ledger": payload.get("issue_ledger", {}),
                    }, [case.model_dump() for case in cases])
                if not isinstance(critique, dict) or not isinstance(critique.get("findings"), list):
                    raise LLMError("Critique requires an explicit findings array")
                semantic = []
                case_map = {case.id: case for case in cases}
                known_requirements = {req.id for req in analysis.atomic_requirements}
                source = {"requirement": payload.get("requirement", ""),
                          "input_evidence": context.get("input_evidence", {}),
                          "project_context": payload.get("project_context", ""),
                          "analysis": analysis.model_dump(),
                          "constraints": payload.get("review_constraints", "")}
                for item in critique["findings"]:
                    if not isinstance(item, dict) or not isinstance(item.get("message"), str) or not item["message"].strip():
                        raise LLMError("Invalid critique finding")
                    if not isinstance(item.get("case_id"), str) or item["case_id"] not in case_map:
                        raise LLMError("Critique references an unknown case")
                    severity = item.get("severity", "medium")
                    disposition = item.get("disposition", "clarification")
                    if any(not isinstance(value, str) for value in [severity, disposition, item.get("issue_type", "other")]):
                        raise LLMError("Invalid critique classification")
                    evidence = item.get("evidence", "")
                    req_ids = item.get("requirement_ids", [])
                    if not isinstance(evidence, str) or not isinstance(req_ids, list) or any(not isinstance(i, str) for i in req_ids):
                        raise LLMError("Invalid critique evidence")
                    grounded = bool(evidence.strip()) and contains_evidence([source, case_map[item["case_id"]].model_dump()], evidence)
                    if (disposition not in {"defect", "clarification", "suggestion"}
                            or not grounded or not set(req_ids) <= known_requirements):
                        disposition = "clarification"
                    if severity in {"high", "critical", "error"} and disposition == "suggestion":
                        disposition = "defect" if grounded else "clarification"
                    finding = ReviewFinding(
                            severity=severity if severity in {"high", "critical", "error", "medium", "low"} else "medium",
                            category="semantic",
                            case_id=item.get("case_id"),
                            message=item["message"][:1500], disposition=disposition,
                            evidence=evidence[:800], requirement_ids=req_ids,
                            issue_type=item.get("issue_type") if item.get("issue_type") in ISSUE_TYPES else "other",
                        )
                    kind = item.get("clarification_kind", "unspecified")
                    reason = item.get("clarification_reason", "")
                    if isinstance(kind, str) and kind in {"execution_detail", "out_of_scope", "behavior_blocker"} and isinstance(reason, str):
                        finding.clarification_kind = kind
                        finding.clarification_reason = reason[:1500]
                        finding.clarification_basis_verified = bool(
                            grounded and reason.strip() and req_ids and set(req_ids) <= known_requirements
                            and item.get("disposition") == "clarification"
                            and disposition == "clarification")
                    finding.issue_id = issue_key(finding)
                    semantic.append(finding)
                findings.extend(semantic)
            except LLMError as exc:
                # Persist a safe category, never provider response bodies or credentials.
                reasons = {
                    "output_limit": "模型输出达到长度限制（finish_reason=length），未返回完整评审；请调整思考强度或输出上限后重试",
                    "timeout": "模型读取超时，请检查连接或请求超时配置后重试",
                    "invalid_json": "模型响应不是有效 JSON，请检查响应格式后重试",
                    "invalid_schema": "评审引文或分类未通过校验，反馈修正后仍不符合约定；这是评审技术失败，无需补充业务契约",
                    "incomplete_response": "模型响应中断或未正常完成，请检查服务状态后重试",
                    "request_failed": "模型请求失败或评审响应不符合约定，请检查调用诊断后重试",
                    "context_limit": "评审输入或分批调用数量超过限制，请缩小用例集或拆分过长内容后重试",
                }
                code = exc.code if exc.code in reasons else "request_failed"
                findings.append(ReviewFinding(
                    severity="high", category="review_incomplete", detail=code,
                    message="模型语义评审未完成：" + reasons[code] + "；不能据结构检查判定通过。",
                ))

        flagged_case_ids = {item.case_id for item in findings if item.case_id and is_blocking(item, context.get("clarification_policy", "strict"))}
        incomplete = any(item.category == "review_incomplete" for item in findings)
        for case in cases:
            case.review_status = "needs_attention" if incomplete or case.id in flagged_case_ids else "approved"

        penalty = sum({"high": 12, "medium": 5}.get(item.severity, 2) for item in findings)
        score = max(0, 100 - penalty)
        report = ReviewReport(
            score=score,
            findings=findings,
            added_case_ids=[],
            coverage_by_type=dict(coverage),
            requirement_coverage=dict(requirement_coverage),
        )
        return {"cases": [case.model_dump() for case in cases], "review": report.model_dump()}


class CaseRevisionAgent(Agent):
    """Closes the Review-Critique loop: consumes review findings and repairs the
    case set (fills missing case types, covers empty modules, fixes assertions,
    covers orphan requirements) instead of leaving findings as display-only."""

    name = "case_revision"

    SYSTEM = """You are a senior test case repair agent. You receive generated cases plus independent review findings.
Fix every fixable finding: add missing case types, cover modules and requirements without cases, and complete missing assertions.
Grouping modules inherit coverage from cases in descendant modules; do not add redundant parent cases solely for direct assignment.
Repair semantic findings on the identified cases, including preconditions, actions, test data and expected results.
Use canonical English case_type labels: functional, boundary, exception, permission, state_transition.
Keep each existing case's ID, module, requirement mappings and human status; do not delete existing cases.
Keep existing correct cases unchanged (same IDs). New cases must use new sequential IDs and set generated_by to "revision".
Return a JSON object with a cases array containing the FULL corrected case set."""

    def __init__(self, llm: OpenAICompatibleClient, generator: CaseGenerationAgent) -> None:
        super().__init__(llm)
        self.generator = generator

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        analysis = RequirementAnalysis.model_validate(payload["analysis"])
        tree = ModuleTree.model_validate(payload["module_tree"])
        cases = [TestCase.model_validate(item) for item in payload["cases"]]
        review = ReviewReport.model_validate(payload["review"])
        fixable = [item for item in review.findings if is_auto_fixable(item)]
        if not fixable:
            return {"cases": [case.model_dump() for case in cases], "added_case_ids": []}

        if self.llm.enabled and not context.get("force_demo"):
            case_schema = {
                "type": "object",
                "properties": {"cases": {"type": "array", "items": TestCase.model_json_schema()}},
                "required": ["cases"],
            }
            user_prompt = "Analysis:\n{}\nModule tree:\n{}\nCurrent cases:\n{}\nReview findings to fix:\n{}\nKnowledge:\n{}".format(
                    analysis.model_dump_json(),
                    tree.model_dump_json(),
                    json.dumps([case.model_dump() for case in cases], ensure_ascii=False),
                    json.dumps([item.model_dump() for item in fixable], ensure_ascii=False),
                    context.get("knowledge", ""),
                )
            feedback = ""
            for attempt in range(2):
                system = policy_system(self.SYSTEM, context)
                if attempt:
                    system += ("\nThe previous revision failed validation: " + feedback +
                               " Regenerate the FULL corrected case set from the same input. "
                               "Each step needs action and expected. Preserve existing IDs, modules, "
                               "mappings and human status. Do not invent missing business contracts.")
                try:
                    result = self.llm.generate_json(system, user_prompt, case_schema)
                    raw_cases = result.get("cases") if isinstance(result, dict) else None
                    if not isinstance(raw_cases, list):
                        raise LLMError("cases must be an array", code="invalid_schema")
                    revised = []
                    for index, item in enumerate(raw_cases):
                        try:
                            revised.append(TestCase.model_validate(item))
                        except ValidationError as exc:
                            paths = ["cases[{}].{} ({})".format(index, ".".join(map(str, e["loc"])), e["type"])
                                     for e in exc.errors()[:8]]
                            raise LLMError("; ".join(paths), code="invalid_schema") from exc
                    try:
                        protected = self._protect_existing_cases(cases, revised, fixable, tree)
                    except LLMError as exc:
                        raise LLMError("Revision violated existing case identity, module or step protection", code="invalid_schema") from exc
                    # Merely marking a case pending is not an effective repair.
                    before = [c.model_dump(exclude={"review_status"}) for c in cases]
                    after = [c.model_dump(exclude={"review_status"}) for c in protected]
                    if before == after:
                        raise LLMError("Revision made no accepted change to the cases", code="revision_unchanged")
                    previous_ids = {case.id for case in cases}
                    added = [case.id for case in protected if case.id not in previous_ids]
                    return {"cases": [case.model_dump() for case in protected], "added_case_ids": added}
                except LLMError as exc:
                    if attempt or exc.code not in {"invalid_json", "invalid_schema", "revision_unchanged"}:
                        raise
                    feedback = "Return a single valid JSON object" if exc.code == "invalid_json" else str(exc)
        return self._demo(analysis, tree, cases, fixable, context.get("selected_skills"))

    @staticmethod
    def _protect_existing_cases(
        previous: List[TestCase],
        revised: List[TestCase],
        fixable: List[ReviewFinding],
        tree: ModuleTree,
    ) -> List[TestCase]:
        revised_ids = [case.id for case in revised]
        if len(revised_ids) != len(set(revised_ids)):
            raise LLMError("Revision output contains duplicate case IDs")

        previous_by_id = {case.id: case for case in previous}
        revised_by_id = {case.id: case for case in revised}
        missing_ids = [case_id for case_id in previous_by_id if case_id not in revised_by_id]
        if missing_ids:
            raise LLMError(
                "Revision output dropped existing cases: {}".format(", ".join(missing_ids[:10]))
            )

        known_modules = {module.id for module in ModulePlanningAgent._walk(tree.modules)}
        added = [case for case in revised if case.id not in previous_by_id]
        unknown_modules = [case.module_id for case in added if case.module_id not in known_modules]
        if unknown_modules:
            raise LLMError(
                "Revision output uses unknown modules: {}".format(", ".join(sorted(set(unknown_modules))))
            )

        assertion_ids = {
            finding.case_id
            for finding in fixable
            if finding.category == "assertion" and finding.case_id
        }
        semantic_ids = {finding.case_id for finding in fixable
                        if finding.category == "semantic" and finding.case_id}
        type_ids = {finding.case_id for finding in fixable
                    if finding.category == "case_type" and finding.case_id}
        protected: List[TestCase] = []
        for original in previous:
            if original.id in semantic_ids:
                candidate = revised_by_id[original.id].model_copy(deep=True)
                if candidate.module_id != original.module_id:
                    raise LLMError("Semantic repair changed module for {}".format(original.id))
                if not candidate.steps or any(not s.action.strip() or not s.expected.strip() for s in candidate.steps):
                    raise LLMError("Semantic repair requires executable steps for {}".format(original.id))
                candidate.requirement_ids = list(dict.fromkeys(original.requirement_ids + candidate.requirement_ids))
                candidate.source_evidence = list(dict.fromkeys(original.source_evidence + candidate.source_evidence))
                candidate.human_status = original.human_status
                candidate.generated_by = original.generated_by
                candidate.review_status = "pending"
                protected.append(candidate)
                continue
            if original.id not in assertion_ids:
                repaired = original.model_copy(deep=True)
                if original.id in type_ids:
                    repaired.case_type = revised_by_id[original.id].case_type
                    repaired.review_status = "pending"
                protected.append(repaired)
                continue
            candidate = revised_by_id[original.id]
            if len(candidate.steps) != len(original.steps) or any(
                candidate.steps[index].action != step.action
                for index, step in enumerate(original.steps)
            ):
                raise LLMError(
                    "Assertion repair changed step actions for {}".format(original.id)
                )
            repaired = original.model_copy(deep=True)
            for index, step in enumerate(repaired.steps):
                if candidate.steps[index].expected.strip():
                    step.expected = candidate.steps[index].expected
            repaired.review_status = "pending"
            protected.append(repaired)

        for case in added:
            case.generated_by = "revision"
            case.human_status = "pending"
            protected.append(case)
        return protected

    def _demo(
        self,
        analysis: RequirementAnalysis,
        tree: ModuleTree,
        cases: List[TestCase],
        fixable: List[ReviewFinding],
        selected_skills=None,
    ) -> Dict[str, Any]:
        req_map = {req.id: req for req in analysis.atomic_requirements}
        module_map = {module.id: module for module in ModulePlanningAgent._walk(tree.modules)}
        next_index = 1 + max(
            [int(match.group(1)) for case in cases for match in [re.match(r"TC-(\d+)", case.id)] if match] or [0]
        )
        added: List[str] = []

        def new_case(module: TestModule, case_type: str) -> TestCase:
            nonlocal next_index
            evidence = [req_map[rid].source_quote for rid in module.requirement_ids if rid in req_map][:3] or [analysis.summary]
            case = self.generator.build_case(
                case_id="TC-{:04d}".format(next_index),
                module=module,
                case_type=case_type,
                source=evidence[0],
                evidence=evidence,
                analysis=analysis,
                first_module_id=tree.modules[0].id,
                generated_by="revision",
            )
            next_index += 1
            added.append(case.id)
            return case

        def module_for_type(case_type: str) -> TestModule:
            for module in ModulePlanningAgent._walk(tree.modules):
                if case_type in self.generator._case_types(module, resolve_skills(module.name, selected_skills)):
                    return module
            return tree.modules[0]

        for finding in fixable:
            if finding.category == "case_type" and finding.detail:
                if not any(case.case_type == finding.detail for case in cases):
                    cases.append(new_case(module_for_type(finding.detail), finding.detail))
            elif finding.category == "module_coverage" and finding.detail in module_map:
                if not any(case.module_id == finding.detail for case in cases):
                    module = module_map[finding.detail]
                    case_types = self.generator._case_types(module, resolve_skills(module.name, selected_skills))
                    cases.append(new_case(module, case_types[0] if case_types else "functional"))
            elif finding.category == "assertion" and finding.detail:
                for case in cases:
                    if case.id == finding.detail:
                        for step in case.steps:
                            if not step.expected.strip():
                                step.expected = "系统行为与需求声明一致（自动修复占位断言，需人工确认具体预期）"
                        case.review_status = "pending"
            elif finding.category == "requirement_coverage" and finding.detail in req_map:
                covered = {rid for case in cases for rid in case.requirement_ids}
                if finding.detail not in covered:
                    requirement = req_map[finding.detail]
                    host = next((module for module in ModulePlanningAgent._walk(tree.modules) if finding.detail in module.requirement_ids), None)
                    module = host or TestModule(
                        id="MOD-UN",
                        name=UNASSIGNED_MODULE_NAME,
                        objective="需求补覆盖",
                        requirement_ids=[finding.detail],
                        risks=["覆盖缺口"],
                        case_types=["functional"],
                    )
                    case = new_case(module, "functional")
                    case.requirement_ids = [finding.detail]
                    case.source_evidence = [requirement.source_quote]
                    cases.append(case)
        return {"cases": [case.model_dump() for case in cases], "added_case_ids": added}
