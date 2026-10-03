import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from .agents import (
    AgentHarness,
    KnowledgeRetrievalAgent,
    ModuleRevisionAgent,
)
from .adaptive_memory import AdaptiveMemory
from .review_policy import is_auto_fixable
from .core_agents import (
    CORE_AGENT_NAMES,
    ModulePlanningAgent,
    QualityCriticAgent,
    RequirementUnderstandingAgent,
    TestCaseGenerationAgent,
)
from .chunk_management import KnowledgeChunkManager
from .case_memory import CaseConversationMemory
from .feedback import case_fingerprint, example_document
from .document_parser import DocumentParsingAgent
from .domain_registry import detect_ticket_type, get_domain_facts
from .domain_equipment import build_equipment_knowledge
from .ebt_dataset import EBT_DATASET_ID, EBTRepository
from .evaluation import DATASETS, OfflineEvaluationAgent, dataset_for
from .knowledge_ingestion import KnowledgeIngestionPipeline
from .llm import OpenAICompatibleClient
from .module_memory import ModuleConversationMemory
from .mindmap import MindMapConversionTool
from .observability import TraceManager
from .public_benchmarks import PublicBenchmarkService
from .xmind_export import XMindExportAdapter
from .models import (
    CaseFeedback,
    ConversationDecisionState,
    EvaluationReport,
    KnowledgeContext,
    ModuleReviewReport,
    ModuleTree,
    ParsedDocument,
    ProjectMetrics,
    ProjectState,
    RequirementInput,
    ReviewReport,
    TestCase,
)
from .retrieval_evaluation import RETRIEVAL_DATASETS, RetrievalEvaluationAgent
from .skills import select_skills, resolve_skills
from .store import JsonStore
from .tooling import ReActRuntime, ToolRegistry
from .template_evolution import (
    TemplateEvolutionAgent,
    rules_as_knowledge,
    templates_as_knowledge,
)


class PipelineError(RuntimeError):
    pass


FEEDBACK_ACTIONS = {"adopted", "edited", "rejected"}
EDITABLE_CASE_FIELDS = {"title", "priority", "case_type", "preconditions", "steps", "test_data", "risk_tags"}

# Badcase triage categories used by this demo:
# coverage -> knowledge gap, quality -> generation quality, maintenance -> stale template/rule.
COVERAGE_WORDS = ["缺少", "漏", "遗漏", "没覆盖", "未覆盖", "少了", "没有考虑", "缺失"]
MAINTENANCE_WORDS = ["过时", "下线", "废弃", "已变更", "不再适用", "历史版本", "旧版"]


def classify_badcase(reason: str) -> str:
    if any(word in reason for word in COVERAGE_WORDS):
        return "coverage"
    if any(word in reason for word in MAINTENANCE_WORDS):
        return "maintenance"
    return "quality"


class TestCaseOrchestrator:
    def __init__(self, store: JsonStore, *, knowledge_policy: str = "application") -> None:
        self.knowledge_policy = knowledge_policy
        self.clarification_policy = "strict"
        self.store = store
        self.tracer = TraceManager(store.root)
        self.store.set_tracer(self.tracer)
        self.llm = OpenAICompatibleClient(self.tracer)
        self.harness = AgentHarness(self.tracer)
        if knowledge_policy == "application":
            self.store.upsert_knowledge(build_equipment_knowledge())
        self.document_agent = DocumentParsingAgent(self.llm)
        self.retrieval_agent = KnowledgeRetrievalAgent(self.llm, self.store.knowledge_index_file)
        self.tool_registry = self._build_tool_registry()
        react_steps = max(1, min(8, int(os.getenv("REACT_MAX_STEPS", "4"))))
        self.react_runtime = ReActRuntime(
            self.llm, self.tool_registry, max_steps=react_steps
        )
        self.requirement_agent = RequirementUnderstandingAgent(
            self.llm, self.react_runtime
        )
        self.module_agent = ModulePlanningAgent(self.llm)
        self.module_memory = ModuleConversationMemory(tracer=self.tracer)
        self.case_memory = CaseConversationMemory(tracer=self.tracer)
        self.adaptive_memory = AdaptiveMemory(self.store, self.llm, self.tracer)
        self.adaptive_memory.migrate_legacy()
        self.module_revision_service = ModuleRevisionAgent(self.llm)
        self.case_agent = TestCaseGenerationAgent(self.llm, self.react_runtime)
        self.quality_agent = QualityCriticAgent(self.llm)
        self.evaluation_agent = OfflineEvaluationAgent(self.llm)
        self.template_agent = TemplateEvolutionAgent(self.llm)
        self.knowledge_ingestion = KnowledgeIngestionPipeline()
        self.chunk_manager = KnowledgeChunkManager(self.store)
        self.retrieval_evaluation_agent = RetrievalEvaluationAgent(self.retrieval_agent)
        self.ebt_repository = EBTRepository(self.store.root / "external" / "ebt")
        self.public_benchmarks = PublicBenchmarkService(
            self.store.root, self.ebt_repository
        )
        self.mindmap_tool = MindMapConversionTool()
        self.xmind_exporter = XMindExportAdapter()
        # Conservative legacy migration: only snapshot-proven decisions create examples.
        # Project/run files are not rewritten on startup.
        for project in self.store.list_projects():
            self.store.sync_case_examples(project)

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry(self.tracer)
        research_agents = ["requirement_understanding", "case_generation"]
        registry.register(
            "search_knowledge",
            "Search scoped active knowledge with hybrid RAG and return top evidence.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            self._tool_search_knowledge,
            research_agents,
        )
        registry.register(
            "get_domain_facts",
            "Read interfaces, states, events, and permissions for a ticket type.",
            {
                "type": "object",
                "properties": {"ticket_type": {"type": "string"}},
                "required": ["ticket_type"],
            },
            self._tool_get_domain_facts,
            research_agents,
        )
        registry.register(
            "get_team_memory",
            "Read team testing rules scoped to a ticket type.",
            {
                "type": "object",
                "properties": {"ticket_type": {"type": "string"}},
                "required": ["ticket_type"],
            },
            self._tool_get_team_memory,
            research_agents,
        )
        registry.register(
            "get_test_skills",
            "Select test strategy skills matching requirement or module text.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            self._tool_get_test_skills,
            ["case_generation"],
        )
        return registry

    def _tool_search_knowledge(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        result = self.search_knowledge(str(arguments["query"]))
        return {
            "ticket_type": result["ticket_type"],
            "retrieval_mode": result["retrieval_mode"],
            "hits": result["hits"][:6],
        }

    def _tool_get_domain_facts(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.knowledge_policy == "sample_only":
            return {"available": False, "reason": "Benchmark uses supplied sample material only."}
        ticket_type = str(arguments["ticket_type"])
        facts = get_domain_facts(ticket_type)
        return (
            facts.model_dump()
            if facts
            else {"ticket_type": ticket_type, "knowledge_gap": True}
        )

    def _tool_get_team_memory(self, arguments: Dict[str, Any]) -> str:
        self.adaptive_memory.migrate_legacy()
        query = str(arguments.get("query") or arguments["ticket_type"])
        return self.adaptive_memory.search(
            query,
            ticket_type=str(arguments["ticket_type"]),
            top_k=10,
            token_budget=1200,
        ).context

    @staticmethod
    def _tool_get_test_skills(arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "instruction": skill.instruction,
                "case_types": skill.case_types,
            }
            for skill in select_skills(str(arguments["text"]))
        ]

    def tool_catalog(self) -> Dict[str, Any]:
        return {
            "core_agents": list(CORE_AGENT_NAMES),
            "tools_by_agent": {
                agent: self.tool_registry.schemas_for(agent)
                for agent in CORE_AGENT_NAMES
            },
            "output_tools_by_agent": {
                "module_planning": [self.mindmap_tool.schema(), self.xmind_exporter.schema()],
                "case_generation": [self.mindmap_tool.schema(), self.xmind_exporter.schema()],
            },
        }

    def mindmap(self, project: ProjectState, view: str = "cases"):
        return self.mindmap_tool.convert(project, view)

    def export_xmind(self, project: ProjectState, view: str = "cases") -> bytes:
        return self.xmind_exporter.export(self.mindmap(project, view))

    @staticmethod
    def _research_context(result: Dict[str, Any]) -> str:
        return "{}\n{}".format(
            result.get("summary", ""),
            "\n".join(str(item) for item in result.get("observations", [])),
        ).strip()
    def parse_document(self, filename: str, data: bytes) -> Dict[str, Any]:
        with self.tracer.span(
            "document.extract",
            kind="document",
            attributes={"filename": filename, "bytes": len(data)},
        ) as span:
            extracted = self.document_agent.extract(
                {"filename": filename, "data": data},
                {},
            )
            if span:
                span.output_summary = "{} chunks, {} pages".format(
                    len(extracted.chunks),
                    extracted.extraction_summary.get("total_pages", 0),
                )
        document, trace = self.harness.execute(
            self.document_agent,
            {"document": extracted.model_dump()},
            {},
        )
        return {"document": document.model_dump(), "trace": trace.model_dump()}

    def ingest_document(
        self,
        filename: str,
        data: bytes,
        ticket_type: str,
        doc_type: str = "",
        version: str = "1.0",
        effective_at: str = "",
        expires_at: str = "",
        chunking_version: str = "section-child-v1",
    ) -> Dict[str, Any]:
        # Knowledge ingestion is an independent entry point. It indexes extracted
        # evidence directly and does not run PRD compression.
        with self.tracer.span(
            "document.extract",
            kind="document",
            attributes={"filename": filename, "bytes": len(data)},
        ):
            document = self.document_agent.extract(
                {"filename": filename, "data": data},
                {},
            )
        with self.tracer.span(
            "rag.chunk",
            kind="rag",
            attributes={
                "ticket_type": ticket_type,
                "chunking_version": chunking_version,
            },
        ) as chunk_span:
            knowledge_documents = self.knowledge_ingestion.build(
                document,
                ticket_type=ticket_type,
                doc_type=doc_type,
                version=version,
                effective_at=effective_at,
                expires_at=expires_at,
                chunking_version=chunking_version,
            )
            if chunk_span:
                chunk_span.output_summary = "{} knowledge chunks".format(
                    len(knowledge_documents)
                )
        if not knowledge_documents:
            raise ValueError("Document produced no knowledge chunks")
        source_id = knowledge_documents[0].source_id
        with self.tracer.span(
            "store.replace_knowledge",
            kind="storage",
            attributes={"source_id": source_id},
        ) as store_span:
            replacement = self.store.replace_source_knowledge(
                source_id,
                [item.model_dump() for item in knowledge_documents],
            )
            if store_span:
                store_span.output_summary = "changed={}, invalidated={}".format(
                    replacement["changed"], replacement["invalidated"]
                )
        return {
            "document": document.model_dump(),
            "knowledge_documents": [item.model_dump() for item in knowledge_documents],
            "changed": replacement["changed"],
            "invalidated": replacement["invalidated"],
            "invalidated_ids": replacement["invalidated_ids"],
            "source_id": source_id,
            "trace": {
                "agent": "document_extraction",
                "status": "success",
                "mode": "deterministic",
                "tool_calls": [],
            },
        }

    def convert_knowledge_chunk(self, document_id: str) -> Dict[str, Any]:
        return self.chunk_manager.convert_to_parent_child(document_id)
    def suggest_knowledge_split(self, document_id: str) -> Dict[str, Any]:
        return self.chunk_manager.suggest_split(document_id)
    def split_knowledge_chunk(
        self, document_id: str, parts: List[str]
    ) -> Dict[str, Any]:
        return self.chunk_manager.split(document_id, parts)

    def merge_knowledge_chunks(
        self, document_ids: List[str], title: str = ""
    ) -> Dict[str, Any]:
        return self.chunk_manager.merge(document_ids, title)
    def search_knowledge(self, query: str, project_id: str = "") -> Dict[str, Any]:
        context = self._retrieve(query, project_id=project_id)
        return context.model_dump()

    def _retrieve(self, query: str, project_id: str = "") -> KnowledgeContext:
        if self.knowledge_policy == "sample_only":
            return KnowledgeContext(query=query)
        with self.tracer.span(
            "rag.retrieve",
            kind="rag",
            attributes={"query_chars": len(query)},
            input_value=query,
        ) as span:
            context = self.retrieval_agent.run(
                query, {"documents": self.store.project_knowledge(project_id), "project_id": project_id}
            )
            if span:
                span.output_summary = "{} hits from {} candidates".format(
                    len(context.hits), context.candidate_count
                )
                span.attributes.update({
                    "ticket_type": context.ticket_type,
                    "retrieval_mode": context.retrieval_mode,
                    "candidate_count": context.candidate_count,
                    "selected_count": len(context.hits),
                    "filtered_count": context.filtered_count,
                    "embedding_provider": context.trace.get(
                        "embedding_provider", ""
                    ),
                })
            return context

    def add_memory_rule(
        self,
        rule: str,
        ticket_type: str = "COMMON",
        *,
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        fact_key: str = "",
        valid_from: str = "",
        valid_to: str = "",
    ) -> Dict[str, Any]:
        legacy = self.store.get_memory()
        result = self.adaptive_memory.add_fact(
            rule,
            memory_type="team_rule",
            user_id=user_id,
            project_id=project_id,
            agent_id=agent_id,
            ticket_type=ticket_type,
            source="manual",
            importance=0.85,
            fact_key=fact_key, valid_from=valid_from, valid_to=valid_to,
        )
        return {
            "added": result["added"],
            "duplicate_ids": result["duplicate_ids"],
            "records": [item.model_dump() for item in result["records"]],
            "legacy": legacy,
        }

    def remember_text(self, text: str, **scope: Any) -> Dict[str, Any]:
        result = self.adaptive_memory.remember(text, **scope)
        return {
            "added": result["added"],
            "extracted": result.get("extracted", 0),
            "duplicate_ids": result["duplicate_ids"],
            "records": [item.model_dump() for item in result["records"]],
        }

    def search_memory(self, query: str, **scope: Any) -> Dict[str, Any]:
        return self.adaptive_memory.search(query, **scope).model_dump()

    def _restore_conversation_memories(self, project, source, version):
        state = (project.module_decision_state if source == "module_conversation"
                 else project.case_decision_state)
        valid_versions = {entry.source_version_id for entry in state.entries
                          if entry.status == "active" and entry.source_version_id}
        valid_versions.add(version.id)
        valid_messages = {entry.source_message_id for entry in state.entries if entry.status == "active"}
        with self.store.memory_transaction():
            records = self.store.list_memory_records(include_inactive=True)
            for item in records:
                if item.project_id != project.id or item.source != source:
                    continue
                message_id = item.metadata.get("source_message_id")
                is_current = (message_id in valid_messages if message_id
                              else item.source_version_id in valid_versions)
                if is_current:
                    if item.status == "inactive" and item.invalidation_reason == "version_restored":
                        item.status = "active"
                        item.valid_to = ""
                        item.invalidation_reason = ""
                elif item.status == "active":
                    item.status = "inactive"
                    item.invalidation_reason = "version_restored"
            self.store.save_memory_records(records)

    def _remember_conversation_decision(self, project, source, agent_id, version, ticket_type):
        state = (project.module_decision_state if source == "module_conversation"
                 else project.case_decision_state)
        # Retire replaced entries before deduplication, including a repeated value
        # with a new message ID; otherwise its only copy could be retired afterward.
        self._restore_conversation_memories(project, source, version)
        for entry in state.entries:
            if (entry.status != "active" or entry.kind == "question"
                    or entry.source_version_id != version.id):
                continue
            self.adaptive_memory.remember(
                entry.content, project_id=project.id, agent_id=agent_id,
                source_run_id=self.tracer.current_trace_id,
                source_version_id=version.id, ticket_type=ticket_type,
                source=source, source_ref=entry.source_message_id,
                metadata={"source_message_id": entry.source_message_id, "decision_key": entry.key},
            )

    def _retire_project_memories(self, project, sources):
        with self.store.memory_transaction():
            records = self.store.list_memory_records(include_inactive=True)
            for item in records:
                if item.project_id == project.id and item.source in sources and item.status == "active":
                    item.status = "inactive"
                    item.invalidation_reason = "project_structure_changed"
            self.store.save_memory_records(records)

    def memory_catalog(self, include_inactive: bool = False) -> Dict[str, Any]:
        legacy = self.store.get_memory()
        return {
            **legacy,
            "records": [
                item.model_dump() for item in self.store.list_memory_records(include_inactive=include_inactive)
            ],
            "stats": self.adaptive_memory.stats(),
        }

    def evaluate_retrieval(
        self, dataset_id: str = "EQUIPMENT-RAG-V1", k: int = 5
    ) -> Dict[str, Any]:
        if dataset_id == EBT_DATASET_ID:
            dataset, documents = self.ebt_repository.build_retrieval_dataset()
            return self.retrieval_evaluation_agent.run_dataset(
                documents, dataset, k
            ).model_dump()
        return self.retrieval_evaluation_agent.run(
            self.store.list_knowledge(), dataset_id, k
        ).model_dump()

    def retrieval_datasets(self) -> List[Dict[str, Any]]:
        datasets = [dataset.model_dump() for dataset in RETRIEVAL_DATASETS.values()]
        if self.ebt_repository.ready:
            dataset, _ = self.ebt_repository.build_retrieval_dataset()
            datasets.append(dataset.model_dump())
        return datasets

    def ebt_status(self) -> Dict[str, Any]:
        return self.ebt_repository.status()

    def import_ebt(self) -> Dict[str, Any]:
        return self.ebt_repository.download()

    def benchmark_catalog(self) -> Dict[str, Any]:
        return self.public_benchmarks.catalog()

    def import_public_benchmark(self, dataset_id: str) -> Dict[str, Any]:
        return self.public_benchmarks.import_dataset(dataset_id)

    def benchmark_report(self, report_id: str) -> Dict[str, Any]:
        return self.public_benchmarks.get_report(report_id)

    def run_public_benchmark(
        self,
        suite: str,
        limit: int = 3,
        split: str = "test",
        mode: str = "offline",
        execution: str = "workflow",
        human_policy: str = "pause",
        max_steps: int = 12,
        llm_options: Optional[Dict[str, Any]] = None,
        on_event=None,
        clarification_policy: str = "strict",
    ) -> Dict[str, Any]:
        with self.tracer.span(
            "benchmark.run",
            kind="evaluation",
            attributes={
                "suite": suite,
                "limit": limit,
                "split": split,
                "mode": mode,
                "execution": execution,
                "human_policy": human_policy,
            },
        ) as span:
            report = self.public_benchmarks.run(
                suite,
                limit,
                split,
                mode,
                lambda root: TestCaseOrchestrator(JsonStore(root), knowledge_policy="sample_only"),
                execution=execution, human_policy=human_policy, max_steps=max_steps, llm_options=llm_options, on_event=on_event,
                clarification_policy=clarification_policy,
            )
            if span:
                span.output_summary = "{} score={} samples={}".format(
                    report["dataset_id"], report["score"], report["sample_count"]
                )
            return report

    def analyze(self, project: ProjectState, instruction: str = "", supervisor_evidence: str = "") -> ProjectState:
        knowledge_context = self._retrieve(project.requirement, project_id=project.id)
        context = self._context(
            project.requirement,
            knowledge_context,
            project_id=project.id,
            agent_id="requirement_understanding",
        )
        ticket_type = detect_ticket_type(project.requirement).key
        self._supervisor_context(context, None, supervisor_evidence)
        context["research_payload"] = {
            "query": project.requirement,
            "ticket_type": ticket_type,
            "objective": (
                "Identify missing requirement evidence, domain contracts, "
                "state transitions, permissions, and team rules before analysis."
            ),
        }
        result, trace = self.harness.execute(
            self.requirement_agent,
            RequirementInput(title=project.title, content=project.requirement, context="\n".join(filter(None, [project.context, instruction]))),
            context,
        )
        self._retire_project_memories(project, {"module_conversation", "case_conversation", "human_gate"})
        project.analysis = result
        project.module_tree = None
        project.module_review = None
        project.module_conversation = []
        project.module_versions = []
        project.module_memory_summary = ""
        project.module_decision_state = ConversationDecisionState()
        project.cases = []
        project.case_conversation = []
        project.case_versions = []
        project.case_memory_summary = ""
        project.case_decision_state = ConversationDecisionState()
        project.review = None
        project.evaluation = None
        project.phase = "analyzed"
        project.traces.append(trace)
        self.store.save_project(project)
        return project

    def plan_modules(self, project: ProjectState) -> ProjectState:
        return self.operate_modules(project, mode="full")

    def operate_modules(
        self,
        project: ProjectState,
        mode: str = "full",
        instruction: str = "",
        target_module_id: str = "",
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        supervisor_evidence: str = "",
    ) -> ProjectState:
        if not project.analysis:
            raise PipelineError("Run requirement analysis first")
        allowed = {"full", "regenerate", "continue", "targeted", "chat"}
        if mode not in allowed:
            raise PipelineError("Unsupported module mode: {}".format(mode))
        if mode in {"continue", "targeted", "chat"} and not project.module_tree:
            raise PipelineError("{} mode requires an existing module tree".format(mode))
        if mode == "targeted" and not target_module_id:
            raise PipelineError("Select a target module first")

        has_explicit_instruction = bool(instruction.strip())
        instruction = instruction.strip() or {
            "full": "根据需求分析生成完整测试模块树",
            "regenerate": "清空既有结果并重新生成完整测试模块树",
            "continue": "保留已有模块并补充遗漏的测试关注点",
            "targeted": "重新生成指定模块及其子模块",
            "chat": "根据本轮对话修改模块树",
        }[mode]
        self.module_memory.remember_user(
            project, instruction, mode, target_module_id
        )
        if progress:
            progress({"type": "stage", "stage": "context", "message": "正在构建需求、知识与历史会话上下文"})
        context = self._context(
            project.requirement + "\n" + instruction,
            project_id=project.id,
            agent_id="module_planning",
        )
        context["module_conversation"] = self.module_memory.context(project)
        self._supervisor_context(context, None, supervisor_evidence)
        if progress:
            context["stream_callback"] = lambda delta: progress({
                "type": "delta", "stage": "llm", "delta": delta
            })
            progress({"type": "stage", "stage": "llm", "message": "模块生成 Agent 正在流式生成"})
        payload = {
            "analysis": project.analysis.model_dump(),
            "mode": mode,
            "instruction": instruction,
            "target_module_id": target_module_id,
            "existing_tree": (
                project.module_tree.model_dump() if project.module_tree else None
            ),
        }
        result, trace = self.harness.execute(self.module_agent, payload, context)
        if progress:
            progress({
                "type": "stage",
                "stage": "critique",
                "message": "正在检查需求覆盖、重复模块与模块粒度",
                "fallback": trace.mode == "fallback",
            })
        result, module_review, critic_traces = self._review_module_tree(
            project.analysis, result, context
        )
        project.module_tree = result
        project.module_review = module_review
        version = self.module_memory.remember_result(
            project, result, mode, instruction, target_module_id
        )
        if has_explicit_instruction and mode in {"continue", "targeted", "chat"}:
            ticket_type = (
                project.analysis.ticket_types[0]
                if project.analysis.ticket_types
                else "COMMON"
            )
            self._remember_conversation_decision(
                project, "module_conversation", "module_planning", version, ticket_type
            )
        if len(project.module_versions) > 50:
            project.module_versions = project.module_versions[-50:]
        project.cases = []
        self._retire_project_memories(project, {"case_conversation", "human_gate"})
        project.case_conversation = []
        project.case_versions = []
        project.case_memory_summary = ""
        project.case_decision_state = ConversationDecisionState()
        project.review = None
        project.evaluation = None
        project.phase = "modules_planned"
        project.traces.extend([trace] + critic_traces)
        self.store.save_project(project)
        if progress:
            progress({
                "type": "stage",
                "stage": "persisted",
                "message": "模块树、会话 Memory 与版本快照已保存",
                "version_id": version.id,
            })
        return project

    def _review_module_tree(
        self,
        analysis: Any,
        tree: ModuleTree,
        context: Dict[str, Any],
    ) -> Tuple[ModuleTree, ModuleReviewReport, List[Any]]:
        review_payload = {
            "target": "module",
            "analysis": analysis.model_dump(),
            "module_tree": tree.model_dump(),
        }
        module_review, critic_trace = self.harness.execute(
            self.quality_agent, review_payload, context
        )
        traces = [critic_trace]
        if any(
            item.category in {"duplicate_id", "duplicate_name", "empty_module"}
            for item in module_review.findings
        ):
            tree = self.module_revision_service.run(
                {
                    "module_tree": tree.model_dump(),
                    "module_review": module_review.model_dump(),
                },
                context,
            )
            review_payload["module_tree"] = tree.model_dump()
            module_review, second_trace = self.harness.execute(
                self.quality_agent, review_payload, context
            )
            module_review.rounds = 1
            traces.append(second_trace)
        return tree, module_review, traces

    def restore_module_version(
        self, project: ProjectState, version_id: str
    ) -> ProjectState:
        if not project.analysis:
            raise PipelineError("Run requirement analysis first")
        version = next(
            (item for item in project.module_versions if item.id == version_id),
            None,
        )
        if not version:
            raise PipelineError("Module version not found: {}".format(version_id))
        self.module_memory.restore_state(project, version)
        instruction = "恢复模块版本 {}".format(version_id)
        self.module_memory.remember_user(project, instruction, "restore")
        tree = version.module_tree.model_copy(deep=True)
        tree.confirmed = False
        context = self._context(
            project.requirement,
            project_id=project.id,
            agent_id="module_planning",
        )
        tree, review, traces = self._review_module_tree(
            project.analysis, tree, context
        )
        self._restore_conversation_memories(project, "module_conversation", version)
        project.module_tree = tree
        project.module_review = review
        self.module_memory.remember_result(
            project, tree, "restore", instruction
        )
        project.cases = []
        self._retire_project_memories(project, {"case_conversation", "human_gate"})
        project.case_conversation = []
        project.case_versions = []
        project.case_memory_summary = ""
        project.case_decision_state = ConversationDecisionState()
        project.review = None
        project.evaluation = None
        project.phase = "modules_planned"
        project.traces.extend(traces)
        self.store.save_project(project)
        return project

    def confirm_modules(self, project: ProjectState, modules: List[Dict[str, Any]]) -> ProjectState:
        if not project.module_tree:
            raise PipelineError("Generate a module tree first")
        tree = ModuleTree(
            modules=modules,
            coverage_notes=project.module_tree.coverage_notes,
            confirmed=True,
        )
        project.module_tree = tree
        self.module_memory.remember_user(
            project, "人工编辑并确认当前模块树", "chat"
        )
        self.module_memory.remember_result(
            project, tree, "chat", "人工编辑并确认当前模块树"
        )
        ticket_type = (
            project.analysis.ticket_types[0]
            if project.analysis and project.analysis.ticket_types
            else "COMMON"
        )
        self._retire_project_memories(project, {"human_gate"})
        self.adaptive_memory.add_fact(
            "人工确认测试模块：{}".format(
                "、".join(item.name for item in tree.modules)
            ),
            memory_type="project_decision",
            project_id=project.id,
            agent_id="module_planning",
            ticket_type=ticket_type,
            source="human_gate",
            source_version_id=project.module_versions[-1].id,
            fact_key="confirmed_modules",
            importance=0.9,
        )
        project.phase = "modules_confirmed"
        self.store.save_project(project)
        return project
    def generate_cases(self, project: ProjectState) -> ProjectState:
        return self.operate_cases(project, mode="full")

    def operate_cases(
        self,
        project: ProjectState,
        mode: str = "full",
        instruction: str = "",
        target_module_id: str = "",
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        selected_skills: Optional[List[str]] = None,
        supervisor_evidence: str = "",
        persist_decision: bool = True,
    ) -> ProjectState:
        if selected_skills is not None:
            resolve_skills("", selected_skills)
        if not project.analysis or not project.module_tree or not project.module_tree.confirmed:
            raise PipelineError("Confirm the module tree before generating cases")
        allowed = {"full", "regenerate", "continue", "targeted", "chat"}
        if mode not in allowed:
            raise PipelineError("Unsupported case mode: {}".format(mode))
        if mode in {"continue", "targeted", "chat"} and not project.cases:
            raise PipelineError("{} mode requires existing test cases".format(mode))
        all_modules = self.module_agent.strategy._walk(project.module_tree.modules)
        module_by_id = {item.id: item for item in all_modules}
        if mode == "targeted" and not target_module_id:
            raise PipelineError("Select a target module first")
        if target_module_id and target_module_id not in module_by_id:
            raise PipelineError("Target module not found: {}".format(target_module_id))

        has_explicit_instruction = bool(instruction.strip())
        instruction = instruction.strip() or {
            "full": "根据已确认模块生成完整测试用例集",
            "regenerate": "清空已有用例并重新生成完整测试用例集",
            "continue": "保留已有用例并补充遗漏场景",
            "targeted": "重新生成指定模块及其子模块的测试用例",
            "chat": "根据本轮对话修改测试用例集",
        }[mode]
        self.case_memory.remember_user(
            project, instruction, mode, target_module_id
        )
        if progress:
            progress({
                "type": "stage",
                "stage": "context",
                "message": "正在构建模块、RAG、Skills 与历史用例会话上下文",
            })
        context = self._context(
            project.requirement + "\n" + instruction,
            project_id=project.id,
            agent_id="case_generation",
        )
        context["case_conversation"] = self.case_memory.context(project)
        self._supervisor_context(context, selected_skills, supervisor_evidence)
        if progress:
            context["stream_callback"] = lambda delta: progress({
                "type": "delta", "stage": "llm", "delta": delta
            })
            progress({
                "type": "stage",
                "stage": "llm",
                "message": "用例生成 Agent 正在流式生成",
            })

        generation_tree = project.module_tree
        if mode == "targeted":
            generation_tree = ModuleTree(
                modules=[module_by_id[target_module_id].model_copy(deep=True)],
                coverage_notes=project.module_tree.coverage_notes,
                confirmed=True,
            )
        payload = {
            "analysis": project.analysis.model_dump(),
            "module_tree": generation_tree.model_dump(),
            "mode": mode,
            "instruction": instruction,
            "target_module_id": target_module_id,
            "existing_cases": [item.model_dump() for item in project.cases],
        }
        ticket_type = (
            project.analysis.ticket_types[0]
            if project.analysis.ticket_types
            else detect_ticket_type(project.requirement).key
        )
        module_text = " ".join(
            "{} {}".format(module.name, module.objective)
            for module in self.module_agent.strategy._walk(generation_tree.modules)
        )
        context["research_payload"] = {
            "query": "{}\n{}\n{}".format(
                project.requirement, module_text, instruction
            ),
            "skill_text": "{}\n{}".format(
                module_text, project.analysis.model_dump_json()
            ),
            "ticket_type": ticket_type,
            "objective": (
                "Collect missing domain evidence, relevant historical cases, "
                "team rules, and test strategies before writing test cases."
            ),
        }
        candidate, trace = self.harness.execute(self.case_agent, payload, context)
        if progress:
            progress({
                "type": "stage",
                "stage": "merge",
                "message": "正在校验模块边界、稳定 ID 与已有用例保护",
                "fallback": trace.mode == "fallback",
            })
        cases = self._apply_case_mode(
            project, mode, candidate, target_module_id
        )
        for item in cases:
            item.review_status = "pending"
        if progress:
            for index, item in enumerate(cases, 1):
                progress({
                    "type": "case",
                    "index": index,
                    "total": len(cases),
                    "module_id": item.module_id,
                    "case": item.model_dump(),
                })

        project.cases = cases
        project.review = None
        project.evaluation = None
        version = self.case_memory.remember_result(
            project, cases, mode, instruction, target_module_id
        )
        if persist_decision and has_explicit_instruction and mode in {"continue", "targeted", "chat"}:
            self._remember_conversation_decision(
                project, "case_conversation", "case_generation", version, ticket_type
            )
        if len(project.case_versions) > 50:
            project.case_versions = project.case_versions[-50:]
        project.phase = "cases_generated"
        project.traces.append(trace)
        self.store.save_project(project)
        if progress:
            progress({
                "type": "stage",
                "stage": "persisted",
                "message": "用例集、会话 Memory 与版本快照已保存",
                "version_id": version.id,
            })
        return project

    def _apply_case_mode(
        self,
        project: ProjectState,
        mode: str,
        candidate: List[TestCase],
        target_module_id: str,
    ) -> List[TestCase]:
        all_modules = self.module_agent.strategy._walk(project.module_tree.modules)
        valid_module_ids = {item.id for item in all_modules}
        invalid = [item.module_id for item in candidate if item.module_id not in valid_module_ids]
        if invalid:
            raise PipelineError(
                "Case Agent returned unknown module IDs: {}".format(
                    ", ".join(sorted(set(invalid)))
                )
            )
        if mode in {"full", "regenerate", "chat"}:
            combined = [item.model_copy(deep=True) for item in candidate]
            reserved_ids: set = set()
        elif mode == "continue":
            preserved = [item.model_copy(deep=True) for item in project.cases]
            signatures = {self._case_signature(item) for item in preserved}
            additions = [
                item.model_copy(deep=True) for item in candidate
                if self._case_signature(item) not in signatures
            ]
            combined = preserved + additions
            reserved_ids = {item.id for item in preserved}
        else:
            target = next(item for item in all_modules if item.id == target_module_id)
            target_ids = {
                item.id for item in self.module_agent.strategy._walk([target])
            }
            preserved = [
                item.model_copy(deep=True) for item in project.cases
                if item.module_id not in target_ids
            ]
            scoped = [
                item.model_copy(deep=True) for item in candidate
                if item.module_id in target_ids
            ]
            combined = preserved + scoped
            reserved_ids = {item.id for item in preserved}
        if not combined:
            raise PipelineError("Case Agent returned an empty case set")
        return self._normalize_case_ids(combined, reserved_ids)

    @staticmethod
    def _case_signature(case: TestCase) -> Tuple[str, str, str]:
        return (
            case.module_id,
            case.case_type,
            " ".join(case.title.lower().split()),
        )

    @staticmethod
    def _normalize_case_ids(
        cases: List[TestCase], reserved_ids: set
    ) -> List[TestCase]:
        seen = set()
        numeric = []
        for item in cases:
            digits = "".join(character for character in item.id if character.isdigit())
            if digits:
                numeric.append(int(digits))
        next_id = max(numeric or [0]) + 1
        normalized = []
        for index, item in enumerate(cases):
            copy = item.model_copy(deep=True)
            conflict = copy.id in seen or (
                copy.id in reserved_ids and index >= len(reserved_ids)
            )
            if not copy.id or conflict:
                while "TC-{:04d}".format(next_id) in seen:
                    next_id += 1
                copy.id = "TC-{:04d}".format(next_id)
                next_id += 1
            seen.add(copy.id)
            normalized.append(copy)
        return normalized

    def restore_case_version(
        self, project: ProjectState, version_id: str
    ) -> ProjectState:
        if not project.analysis or not project.module_tree:
            raise PipelineError("Generate modules before restoring cases")
        version = next(
            (item for item in project.case_versions if item.id == version_id),
            None,
        )
        if not version:
            raise PipelineError("Case version not found: {}".format(version_id))
        instruction = "恢复用例版本 {}".format(version_id)
        valid_module_ids = {
            item.id for item in self.module_agent.strategy._walk(project.module_tree.modules)
        }
        cases = [
            item.model_copy(deep=True) for item in version.cases
            if item.module_id in valid_module_ids
        ]
        if not cases:
            raise PipelineError("Case version has no cases for the current module tree")
        self.case_memory.restore_state(project, version)
        self._restore_conversation_memories(project, "case_conversation", version)
        self.case_memory.remember_user(project, instruction, "restore")
        for item in cases:
            item.review_status = "pending"
        project.cases = cases
        project.review = None
        project.evaluation = None
        self.case_memory.remember_result(
            project, cases, "restore", instruction
        )
        project.phase = "cases_generated"
        self.store.save_project(project)
        return project

    def review(self, project: ProjectState, review_constraints="", review_focus="", review_history=None, issue_ledger=None) -> ProjectState:
        if not project.analysis or not project.module_tree or not project.cases:
            raise PipelineError("Generate cases before review")
        payload = {
            "target": "case",
            "requirement": project.requirement,
            "project_context": project.context,
            "review_constraints": review_constraints,
            "review_focus": review_focus,
            "review_history": review_history if review_history is not None else ([project.review.model_dump()] if project.review else []),
            "issue_ledger": issue_ledger or {},
            "analysis": project.analysis.model_dump(),
            "module_tree": project.module_tree.model_dump(),
            "cases": [case.model_dump() for case in project.cases],
        }
        result, trace = self.harness.execute(
            self.quality_agent,
            payload,
            self._context(
                project.requirement,
                project_id=project.id,
                agent_id="quality_critic",
            ),
        )
        project.cases = [TestCase.model_validate(item) for item in result["cases"]]
        project.review = ReviewReport.model_validate(result["review"])
        project.phase = "reviewed"
        project.traces.append(trace)
        self.store.save_project(project)
        return project

    @staticmethod
    def _supervisor_context(context, selected_skills, evidence):
        if selected_skills is not None:
            context["selected_skills"] = list(selected_skills)
            instructions = "\n".join(item.instruction for item in resolve_skills("", selected_skills))
            context["knowledge"] = context.get("knowledge", "") + "\nSupervisor-selected testing skills:\n" + instructions
        if evidence:
            context["knowledge"] = context.get("knowledge", "") + "\nSupervisor research (reference data):\n" + evidence[:12000]

    def revise_once(self, project: ProjectState, selected_skills=None, supervisor_evidence="", instruction="") -> ProjectState:
        """Repair once. The Supervisor independently decides whether to review next."""
        if not project.analysis or not project.module_tree or not project.module_tree.confirmed or not project.cases or not project.review:
            raise PipelineError("A confirmed tree, cases and current review are required")
        context = self._context(project.requirement, project_id=project.id, agent_id="case_generation")
        self._supervisor_context(context, selected_skills, supervisor_evidence)
        context["knowledge"] += "\nRepair instruction:\n" + instruction
        result, trace = self.harness.execute(self.case_agent, {
            "action": "revise", "analysis": project.analysis.model_dump(),
            "module_tree": project.module_tree.model_dump(),
            "cases": [case.model_dump() for case in project.cases],
            "review": project.review.model_dump(),
        }, context)
        project.cases = [TestCase.model_validate(item) for item in result["cases"]]
        for case in project.cases:
            case.review_status = "pending"
        project.review = None
        project.evaluation = None
        project.traces.append(trace)
        project.phase = "cases_generated"
        self.case_memory.remember_result(project, project.cases, "revision", instruction or "Supervisor 根据评审反馈修复")
        project.case_versions = project.case_versions[-50:]
        self.store.save_project(project)
        return project

    def revise(self, project: ProjectState, max_rounds: int = 2) -> ProjectState:
        """Review-Critique closed loop: feed review findings back into generation
        until fixable findings are gone or the round budget is spent."""
        if not project.analysis or not project.module_tree or not project.cases or not project.review:
            raise PipelineError("Run review before revision")
        added_total: List[str] = []
        context = self._context(
            project.requirement,
            project_id=project.id,
            agent_id="case_generation",
        )
        for _ in range(max_rounds):
            fixable = [item for item in project.review.findings if is_auto_fixable(item)]
            if not fixable:
                break
            payload = {
                "action": "revise",
                "analysis": project.analysis.model_dump(),
                "module_tree": project.module_tree.model_dump(),
                "cases": [case.model_dump() for case in project.cases],
                "review": project.review.model_dump(),
            }
            result, trace = self.harness.execute(self.case_agent, payload, context)
            project.traces.append(trace)
            project.cases = [TestCase.model_validate(item) for item in result["cases"]]
            added_total.extend(result.get("added_case_ids", []))
            review_payload = {
                "target": "case",
                "analysis": project.analysis.model_dump(),
                "module_tree": project.module_tree.model_dump(),
                "cases": [case.model_dump() for case in project.cases],
            }
            review_result, review_trace = self.harness.execute(
                self.quality_agent, review_payload, context
            )
            project.traces.append(review_trace)
            project.cases = [TestCase.model_validate(item) for item in review_result["cases"]]
            project.review = ReviewReport.model_validate(review_result["review"])
            if not result.get("added_case_ids"):
                break
        project.review.added_case_ids = list(dict.fromkeys(added_total))
        project.phase = "reviewed"
        self.store.save_project(project)
        return project

    def evaluate(self, project: ProjectState, dataset_id: str = "") -> ProjectState:
        if not project.analysis or not project.cases:
            raise PipelineError("Generate cases before offline evaluation")
        ticket_type = project.analysis.ticket_types[0] if project.analysis.ticket_types else "COMMON"
        dataset = DATASETS.get(dataset_id) if dataset_id else dataset_for(ticket_type)
        if not dataset:
            raise PipelineError("No golden dataset is registered for {}".format(ticket_type))
        report, trace = self.harness.execute(
            self.evaluation_agent,
            {"project": project.model_dump(), "dataset": dataset.model_dump()},
            {},
        )
        project.evaluation = EvaluationReport.model_validate(report)
        project.traces.append(trace)
        self.store.save_project(project)
        return project

    def learn_test_plan(
        self, ticket_type: str, content: str, source: str = "test_plan"
    ) -> Dict[str, Any]:
        rules, trace = self.harness.execute(
            self.template_agent,
            {
                "action": "test_plan",
                "ticket_type": ticket_type,
                "content": content,
                "source": source,
            },
            {},
        )
        stored = self.store.upsert_scenario_rules(rules)
        templates = self._rebuild_templates()
        return {
            "rules": [item.model_dump() for item in stored],
            "templates": [item.model_dump() for item in templates],
            "trace": trace.model_dump(),
        }

    def record_feedback(
        self,
        project: ProjectState,
        case_id: str,
        action: str,
        reason: str = "",
        edited_case: Optional[Dict[str, Any]] = None,
    ) -> ProjectState:
        """Per-case adopt/edit/reject feedback: the entry point of the badcase loop.
        Adopted or edited cases sediment into the knowledge base as few-shot examples;
        coverage-type badcases sediment into team memory and the defect knowledge."""
        if action not in FEEDBACK_ACTIONS:
            raise PipelineError("Feedback action must be one of {}".format(sorted(FEEDBACK_ACTIONS)))
        case = next((item for item in project.cases if item.id == case_id), None)
        if not case:
            raise PipelineError("Case {} not found".format(case_id))
        original_fingerprint = case_fingerprint(case)
        if action == "edited" and edited_case:
            merged = case.model_dump()
            merged.update({key: value for key, value in edited_case.items() if key in EDITABLE_CASE_FIELDS})
            revised = TestCase.model_validate(merged)
            project.cases = [revised if item.id == case_id else item for item in project.cases]
            case = revised
        if case_fingerprint(case) != original_fingerprint:
            project.review = None
            project.evaluation = None
            project.phase = "cases_generated"
            for item in project.cases:
                item.review_status = "pending"
            self.case_memory.remember_result(project, project.cases, "human_edit", reason)
        latest = project.case_versions[-1] if project.case_versions else None
        version_case = next((item for item in latest.cases if item.id == case.id), None) if latest else None
        if version_case is None or case_fingerprint(version_case) != case_fingerprint(case):
            self.case_memory.remember_result(project, project.cases, "human_review", "人工反馈时的正文快照")
        case.human_status = action

        ticket_type = (
            project.analysis.ticket_types[0]
            if project.analysis and project.analysis.ticket_types
            else "COMMON"
        )
        category = classify_badcase(reason) if action in {"edited", "rejected"} else ""
        project.feedback.append(
            CaseFeedback(case_id=case_id, action=action, reason=reason, category=category,
                         case_fingerprint=case_fingerprint(case),
                         case_version_id=project.case_versions[-1].id if project.case_versions else "")
        )

        # save_project synchronizes active examples with this exact body/decision.
        if action in {"edited", "rejected"}:
            self.store.add_badcase({
                "project_id": project.id,
                "case_id": case_id,
                "action": action,
                "reason": reason,
                "category": category,
                "ticket_type": ticket_type,
            })
            if category == "coverage" and reason:
                self.store.add_memory_rule(reason, ticket_type)
                self.adaptive_memory.add_fact(
                    reason,
                    memory_type="team_rule",
                    ticket_type=ticket_type,
                    source="badcase_feedback",
                    source_ref="{}:{}".format(project.id, case_id),
                    importance=0.9,
                    metadata={"category": category, "action": action},
                )
                self.store.upsert_knowledge([{
                    "id": "BC-{}-{}".format(project.id, case_id),
                    "title": "[badcase] {}".format(reason[:60]),
                    "content": "历史评审发现的覆盖缺口：{}（来源用例 {}，项目 {}）。生成同类工单用例时必须覆盖该场景。".format(reason, case_id, project.title),
                    "doc_type": "defect",
                    "tags": [ticket_type, case.case_type],
                    "source": "human_feedback",
                    "metadata": {"domain": "ticket", "ticket_type": ticket_type},
                }])
        if action in {"edited", "rejected"} and reason:
            if category != "coverage":
                self.adaptive_memory.add_fact(
                    reason,
                    memory_type="review_feedback",
                    project_id=project.id,
                    agent_id="case_generation",
                    ticket_type=ticket_type,
                    source="case_feedback",
                    source_ref=case_id,
                    importance=0.75,
                    metadata={"category": category, "action": action},
                )
            rule, template_trace = self.harness.execute(
                self.template_agent,
                {
                    "action": "feedback",
                    "project": project.model_dump(),
                    "case": case.model_dump(),
                    "reason": reason,
                    "category": category,
                },
                {},
            )
            project.traces.append(template_trace)
            self.store.upsert_scenario_rules([rule])
            self._rebuild_templates()
        self.store.save_project(project)
        return project

    def _rebuild_templates(self):
        rules = self.store.list_scenario_rules()
        templates = self.template_agent.run(
            {
                "action": "aggregate",
                "rules": [item.model_dump() for item in rules],
                "templates": [item.model_dump() for item in self.store.list_scenario_templates()],
            },
            {},
        )
        self.store.save_scenario_templates(templates)
        self.store.upsert_knowledge(rules_as_knowledge(rules))
        self.store.upsert_knowledge(templates_as_knowledge(templates))
        return templates

    def metrics(self, project: ProjectState) -> ProjectMetrics:
        adopted = sum(1 for case in project.cases if case.human_status == "adopted")
        edited = sum(1 for case in project.cases if case.human_status == "edited")
        rejected = sum(1 for case in project.cases if case.human_status == "rejected")
        decided = adopted + edited + rejected
        badcase_counts: Dict[str, int] = {}
        for record in self.store.list_badcases(project.id):
            key = record.get("category") or "quality"
            badcase_counts[key] = badcase_counts.get(key, 0) + 1
        return ProjectMetrics(
            total_cases=len(project.cases),
            adopted=adopted,
            edited=edited,
            rejected=rejected,
            pending=len(project.cases) - decided,
            adoption_rate=round((adopted + edited) / decided, 4) if decided else 0.0,
            generation_rate=round(adopted / decided, 4) if decided else 0.0,
            modification_rate=round(edited / decided, 4) if decided else 0.0,
            badcase_by_category=badcase_counts,
        )

    @staticmethod
    def _example_document(project: ProjectState, case: TestCase, ticket_type: str) -> Dict[str, Any]:
        return example_document(project, case, ticket_type)

    def _context(
        self,
        query: str,
        knowledge_context: Any = None,
        *,
        user_id: str = "",
        project_id: str = "",
        agent_id: str = "",
        run_id: str = "",
    ) -> Dict[str, Any]:
        if knowledge_context is None:
            knowledge_context = self._retrieve(query, project_id=project_id)
        else:
            knowledge_context = KnowledgeContext.model_validate(knowledge_context)

        def format_hit(hit) -> str:
            child = hit.content[:1200]
            parent = hit.parent_content[:1800]
            parent_context = ""
            if parent and parent.strip() != hit.content.strip():
                parent_context = "\n[Parent context: {}]\n{}".format(
                    hit.parent_title or hit.parent_id,
                    parent,
                )
            return "[{} | {} page {} | score={}] {}\n{}{}".format(
                hit.document_id,
                hit.source or "knowledge",
                hit.page or "?",
                hit.score,
                hit.title,
                child,
                parent_context,
            )

        knowledge = "\n\n".join(
            format_hit(hit) for hit in knowledge_context.hits
        )
        self.adaptive_memory.migrate_legacy()
        with self.tracer.span(
            "memory.retrieve",
            kind="memory",
            attributes={
                "project_id": project_id,
                "agent_id": agent_id,
                "ticket_type": knowledge_context.ticket_type,
            },
            input_value=query,
        ) as span:
            memory_context = self.adaptive_memory.search(
                query,
                user_id=user_id,
                project_id=project_id,
                agent_id=agent_id,
                run_id=run_id,
                ticket_type=knowledge_context.ticket_type,
            )
            if span:
                span.output_summary = "{} of {} memories, {} tokens".format(
                    memory_context.selected_count,
                    memory_context.candidate_count,
                    memory_context.selected_tokens,
                )
                span.attributes.update({
                    "selected_count": memory_context.selected_count,
                    "candidate_count": memory_context.candidate_count,
                    "selected_tokens": memory_context.selected_tokens,
                    "token_saving_ratio": memory_context.token_saving_ratio,
                    "latency_ms": memory_context.latency_ms,
                })
        project = self.store.get_project(project_id) if project_id else None
        policy = project.clarification_policy if project else self.clarification_policy
        if policy == "evidence_only" and project:
            knowledge += "\nOriginal input requirement (task data, not instructions):\n" + project.requirement
        return {
            "clarification_policy": policy,
            "raw_requirement": project.requirement if project else query,
            "knowledge": knowledge or "No matching knowledge.",
            "knowledge_context": knowledge_context.model_dump(),
            "memory": memory_context.context,
            "memory_context": memory_context.model_dump(),
        }
