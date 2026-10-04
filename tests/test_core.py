import io
import json
import re
import sqlite3
import tempfile
import zipfile
import unittest
from pathlib import Path

from app.document_parser import DocumentParsingAgent, OCRSpanData, PageExtraction
from app.domain_registry import detect_ticket_type
from app.llm import OpenAICompatibleClient
from app.orchestrator import PipelineError, TestCaseOrchestrator
from app.observability import TraceManager
from app.store import JsonStore
from app.text_chunking import split_text_naturally, suggest_two_parts
from app.tooling import ReActRuntime, ToolRegistry


REQUIREMENT = """用户登录后可以领取活动优惠券。每位用户每个活动最多领取1张，库存不足时不能领取。
优惠券只能在有效期内使用，订单金额必须达到门槛。支付失败或订单取消后，优惠券应退回为可用状态。
管理员可以创建、发布和停止活动，普通用户不能访问管理功能。"""

from app.domain_equipment import EQUIPMENT_REQUIREMENT

P2P_REQUIREMENT = """P2P 交易纠纷工单由买家或卖家发起，双方需要在详情中提交举证材料。
审核员完成仲裁后通知交易系统执行放币或退款。重复提交不得创建多张工单，回调失败需要重试，
买卖双方只能查看自己的敏感材料，上线时需要支持灰度和历史工单兼容。"""

RISK_REQUIREMENT = """风控工单在用户命中风险规则后自动创建，详情展示命中记录和处置建议。
风控审核员可以执行封禁、解冻或转人工复核；普通客服只能查看摘要，不能修改处置结果。
相同风险事件重复上报必须幂等，处置通知失败后需要重试，并确保终态不会回退。"""


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.orchestrator = TestCaseOrchestrator(JsonStore(Path(self.temp.name) / "data"))
        self.orchestrator.llm.api_key = ""

    def tearDown(self):
        self.temp.cleanup()

    def test_full_demo_pipeline_is_traceable(self):
        project = self.orchestrator.store.create_project("优惠券领取与使用", REQUIREMENT)
        project = self.orchestrator.analyze(project)
        self.assertTrue(project.analysis.atomic_requirements)
        self.assertTrue(all(item.source_quote for item in project.analysis.atomic_requirements))
        project = self.orchestrator.plan_modules(project)
        self.assertFalse(project.module_tree.confirmed)
        project = self.orchestrator.confirm_modules(project, [item.model_dump() for item in project.module_tree.modules])
        project = self.orchestrator.generate_cases(project)
        self.assertTrue(project.cases)
        self.assertTrue(all(case.requirement_ids and case.source_evidence for case in project.cases))
        project = self.orchestrator.review(project)
        self.assertEqual(project.phase, "reviewed")
        self.assertEqual([trace.agent for trace in project.traces], [
            "requirement_understanding",
            "module_planning",
            "quality_critic",
            "case_generation",
            "quality_critic",
        ])
        requirement_trace = next(
            trace for trace in project.traces
            if trace.agent == "requirement_understanding"
        )
        case_trace = next(
            trace for trace in project.traces if trace.agent == "case_generation"
        )
        self.assertEqual(requirement_trace.react_steps, 3)
        self.assertEqual(case_trace.react_steps, 4)
        self.assertIn(
            "search_knowledge",
            [item["tool"] for item in requirement_trace.tool_calls],
        )
        self.assertIn(
            "get_test_skills",
            [item["tool"] for item in case_trace.tool_calls],
        )

    def test_mindmap_output_tool_preserves_modules_cases_and_steps(self):
        project = self.orchestrator.store.create_project("Mind map test", REQUIREMENT)
        project = self.orchestrator.analyze(project)
        project = self.orchestrator.plan_modules(project)

        module_map = self.orchestrator.mindmap(project, "modules")
        self.assertEqual(module_map.tool, "mindmap_conversion")
        self.assertEqual(module_map.stats["modules"], len(project.module_tree.modules))
        self.assertFalse([
            node for node in self._mindmap_nodes(module_map.root)
            if node.node_type == "case"
        ])

        project = self.orchestrator.confirm_modules(
            project, [item.model_dump() for item in project.module_tree.modules]
        )
        project = self.orchestrator.generate_cases(project)
        case_map = self.orchestrator.mindmap(project, "cases")
        nodes = list(self._mindmap_nodes(case_map.root))
        self.assertEqual(
            len([node for node in nodes if node.node_type == "case"]),
            len(project.cases),
        )
        self.assertEqual(
            len([node for node in nodes if node.node_type == "expected"]),
            sum(len(case.steps) for case in project.cases),
        )
        self.assertEqual(case_map.stats["orphan_cases"], 0)

        catalog = self.orchestrator.tool_catalog()
        self.assertEqual(
            catalog["output_tools_by_agent"]["case_generation"][0]["name"],
            "mindmap_conversion",
        )
        self.assertNotIn(
            "mindmap_conversion",
            [tool["name"] for tool in catalog["tools_by_agent"]["case_generation"]],
        )

    def test_xmind_export_is_valid_deterministic_archive(self):
        project = self.orchestrator.store.create_project("XMind export", REQUIREMENT)
        project = self.orchestrator.analyze(project)
        project = self.orchestrator.plan_modules(project)
        project = self.orchestrator.confirm_modules(
            project, [item.model_dump() for item in project.module_tree.modules]
        )
        project = self.orchestrator.generate_cases(project)

        first = self.orchestrator.export_xmind(project, "cases")
        second = self.orchestrator.export_xmind(project, "cases")
        self.assertEqual(first, second)

        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            self.assertEqual(
                archive.namelist(),
                ["content.json", "metadata.json", "manifest.json"],
            )
            content = json.loads(archive.read("content.json"))
            metadata = json.loads(archive.read("metadata.json"))
            manifest = json.loads(archive.read("manifest.json"))

        self.assertEqual(metadata["dataStructureVersion"], "2")
        self.assertEqual(
            set(manifest["file-entries"]),
            {"content.json", "metadata.json"},
        )
        root = content[0]["rootTopic"]
        self.assertEqual(root["title"], project.title)
        topics = list(self._xmind_topics(root))
        self.assertEqual(
            len([topic for topic in topics if topic["title"] in {
                case.title for case in project.cases
            }]),
            len(project.cases),
        )
        self.assertTrue(any("labels" in topic for topic in topics))
        self.assertTrue(any("notes" in topic for topic in topics))

        catalog = self.orchestrator.tool_catalog()
        self.assertEqual(
            catalog["output_tools_by_agent"]["case_generation"][1]["name"],
            "xmind_export",
        )

    @staticmethod
    def _xmind_topics(topic):
        yield topic
        for child in topic.get("children", {}).get("attached", []):
            yield from PipelineTest._xmind_topics(child)

    @staticmethod
    def _mindmap_nodes(node):
        yield node
        for child in node.children:
            yield from PipelineTest._mindmap_nodes(child)

    def test_case_generation_accepts_unconfirmed_nonempty_tree(self):
        project = self.orchestrator.store.create_project("门禁测试", REQUIREMENT)
        project = self.orchestrator.plan_modules(self.orchestrator.analyze(project))
        self.assertFalse(project.module_tree.confirmed)
        project = self.orchestrator.generate_cases(project)
        self.assertTrue(project.cases)
        self.assertFalse(project.module_tree.confirmed)
        project.module_tree.modules = []
        with self.assertRaisesRegex(PipelineError, "nonempty"):
            self.orchestrator.generate_cases(project)

    def test_knowledge_and_memory_are_available(self):
        self.orchestrator.store.add_knowledge(
            "优惠券库存规则", "领取成功必须原子扣减库存，重复请求不得重复领取。", "business_rule", ["优惠券", "库存"]
        )
        self.orchestrator.store.add_memory_rule("涉及支付失败时必须验证优惠券回退和重复通知幂等。")
        context = self.orchestrator._context("优惠券库存不足与支付失败")
        self.assertIn("优惠券库存规则", context["knowledge"])
        self.assertIn("支付失败", context["memory"])

    def test_equipment_pipeline_uses_rag_and_domain_skills(self):
        project = self.orchestrator.store.create_project("设备借用演示", EQUIPMENT_REQUIREMENT)
        project = self.orchestrator.analyze(project)
        self.assertEqual(len(self.orchestrator.store.list_knowledge()), 8)
        self.assertGreaterEqual(set(project.analysis.retrieved_evidence_ids), {
            "DEMO-GUIDE-CREATE", "DEMO-GUIDE-EVENTS", "DEMO-GUIDE-RELEASE"
        })
        self.assertEqual({item.name for item in project.analysis.interfaces}, {
            "CreateBorrowRequest", "GetBorrowRequest", "CheckEquipmentAvailability", "UpdateBorrowNote"
        })
        self.assertEqual(len(project.analysis.state_transitions), 3)
        self.assertEqual(project.analysis.events[0].correlation_key, "payload.borrow_id")
        project = self.orchestrator.plan_modules(project)
        self.assertGreaterEqual({item.name for item in project.module_tree.modules}, {
            "借用申请创建", "借用状态流转", "借用事件", "灰度兼容"
        })
        project = self.orchestrator.confirm_modules(project, [item.model_dump() for item in project.module_tree.modules])
        project = self.orchestrator.generate_cases(project)
        self.assertGreaterEqual({case.case_type for case in project.cases}, {
            "api_contract", "realtime_data", "review_operation", "state_transition",
            "permission", "event_consistency", "exception", "compatibility"
        })
        kafka_case = next(case for case in project.cases if case.case_type == "event_consistency")
        self.assertEqual(kafka_case.test_data["topic"], "demo.equipment.events")
        self.assertTrue(any("borrow_id" in step.expected for step in kafka_case.steps))
        project = self.orchestrator.review(project)
        self.assertGreaterEqual(project.review.score, 80)
        self.assertFalse([item for item in project.review.findings if item.severity == "high"])

    def test_equipment_retrieval_prefers_matching_contract(self):
        context = self.orchestrator._context("设备借用点击 Check 和 Edit 修改借用备注，失败展示 error.message")
        hit_ids = [item["document_id"] for item in context["knowledge_context"]["hits"][:3]]
        self.assertIn("DEMO-GUIDE-OPERATIONS", hit_ids)


    def test_p2p_ticket_isolated_from_equipment_knowledge(self):
        project = self.orchestrator.store.create_project("P2P 交易纠纷", P2P_REQUIREMENT)
        project = self.orchestrator.analyze(project)
        self.assertEqual(project.analysis.ticket_types, ["P2P"])
        self.assertEqual(project.analysis.retrieved_evidence_ids, ["COMMON-DEMO-CHECKLIST"])
        self.assertFalse(project.analysis.interfaces)
        self.assertTrue(any("未加载" in item for item in project.analysis.ambiguities))
        project = self.orchestrator.plan_modules(project)
        rendered = project.model_dump_json()
        for forbidden in ["CheckEquipmentAvailability", "GetBorrowRequest", "DeskAgent", "Supervisor", "demo.equipment.events"]:
            self.assertNotIn(forbidden, rendered)
        project = self.orchestrator.confirm_modules(project, [item.model_dump() for item in project.module_tree.modules])
        project = self.orchestrator.generate_cases(project)
        rendered = project.model_dump_json()
        for forbidden in ["CheckEquipmentAvailability", "GetBorrowRequest", "DeskAgent", "Supervisor", "demo.equipment.events"]:
            self.assertNotIn(forbidden, rendered)

    def _reviewed_equipment_project(self):
        project = self.orchestrator.store.create_project("设备借用 闭环测试", EQUIPMENT_REQUIREMENT)
        project = self.orchestrator.analyze(project)
        project = self.orchestrator.plan_modules(project)
        project = self.orchestrator.confirm_modules(project, [item.model_dump() for item in project.module_tree.modules])
        project = self.orchestrator.generate_cases(project)
        return self.orchestrator.review(project)

    def test_revision_loop_repairs_review_findings(self):
        project = self._reviewed_equipment_project()
        # 删除全部事件一致性用例，制造评审必然发现的覆盖缺口
        project.cases = [case for case in project.cases if case.case_type != "event_consistency"]
        project = self.orchestrator.review(project)
        self.assertTrue(any(
            item.category == "case_type" and item.detail == "event_consistency"
            for item in project.review.findings
        ))
        project = self.orchestrator.revise(project)
        self.assertTrue(project.review.added_case_ids)
        restored = [case for case in project.cases if case.case_type == "event_consistency"]
        self.assertTrue(restored)
        self.assertTrue(all(case.generated_by == "revision" for case in restored))
        self.assertEqual(restored[0].test_data["topic"], "demo.equipment.events")
        self.assertFalse(any(
            item.category == "case_type" and item.detail == "event_consistency"
            for item in project.review.findings
        ))
        self.assertGreaterEqual([trace.agent for trace in project.traces].count("case_generation"), 2)

    def test_unassigned_requirements_are_visible_not_stuffed(self):
        project = self._reviewed_equipment_project()
        names = [module.name for module in project.module_tree.modules]
        self.assertIn("未归类需求", names)
        self.assertTrue(any(item.category == "coverage_gap" for item in project.review.findings))
        # 未归类模块的需求不允许悄悄塞进第一个模块
        first = project.module_tree.modules[0]
        unassigned = next(module for module in project.module_tree.modules if module.name == "未归类需求")
        self.assertFalse(set(first.requirement_ids) & set(unassigned.requirement_ids))

    def test_feedback_metrics_and_knowledge_sedimentation(self):
        project = self._reviewed_equipment_project()
        adopted_case = project.cases[0]
        rejected_case = project.cases[1]
        project = self.orchestrator.record_feedback(project, adopted_case.id, "adopted")
        project = self.orchestrator.record_feedback(
            project, rejected_case.id, "rejected", reason="缺少并发重复提交场景"
        )
        metrics = self.orchestrator.metrics(project)
        self.assertEqual(metrics.adopted, 1)
        self.assertEqual(metrics.rejected, 1)
        self.assertEqual(metrics.generation_rate, 0.5)
        self.assertEqual(metrics.badcase_by_category.get("coverage"), 1)
        # 采纳用例沉淀为 case_example 知识，可作为后续 few-shot 素材
        knowledge = self.orchestrator.store.list_knowledge()
        examples = [doc for doc in knowledge if doc.doc_type == "case_example"]
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].metadata.get("ticket_type"), "EQUIPMENT_BORROW")
        # 覆盖类 badcase 自动写入缺陷知识与团队记忆
        defects = [doc for doc in knowledge if doc.doc_type == "defect"]
        self.assertEqual(len(defects), 1)
        memory = self.orchestrator.store.get_memory()
        self.assertTrue(any(
            item.get("ticket_type") == "EQUIPMENT_BORROW" and "并发重复提交" in item.get("rule", "")
            for item in memory["scoped_rules"]
        ))
        adaptive_records = self.orchestrator.store.list_memory_records()
        self.assertTrue(any(
            item.memory_type == "team_rule"
            and item.ticket_type == "EQUIPMENT_BORROW"
            and "并发重复提交" in item.content
            and item.source == "badcase_feedback"
            for item in adaptive_records
        ))
        # badcase 记录完整可追溯
        badcases = self.orchestrator.store.list_badcases(project.id)
        self.assertEqual(len(badcases), 1)
        self.assertEqual(badcases[0]["category"], "coverage")

    def test_edited_feedback_updates_case_and_counts_as_modification(self):
        project = self._reviewed_equipment_project()
        target = project.cases[0]
        project = self.orchestrator.record_feedback(
            project, target.id, "edited",
            reason="步骤描述不够具体",
            edited_case={"title": "人工修改后的标题", "priority": "P0"},
        )
        updated = next(case for case in project.cases if case.id == target.id)
        self.assertEqual(updated.title, "人工修改后的标题")
        self.assertEqual(updated.priority, "P0")
        self.assertEqual(updated.human_status, "edited")
        metrics = self.orchestrator.metrics(project)
        self.assertEqual(metrics.modification_rate, 1.0)
        self.assertEqual(metrics.adoption_rate, 1.0)
        self.assertEqual(metrics.generation_rate, 0.0)
        badcases = self.orchestrator.store.list_badcases(project.id)
        self.assertEqual(badcases[0]["category"], "quality")

    def test_adopted_example_isolated_by_ticket_type(self):
        project = self._reviewed_equipment_project()
        project = self.orchestrator.record_feedback(project, project.cases[0].id, "adopted")
        # P2P 检索不得召回 设备借用 的已采纳示例
        context = self.orchestrator._context(P2P_REQUIREMENT)
        hit_ids = [hit["document_id"] for hit in context["knowledge_context"]["hits"]]
        self.assertFalse(any(hit_id.startswith("EX-") for hit_id in hit_ids))

    def test_risk_ticket_isolated_from_equipment_knowledge(self):
        project = self.orchestrator.store.create_project("风控处置复核", RISK_REQUIREMENT)
        project = self.orchestrator.analyze(project)
        self.assertEqual(project.analysis.ticket_types, ["RISK_CONTROL"])
        self.assertEqual(project.analysis.retrieved_evidence_ids, ["COMMON-DEMO-CHECKLIST"])
        self.assertFalse(project.analysis.events)
        project = self.orchestrator.plan_modules(project)
        project = self.orchestrator.confirm_modules(project, [item.model_dump() for item in project.module_tree.modules])
        project = self.orchestrator.generate_cases(project)
        project = self.orchestrator.review(project)
        self.assertFalse([item for item in project.review.findings if item.category == "rag_traceability"])
        self.assertGreaterEqual(project.review.score, 80)
        rendered = project.model_dump_json()
        for forbidden in ["CheckEquipmentAvailability", "UpdateBorrowNote", "EQUIPMENT_BORROW", "demo.equipment.events"]:
            self.assertNotIn(forbidden, rendered)


    def test_manual_knowledge_is_retrieved_only_in_selected_scope(self):
        document = self.orchestrator.store.add_knowledge(
            "P2P 举证时限",
            "P2P 工单买卖双方必须在二十四小时内补充举证材料。",
            "business_rule",
            ["P2P", "举证"],
            "P2P",
        )
        p2p_context = self.orchestrator._context(P2P_REQUIREMENT + " 举证时限")
        p2p_ids = [hit["document_id"] for hit in p2p_context["knowledge_context"]["hits"]]
        self.assertIn(document.id, p2p_ids)
        risk_context = self.orchestrator._context(RISK_REQUIREMENT)
        risk_ids = [hit["document_id"] for hit in risk_context["knowledge_context"]["hits"]]
        self.assertNotIn(document.id, risk_ids)

    def test_mixed_risk_appeal_prefers_strong_risk_signal(self):
        self.assertEqual(
            detect_ticket_type("风控申诉工单：命中规则后申请复核").key,
            "RISK_CONTROL",
        )
        self.assertEqual(
            detect_ticket_type("用户举报触发风控工单").key,
            "RISK_CONTROL",
        )
        self.assertEqual(detect_ticket_type("P2P 交易纠纷申诉").key, "P2P")
        self.assertEqual(detect_ticket_type("testing ticket workflow").key, "GENERAL_TICKET")
        self.assertEqual(detect_ticket_type("CheckEquipmentAvailability API failed").key, "EQUIPMENT_BORROW")

    def test_team_memory_is_filtered_by_ticket_type(self):
        self.orchestrator.store.add_memory_rule("Check 失败必须展示 error.message", "EQUIPMENT_BORROW")
        self.orchestrator.store.add_memory_rule("重复提交必须验证幂等", "COMMON")
        equipment_context = self.orchestrator._context(EQUIPMENT_REQUIREMENT)
        risk_context = self.orchestrator._context(RISK_REQUIREMENT)
        self.assertIn("error.message", equipment_context["memory"])
        self.assertNotIn("error.message", risk_context["memory"])
        self.assertIn("重复提交", risk_context["memory"])

    def test_adaptive_memory_is_add_only_and_scope_aware(self):
        first = self.orchestrator.adaptive_memory.add_fact(
            "Split the approval flow into DeskAgent and Librarian modules.",
            memory_type="project_decision",
            project_id="PROJECT-A",
            agent_id="module_planning",
            ticket_type="EQUIPMENT_BORROW",
            source="test",
            importance=0.9,
        )
        duplicate = self.orchestrator.adaptive_memory.add_fact(
            "Split the approval flow into DeskAgent and Librarian modules.",
            memory_type="project_decision",
            project_id="PROJECT-A",
            agent_id="module_planning",
            ticket_type="EQUIPMENT_BORROW",
            source="test",
            importance=0.9,
        )
        self.assertEqual(first["added"], 1)
        self.assertEqual(duplicate["added"], 0)
        self.assertEqual(len(duplicate["duplicate_ids"]), 1)

        visible = self.orchestrator.adaptive_memory.search(
            "DeskAgent Librarian approval flow",
            project_id="PROJECT-A",
            agent_id="module_planning",
            ticket_type="EQUIPMENT_BORROW",
        )
        hidden = self.orchestrator.adaptive_memory.search(
            "DeskAgent Librarian approval flow",
            project_id="PROJECT-B",
            agent_id="module_planning",
            ticket_type="EQUIPMENT_BORROW",
        )
        self.assertEqual(visible.selected_count, 1)
        self.assertIn("DeskAgent and Librarian", visible.context)
        self.assertEqual(hidden.selected_count, 0)

    def test_adaptive_memory_enforces_token_budget_and_reports_savings(self):
        for index in range(8):
            self.orchestrator.adaptive_memory.add_fact(
                "Idempotency retry rule {}: verify duplicate callbacks, retry limits, "
                "terminal state consistency, audit logs, and observable error messages."
                .format(index),
                memory_type="team_rule",
                ticket_type="COMMON",
                source="test",
                importance=0.7,
            )
        result = self.orchestrator.adaptive_memory.search(
            "idempotency retry duplicate callback",
            ticket_type="COMMON",
            top_k=8,
            token_budget=80,
        )
        self.assertGreater(result.candidate_count, result.selected_count)
        self.assertLessEqual(result.selected_tokens, 80)
        self.assertGreater(result.token_saving_ratio, 0.0)
        self.assertEqual(result.retrieval_mode, "bm25+vector+rrf+scope_rerank")

    def test_llm_revision_cannot_drop_existing_cases(self):
        project = self._reviewed_equipment_project()
        project.cases = [case for case in project.cases if case.case_type != "event_consistency"]
        project = self.orchestrator.review(project)
        previous_ids = {case.id for case in project.cases}
        payload = {
            "action": "revise",
            "analysis": project.analysis.model_dump(),
            "module_tree": project.module_tree.model_dump(),
            "cases": [case.model_dump() for case in project.cases],
            "review": project.review.model_dump(),
        }
        self.orchestrator.llm.api_key = "test-key"
        self.orchestrator.llm.generate_json = lambda *args, **kwargs: {
            "cases": [project.cases[0].model_dump()]
        }
        result, trace = self.orchestrator.harness.execute(
            self.orchestrator.case_agent,
            payload,
            self.orchestrator._context(project.requirement),
        )
        revised_ids = {item["id"] for item in result["cases"]}
        self.assertTrue(previous_ids <= revised_ids)
        self.assertEqual(trace.status, "fallback_success")
        self.assertIn("case identity, module or step protection", trace.error)


    def test_hybrid_rag_exposes_stage_scores_and_trace(self):
        context = self.orchestrator.search_knowledge(
            "设备借用 Check Edit 修改借用备注失败展示 error.message"
        )
        self.assertEqual(
            context["retrieval_mode"], "bm25+vector+rrf+metadata_rerank"
        )
        self.assertEqual(context["trace"]["embedding_provider"], "hashing-v1")
        self.assertGreater(context["candidate_count"], 0)
        top = context["hits"][0]
        self.assertEqual(top["document_id"], "DEMO-GUIDE-OPERATIONS")
        self.assertGreater(top["lexical_score"], 0)
        self.assertGreater(top["fusion_score"], 0)
        self.assertIn("page-level evidence", top["reasons"])

    def test_rag_filters_expired_knowledge_before_retrieval(self):
        expired = self.orchestrator.store.add_knowledge(
            "设备借用 过期借用备注规则",
            "设备借用 Edit 曾经允许修改设备编号和借用备注。",
            "business_rule",
            ["设备借用", "Edit", "借用备注"],
            "EQUIPMENT_BORROW",
            expires_at="2000-01-01T00:00:00Z",
        )
        context = self.orchestrator.search_knowledge("设备借用 Edit 借用备注 设备编号")
        hit_ids = [hit["document_id"] for hit in context["hits"]]
        self.assertNotIn(expired.id, hit_ids)
        self.assertGreaterEqual(context["filtered_count"], 1)

    def test_document_ingestion_is_versioned_scoped_and_idempotent(self):
        content = """# P2P 工单测试方案
## 举证规则
买卖双方必须提交举证材料。
## 回调规则
交易回调必须验证重复通知幂等。
""".encode("utf-8")
        first = self.orchestrator.ingest_document(
            "P2P-测试方案.md", content, "P2P", version="2.0"
        )
        second = self.orchestrator.ingest_document(
            "P2P-测试方案.md", content, "P2P", version="2.0"
        )
        self.assertGreater(first["changed"], 0)
        self.assertEqual(second["changed"], 0)
        documents = first["knowledge_documents"]
        self.assertTrue(documents)
        self.assertTrue(all(item["version"] == "2.0" for item in documents))
        self.assertTrue(
            all(item["metadata"]["ticket_type"] == "P2P" for item in documents)
        )
        context = self.orchestrator.search_knowledge(
            "P2P 工单买卖双方举证材料要求"
        )
        self.assertTrue(
            any(hit["document_id"].startswith("CHK-") for hit in context["hits"])
        )

    def test_equipment_retrieval_golden_dataset_metrics(self):
        report = self.orchestrator.evaluate_retrieval("EQUIPMENT-RAG-V1", 5)
        self.assertEqual(report["recall_at_k"], 1.0)
        self.assertEqual(report["hit_rate"], 1.0)
        self.assertGreaterEqual(report["mrr"], 0.9)
        self.assertEqual(report["contamination_rate"], 0.0)
        self.assertEqual(report["stale_hit_rate"], 0.0)
    def test_document_parser_normalizes_sections_and_persists_source(self):
        content = """# 设备借用 技术方案
## 接口设计
CreateBorrowRequest 必须携带 request_id。
## 灰度方案
发布需要验证历史版本兼容。
""".encode("utf-8")
        result = self.orchestrator.parse_document("设备借用-技术方案.md", content)
        document = result["document"]
        self.assertEqual(document["document_type"], "technical_design")
        self.assertGreaterEqual(len(document["chunks"]), 2)
        self.assertIn("[第1页]", document["normalized_text"])
        project = self.orchestrator.store.create_project(
            "文档接入", document["normalized_text"], source_documents=[document]
        )
        loaded = self.orchestrator.store.get_project(project.id)
        self.assertEqual(loaded.source_documents[0].filename, "设备借用-技术方案.md")

    def test_document_parser_uses_llm_to_filter_and_compress_sections(self):
        class FakeCompressionLLM:
            enabled = True

            def __init__(self):
                self.calls = []

            def generate_json(self, system_prompt, user_prompt, schema=None):
                self.calls.append(user_prompt)
                section_ids = re.findall(r"\[SECTION (SEC-\d+)\]", user_prompt)
                answers = []
                for section_id in section_ids:
                    block = user_prompt.split("[SECTION {}]".format(section_id), 1)[1]
                    block = block.split("[SECTION ", 1)[0]
                    if "修改记录" in block:
                        answers.append({
                            "section_id": section_id,
                            "decision": "keep",
                            "reason": "模型错误地保留维护信息",
                            "markdown": "",
                        })
                    else:
                        answers.append({
                            "section_id": section_id,
                            "decision": "compress",
                            "reason": "保留可测试契约",
                            "markdown": "CreateBorrowRequest 必须携带 request_id。",
                        })
                return {"document_summary": "设备借用 工单接口改造。", "sections": answers}

        llm = FakeCompressionLLM()
        parser = DocumentParsingAgent(llm)
        document = parser.run({
            "filename": "设备借用-技术方案.md",
            "data": """# 修改记录
作者张三，2026-01-01 更新。
# 接口设计
CreateBorrowRequest 必须携带 request_id。
""".encode("utf-8"),
        }, {})

        self.assertTrue(llm.calls)
        self.assertTrue(document.compression_summary["llm_used"])
        self.assertEqual(document.excluded_sections[0]["title"], "修改记录")
        self.assertNotIn("作者张三", document.normalized_text)
        self.assertIn("CreateBorrowRequest", document.normalized_text)
        self.assertIn("request_id", document.normalized_text)
        self.assertIn("[第1页]", document.normalized_text)
        self.assertIn("作者张三", document.raw_text)

    def test_long_document_uses_hierarchical_map_reduce(self):
        class FakeHierarchicalLLM:
            enabled = True

            def __init__(self):
                self.map_calls = 0
                self.reduce_calls = 0

            def generate_json(self, system_prompt, user_prompt, schema=None):
                if "[SECTION " in user_prompt:
                    self.map_calls += 1
                    section_ids = re.findall(r"\[SECTION (SEC-\d+)\]", user_prompt)
                    return {
                        "document_summary": "长文档测试摘要。",
                        "sections": [{
                            "section_id": section_id,
                            "decision": "compress",
                            "reason": "压缩重复描述",
                            "markdown": "接口 Order_API_{} 超时 3000ms 后重试。".format(section_id),
                        } for section_id in section_ids],
                    }
                self.reduce_calls += 1
                anchors = list(dict.fromkeys(re.findall(
                    r"Order_API_SEC-\d+|3000ms", user_prompt
                )))
                return {"markdown": "## 压缩结果\n\n" + "，".join(anchors)}

        llm = FakeHierarchicalLLM()
        parser = DocumentParsingAgent(llm)
        parser.compressor.batch_chars = 700
        parser.compressor.target_chars = 350
        sections = []
        for index in range(1, 9):
            sections.append(
                "# 模块 {}\n接口 Order_API_{} 超时 3000ms 后重试。{}".format(
                    index, index, "重复背景。" * 90
                )
            )
        document = parser.run({
            "filename": "超长PRD.md",
            "data": "\n".join(sections).encode("utf-8"),
        }, {})

        self.assertGreater(llm.map_calls, 1)
        self.assertGreaterEqual(llm.reduce_calls, 1)
        self.assertGreater(len(document.raw_text), len(document.normalized_text))
        self.assertIn("3000ms", document.normalized_text)
        self.assertEqual(document.compression_summary["map_calls"], llm.map_calls)
        self.assertEqual(document.compression_summary["reduce_calls"], llm.reduce_calls)

    def test_project_document_parse_does_not_write_rag(self):
        before = len(self.orchestrator.store.list_knowledge())
        parsed = self.orchestrator.parse_document(
            "风控PRD.md",
            "# 处置流程\n审核员确认后执行封禁。".encode("utf-8"),
        )
        after = len(self.orchestrator.store.list_knowledge())

        self.assertEqual(before, after)
        self.assertTrue(parsed["document"]["compression_summary"])

        ingested = self.orchestrator.ingest_document(
            "风控规则.md",
            "# 规则\n命中高风险规则后创建工单。".encode("utf-8"),
            "RISK_CONTROL",
        )
        self.assertGreater(len(self.orchestrator.store.list_knowledge()), after)
        self.assertFalse(ingested["document"]["compression_summary"])
        self.assertEqual(ingested["trace"]["agent"], "document_extraction")
    def test_pdf_low_text_page_uses_ocr_and_keeps_page_evidence(self):
        class FakeOCRBackend:
            available = True
            name = "fake_ocr"

            @staticmethod
            def extract(image_path):
                return [
                    OCRSpanData(
                        text="扫描页接口规则：重复请求必须幂等",
                        bbox=[
                            [10.0, 20.0],
                            [300.0, 20.0],
                            [300.0, 50.0],
                            [10.0, 50.0],
                        ],
                        confidence=0.96,
                    )
                ]

        parser = DocumentParsingAgent(self.orchestrator.llm, FakeOCRBackend())
        parser.min_native_chars = 20
        original_native = parser._extract_native_pdf
        original_render = parser._render_pdf_pages
        try:
            parser._extract_native_pdf = lambda path: [
                "第一页原生文本内容足够完整，用于验证普通 PDF 页面保持原生提取。",
                "",
            ]
            parser._render_pdf_pages = lambda path, pages, output: {
                2: output / "page-2.png"
            }
            document = parser.run(
                {"filename": "扫描技术方案.pdf", "data": b"%PDF-fake"}, {}
            )
        finally:
            parser._extract_native_pdf = original_native
            parser._render_pdf_pages = original_render

        ocr_chunks = [chunk for chunk in document.chunks if chunk.page == 2]
        self.assertEqual(len(ocr_chunks), 1)
        self.assertEqual(ocr_chunks[0].extraction_method, "ocr")
        self.assertEqual(ocr_chunks[0].ocr_confidence, 0.96)
        self.assertEqual(ocr_chunks[0].ocr_spans[0].bbox[0], [10.0, 20.0])
        self.assertIn("[第2页]", document.normalized_text)
        self.assertEqual(document.extraction_summary["ocr_pages"], [2])
        self.assertEqual(document.extraction_summary["ocr_backend"], "fake_ocr")
    def test_module_workspace_supports_four_modes_and_version_restore(self):
        project = self.orchestrator.store.create_project(
            "模块协作测试", EQUIPMENT_REQUIREMENT
        )
        project = self.orchestrator.analyze(project)
        project = self.orchestrator.operate_modules(project, mode="full")
        first_version = project.module_versions[-1]
        original = first_version.module_tree.model_dump()
        original_by_id = {
            item.id: item.model_dump() for item in project.module_tree.modules
        }
        self.assertEqual(project.module_conversation[0].role, "user")
        self.assertEqual(project.module_conversation[1].role, "assistant")

        project = self.orchestrator.operate_modules(
            project,
            mode="continue",
            instruction="补充遗漏的兼容性关注点",
        )
        continued_by_id = {
            item.id: item.model_dump() for item in project.module_tree.modules
        }
        for module_id, payload in original_by_id.items():
            self.assertEqual(continued_by_id[module_id], payload)
        self.assertGreater(len(continued_by_id), len(original_by_id))

        target_id = project.module_tree.modules[0].id
        untouched_id = project.module_tree.modules[1].id
        untouched = next(
            item.model_dump() for item in project.module_tree.modules
            if item.id == untouched_id
        )
        project = self.orchestrator.operate_modules(
            project,
            mode="targeted",
            instruction="拆分 DeskAgent 与 Librarian 审核职责",
            target_module_id=target_id,
        )
        target = next(
            item for item in project.module_tree.modules if item.id == target_id
        )
        self.assertEqual(target.objective, "拆分 DeskAgent 与 Librarian 审核职责")
        self.assertEqual(
            next(
                item.model_dump() for item in project.module_tree.modules
                if item.id == untouched_id
            ),
            untouched,
        )

        project = self.orchestrator.operate_modules(
            project,
            mode="chat",
            instruction="增加越权操作风险说明",
            target_module_id=target_id,
        )
        self.assertEqual(
            next(item for item in project.module_tree.modules if item.id == target_id).objective,
            "增加越权操作风险说明",
        )
        self.assertEqual(project.module_versions[-1].mode, "chat")
        self.assertFalse(project.module_tree.confirmed)
        project_memories = self.orchestrator.store.list_memory_records()
        self.assertTrue(any(
            item.project_id == project.id
            and item.agent_id == "module_planning"
            and item.source == "module_conversation"
            for item in project_memories
        ))

        project = self.orchestrator.restore_module_version(
            project, first_version.id
        )
        restored = project.module_tree.model_dump()
        restored["confirmed"] = original["confirmed"]
        self.assertEqual(restored, original)
        self.assertEqual(project.module_versions[-1].mode, "restore")

    def test_module_conversation_memory_summarizes_old_turns(self):
        project = self.orchestrator.store.create_project(
            "模块记忆测试", EQUIPMENT_REQUIREMENT
        )
        project = self.orchestrator.analyze(project)
        project = self.orchestrator.operate_modules(project, mode="full")
        target_id = project.module_tree.modules[0].id
        for index in range(7):
            project = self.orchestrator.operate_modules(
                project,
                mode="chat",
                instruction="第 {} 轮调整模块目标".format(index + 1),
                target_module_id=target_id,
            )
        self.assertEqual(len(project.module_conversation), 16)
        self.assertTrue(project.module_memory_summary)
        context = self.orchestrator.module_memory.context(project)
        self.assertIn("Earlier conversation summary", context)
        self.assertIn("第 7 轮调整模块目标", context)

    def test_case_workspace_supports_modes_memory_and_version_restore(self):
        project = self.orchestrator.store.create_project(
            "Case collaboration test", EQUIPMENT_REQUIREMENT
        )
        project = self.orchestrator.analyze(project)
        project = self.orchestrator.plan_modules(project)
        project = self.orchestrator.confirm_modules(
            project,
            [item.model_dump() for item in project.module_tree.modules],
        )
        project = self.orchestrator.operate_cases(project, mode="full")
        first_version = project.case_versions[-1]
        original = [item.model_dump() for item in project.cases]
        original_by_id = {item.id: item.model_dump() for item in project.cases}
        self.assertEqual(len(project.case_conversation), 2)

        project = self.orchestrator.operate_cases(
            project,
            mode="continue",
            instruction="Add one missing boundary or failure scenario.",
        )
        self.assertGreater(len(project.cases), len(original))
        for case_id, payload in original_by_id.items():
            current = next(item for item in project.cases if item.id == case_id)
            self.assertEqual(current.model_dump(), payload)

        target_module_id = project.cases[0].module_id
        untouched = next(
            item.model_dump() for item in project.cases
            if item.module_id != target_module_id
        )
        untouched_id = untouched["id"]
        project = self.orchestrator.operate_cases(
            project,
            mode="targeted",
            instruction="Regenerate this module with executable assertions.",
            target_module_id=target_module_id,
        )
        self.assertEqual(
            next(item.model_dump() for item in project.cases if item.id == untouched_id),
            untouched,
        )

        project = self.orchestrator.review(project)
        self.assertIsNotNone(project.review)
        project = self.orchestrator.operate_cases(
            project,
            mode="chat",
            instruction="Strengthen duplicate callback and terminal-state assertions",
            target_module_id=target_module_id,
        )
        adjusted = [
            item for item in project.cases
            if item.module_id == target_module_id
            and item.generated_by == "conversation"
        ]
        self.assertTrue(adjusted)
        self.assertIsNone(project.review)
        self.assertEqual(project.case_versions[-1].mode, "chat")
        self.assertTrue(any(
            item.project_id == project.id
            and item.agent_id == "case_generation"
            and item.source == "case_conversation"
            for item in self.orchestrator.store.list_memory_records()
        ))

        project = self.orchestrator.restore_case_version(
            project, first_version.id
        )
        self.assertEqual(
            [item.model_dump() for item in project.cases],
            original,
        )
        self.assertEqual(project.case_versions[-1].mode, "restore")

    def test_case_workspace_websocket_streams_cases_and_persists_memory(self):
        from fastapi.testclient import TestClient
        import app.api as api

        old_store, old_orchestrator = api.store, api.orchestrator
        temp = tempfile.TemporaryDirectory()
        try:
            api.store = JsonStore(Path(temp.name) / "data")
            api.orchestrator = TestCaseOrchestrator(api.store)
            api.orchestrator.llm.api_key = ""
            project = api.store.create_project("Case WebSocket", EQUIPMENT_REQUIREMENT)
            project = api.orchestrator.analyze(project)
            project = api.orchestrator.plan_modules(project)
            api.orchestrator.confirm_modules(
                project,
                [item.model_dump() for item in project.module_tree.modules],
            )
            client = TestClient(api.app)
            events = []
            with client.websocket_connect(
                "/ws/projects/{}/cases".format(project.id)
            ) as socket:
                socket.send_json({
                    "mode": "full",
                    "instruction": "",
                    "target_module_id": "",
                })
                while True:
                    event = socket.receive_json()
                    events.append(event)
                    if event["type"] in {"complete", "error"}:
                        break
            self.assertEqual(events[-1]["type"], "complete")
            self.assertTrue(any(
                event.get("type") == "case" for event in events
            ))
            self.assertTrue(any(
                event.get("stage") == "persisted" for event in events
            ))
            persisted = api.store.get_project(project.id)
            self.assertTrue(persisted.cases)
            self.assertEqual(len(persisted.case_conversation), 2)
            self.assertEqual(len(persisted.case_versions), 1)
            self.assertTrue(events[-1].get("trace_id"))
            self.assertIn(events[-1]["trace_id"], persisted.trace_run_ids)
        finally:
            api.store, api.orchestrator = old_store, old_orchestrator
            temp.cleanup()

    def test_http_pipeline_persists_hierarchical_trace(self):
        from fastapi.testclient import TestClient
        import app.api as api

        old_store, old_orchestrator = api.store, api.orchestrator
        temp = tempfile.TemporaryDirectory()
        try:
            api.store = JsonStore(Path(temp.name) / "data")
            api.orchestrator = TestCaseOrchestrator(api.store)
            api.orchestrator.llm.api_key = ""
            client = TestClient(api.app)
            project = client.post("/api/projects", json={
                "title": "Trace test",
                "requirement": EQUIPMENT_REQUIREMENT,
                "context": "",
                "source_documents": [],
            }).json()
            response = client.post(
                "/api/projects/{}/analyze".format(project["id"])
            )
            self.assertEqual(response.status_code, 200)
            trace_id = response.headers.get("x-trace-id")
            self.assertTrue(trace_id)

            summaries = client.get(
                "/api/projects/{}/traces".format(project["id"])
            ).json()
            self.assertEqual(summaries[0]["trace_id"], trace_id)
            detail = client.get("/api/traces/{}".format(trace_id)).json()
            spans = detail["spans"]
            names = {span["name"] for span in spans}
            self.assertIn("rag.retrieve", names)
            self.assertIn("memory.retrieve", names)
            self.assertIn("agent.requirement_understanding", names)
            self.assertIn("tool.search_knowledge", names)
            self.assertIn("store.save_project", names)
            by_id = {span["id"]: span for span in spans}
            self.assertTrue(all(
                not span["parent_span_id"]
                or span["parent_span_id"] in by_id
                for span in spans
            ))
            persisted = api.store.get_project(project["id"])
            self.assertIn(trace_id, persisted.trace_run_ids)
        finally:
            api.store, api.orchestrator = old_store, old_orchestrator
            temp.cleanup()

    def test_llm_trace_records_usage_without_storing_secret(self):
        temp = tempfile.TemporaryDirectory()
        try:
            tracer = TraceManager(Path(temp.name) / "data")
            client = OpenAICompatibleClient(tracer)
            client.api_key = "super-secret-key"
            client.disabled = False
            client._post_json = lambda endpoint, body: {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            }
            with tracer.run("test.llm") as run:
                result = client.generate_json(
                    "Never reveal api_key=super-secret-key", "Return JSON"
                )
            self.assertTrue(result["ok"])
            detail = tracer.read(run.trace_id)
            span = next(item for item in detail.spans if item.name == "llm.chat")
            self.assertEqual(span.usage["total_tokens"], 15)
            serialized = detail.model_dump_json()
            self.assertNotIn("super-secret-key", serialized)
            self.assertIn("<redacted>", serialized)
        finally:
            temp.cleanup()
    def test_module_workspace_websocket_streams_progress_and_persists_memory(self):
        import tempfile
        from fastapi.testclient import TestClient
        import app.api as api

        old_store, old_orchestrator = api.store, api.orchestrator
        temp = tempfile.TemporaryDirectory()
        try:
            api.store = JsonStore(Path(temp.name) / "data")
            api.orchestrator = TestCaseOrchestrator(api.store)
            api.orchestrator.llm.api_key = ""
            client = TestClient(api.app)
            created = client.post("/api/projects", json={
                "title": "WebSocket module test",
                "requirement": "A user submits a ticket. DeskAgent reviews it before Librarian. Duplicate submissions must be idempotent.",
                "context": "",
                "source_documents": [],
            }).json()
            client.post("/api/projects/{}/analyze".format(created["id"]))
            events = []
            with client.websocket_connect(
                "/ws/projects/{}/modules".format(created["id"])
            ) as socket:
                socket.send_json({"mode": "full"})
                while True:
                    event = socket.receive_json()
                    events.append(event["type"])
                    if event["type"] in {"complete", "error"}:
                        final = event
                        break
            project = final.get("project", {})
            self.assertEqual(final["type"], "complete")
            self.assertEqual(events[:2], ["stage", "stage"])
            self.assertIn("stage", events)
            self.assertIn("complete", events)
            self.assertEqual(len(project["module_conversation"]), 2)
            self.assertEqual(len(project["module_versions"]), 1)
        finally:
            api.store, api.orchestrator = old_store, old_orchestrator
            temp.cleanup()

    def test_adaptive_memory_api_exposes_search_metrics(self):
        from fastapi.testclient import TestClient
        import app.api as api

        old_store, old_orchestrator = api.store, api.orchestrator
        temp = tempfile.TemporaryDirectory()
        try:
            api.store = JsonStore(Path(temp.name) / "data")
            api.orchestrator = TestCaseOrchestrator(api.store)
            api.orchestrator.llm.api_key = ""
            client = TestClient(api.app)
            created = client.post("/api/memory/rules", json={
                "rule": "Always verify duplicate callback idempotency.",
                "ticket_type": "COMMON",
                "agent_id": "case_generation",
            })
            self.assertEqual(created.status_code, 200)
            searched = client.post("/api/memory/search", json={
                "query": "duplicate callback idempotency",
                "ticket_type": "COMMON",
                "agent_id": "case_generation",
                "top_k": 5,
                "token_budget": 100,
            })
            self.assertEqual(searched.status_code, 200)
            payload = searched.json()
            self.assertEqual(payload["selected_count"], 1)
            self.assertIn("latency_ms", payload)
            self.assertIn("token_saving_ratio", payload)
            self.assertEqual(client.get("/api/memory").json()["stats"]["total"], 1)
        finally:
            api.store, api.orchestrator = old_store, old_orchestrator
            temp.cleanup()
    def test_quality_critic_runs_before_human_confirmation(self):
        project = self.orchestrator.store.create_project("设备借用 模块评审", EQUIPMENT_REQUIREMENT)
        project = self.orchestrator.plan_modules(self.orchestrator.analyze(project))
        self.assertIsNotNone(project.module_review)
        self.assertGreaterEqual(project.module_review.score, 0)
        agents = [trace.agent for trace in project.traces]
        self.assertIn("quality_critic", agents)
        self.assertFalse(project.module_tree.confirmed)

    def test_equipment_offline_evaluation_uses_golden_dataset(self):
        project = self._reviewed_equipment_project()
        project = self.orchestrator.evaluate(project)
        self.assertEqual(project.evaluation.dataset_id, "EQUIPMENT-BASELINE-V1")
        self.assertGreaterEqual(project.evaluation.score, 85)
        self.assertEqual(project.evaluation.metrics["interface_recall"], 1.0)
        self.assertEqual(project.evaluation.metrics["case_type_recall"], 1.0)
        self.assertIn("offline_evaluation", [trace.agent for trace in project.traces])

    def test_test_plan_rules_evolve_into_active_template_and_rag(self):
        content = "买卖双方必须提交举证材料。重复回调需要验证幂等。审核拒绝后不得执行放币。"
        first = self.orchestrator.learn_test_plan("P2P", content, "P2P 测试计划 v1")
        self.assertTrue(all(item["status"] == "candidate" for item in first["templates"]))
        result = self.orchestrator.learn_test_plan("P2P", content, "P2P 测试计划 v2")
        self.assertGreaterEqual(len(result["rules"]), 3)
        active = [item for item in result["templates"] if item["status"] == "active"]
        self.assertTrue(active)
        self.assertGreaterEqual(active[0]["support_count"], 2)
        knowledge = self.orchestrator.store.list_knowledge()
        self.assertTrue(any(item.doc_type == "scenario_template" for item in knowledge))
        context = self.orchestrator._context(P2P_REQUIREMENT + " 举证 幂等 放币")
        hit_types = [hit["doc_type"] for hit in context["knowledge_context"]["hits"]]
        self.assertIn("scenario_template", hit_types)


    def test_ingestion_builds_parent_child_chunks_and_expands_parent_context(self):
        content = """# P2P dispute
## Evidence policy
Buyer uploads receipt evidence.
Seller uploads shipment evidence.
""".encode("utf-8")
        result = self.orchestrator.ingest_document(
            "p2p-policy.md", content, "P2P", version="3.0"
        )
        parents = [
            item for item in result["knowledge_documents"]
            if item["chunk_level"] == "parent"
        ]
        children = [
            item for item in result["knowledge_documents"]
            if item["chunk_level"] == "child"
        ]
        self.assertTrue(parents)
        self.assertTrue(children)
        parent_ids = {item["id"] for item in parents}
        self.assertTrue(all(item["parent_id"] in parent_ids for item in children))
        self.assertTrue(all(item["source_id"] == result["source_id"] for item in children))
        self.assertTrue(all(item["chunking_version"] == "section-child-v1" for item in children))

        context = self.orchestrator.search_knowledge(
            "P2P Buyer receipt evidence shipment"
        )
        hit = next(
            item for item in context["hits"]
            if item["document_id"] in {child["id"] for child in children}
        )
        self.assertIn(hit["parent_id"], parent_ids)
        self.assertTrue(hit["parent_content"])

    def test_reingestion_invalidates_removed_chunks(self):
        first = self.orchestrator.ingest_document(
            "risk-policy.md",
            b"# Risk ticket\n## Decision\nlegacy_unique_signal blocks payout",
            "RISK_CONTROL",
        )
        old_child = next(
            item for item in first["knowledge_documents"]
            if item["chunk_level"] == "child"
        )
        second = self.orchestrator.ingest_document(
            "risk-policy.md",
            b"# Risk ticket\n## Decision\nnew_unique_signal requires manual review",
            "RISK_CONTROL",
        )
        self.assertGreaterEqual(second["invalidated"], 1)
        stored_old = self.orchestrator.store.get_knowledge(old_child["id"])
        self.assertEqual(stored_old.status, "inactive")
        self.assertEqual(
            stored_old.metadata["invalidation_reason"], "source_rechunked"
        )
        hits = self.orchestrator.search_knowledge(
            "RISK_CONTROL legacy_unique_signal payout"
        )["hits"]
        self.assertNotIn(old_child["id"], [item["document_id"] for item in hits])

    def test_manual_split_and_merge_preserve_lineage_and_clear_vectors(self):
        ingested = self.orchestrator.ingest_document(
            "p2p-callback.md",
            b"# P2P callback\n## Retry\ncallback_alpha retries three times and callback_beta is idempotent",
            "P2P",
        )
        child = next(
            item for item in ingested["knowledge_documents"]
            if item["chunk_level"] == "child"
        )
        self.orchestrator.search_knowledge("P2P callback_alpha callback_beta")
        with sqlite3.connect(
            str(self.orchestrator.store.knowledge_index_file)
        ) as connection:
            cached_before = connection.execute(
                "SELECT COUNT(*) FROM knowledge_vectors WHERE document_id = ?",
                (child["id"],),
            ).fetchone()[0]
        self.assertGreater(cached_before, 0)

        split = self.orchestrator.split_knowledge_chunk(
            child["id"],
            ["callback_alpha retries three times", "callback_beta is idempotent"],
        )
        split_docs = split["documents"]
        self.assertEqual(len(split_docs), 2)
        self.assertTrue(all(item["chunk_version"] == 2 for item in split_docs))
        self.assertTrue(all(item["supersedes"] == [child["id"]] for item in split_docs))
        self.assertEqual(
            self.orchestrator.store.get_knowledge(child["id"]).status, "inactive"
        )
        with sqlite3.connect(
            str(self.orchestrator.store.knowledge_index_file)
        ) as connection:
            cached_after = connection.execute(
                "SELECT COUNT(*) FROM knowledge_vectors WHERE document_id = ?",
                (child["id"],),
            ).fetchone()[0]
        self.assertEqual(cached_after, 0)

        merged = self.orchestrator.merge_knowledge_chunks(
            [item["id"] for item in split_docs], "P2P callback combined"
        )
        merged_doc = merged["documents"][0]
        self.assertEqual(merged_doc["chunk_version"], 3)
        self.assertEqual(
            set(merged_doc["supersedes"]), {item["id"] for item in split_docs}
        )
        self.assertTrue(
            all(
                self.orchestrator.store.get_knowledge(item["id"]).status
                == "inactive"
                for item in split_docs
            )
        )
        active = self.orchestrator.store.get_knowledge(merged_doc["id"])
        self.assertEqual(active.status, "active")
        self.assertIn("callback_alpha", active.content)
        self.assertIn("callback_beta", active.content)
    def test_legacy_knowledge_converts_to_parent_child_structure(self):
        legacy = self.orchestrator.store.add_knowledge(
            "P2P legacy arbitration rule",
            "legacy_convert_signal requires buyer and seller evidence",
            "business_rule",
            ["P2P", "evidence"],
            "P2P",
            source="manual_entry",
        )
        self.orchestrator.search_knowledge(
            "P2P legacy_convert_signal buyer seller evidence"
        )
        with sqlite3.connect(
            str(self.orchestrator.store.knowledge_index_file)
        ) as connection:
            cached_before = connection.execute(
                "SELECT COUNT(*) FROM knowledge_vectors WHERE document_id = ?",
                (legacy.id,),
            ).fetchone()[0]
        self.assertGreater(cached_before, 0)

        converted = self.orchestrator.convert_knowledge_chunk(legacy.id)
        parent = converted["parent"]
        child = converted["documents"][0]
        self.assertEqual(parent["chunk_level"], "parent")
        self.assertFalse(parent["metadata"]["indexable"])
        self.assertEqual(child["chunk_level"], "child")
        self.assertEqual(child["parent_id"], parent["id"])
        self.assertEqual(child["source_id"], parent["source_id"])
        self.assertEqual(child["chunk_version"], legacy.chunk_version + 1)
        self.assertEqual(child["supersedes"], [legacy.id])
        self.assertEqual(
            self.orchestrator.store.get_knowledge(legacy.id).status, "inactive"
        )

        with sqlite3.connect(
            str(self.orchestrator.store.knowledge_index_file)
        ) as connection:
            cached_after = connection.execute(
                "SELECT COUNT(*) FROM knowledge_vectors WHERE document_id = ?",
                (legacy.id,),
            ).fetchone()[0]
        self.assertEqual(cached_after, 0)

        hits = self.orchestrator.search_knowledge(
            "P2P legacy_convert_signal buyer seller evidence"
        )["hits"]
        converted_hit = next(
            item for item in hits if item["document_id"] == child["id"]
        )
        self.assertEqual(converted_hit["parent_id"], parent["id"])
        self.assertTrue(converted_hit["parent_content"])
        self.assertNotIn(legacy.id, [item["document_id"] for item in hits])
        with self.assertRaisesRegex(ValueError, "already has"):
            self.orchestrator.convert_knowledge_chunk(child["id"])
    def test_split_suggestion_prefers_complete_sentences(self):
        text = (
            "第一条规则要求创建工单后记录请求编号。"
            "第二条规则要求审核详情每次读取最新数据。"
            "第三条规则要求终态通知支持幂等重试。"
            "第四条规则要求灰度期间兼容旧系统。"
        )
        parts = suggest_two_parts(text)
        self.assertEqual(len(parts), 2)
        self.assertTrue(parts[0].endswith("。"))
        self.assertTrue(parts[1].endswith("。"))
        self.assertEqual("".join(parts), text)

    def test_document_chunking_uses_sentence_boundaries_before_hard_limit(self):
        sentence = "The callback must preserve request identity. "
        text = (sentence * 12).strip()
        chunks = split_text_naturally(text, max_chars=120)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(item) <= 120 for item in chunks))
        self.assertTrue(all(item.endswith(".") for item in chunks))

        parsed = DocumentParsingAgent._chunk_pages(
            [PageExtraction(text=text)], max_chars=120
        )
        self.assertEqual([item.content for item in parsed], chunks)

    def test_chunk_manager_returns_natural_split_preview(self):
        document = self.orchestrator.store.add_knowledge(
            "P2P sentence-aware split",
            "创建争议工单后记录请求号。买方提交付款证据。卖方提交发货证据。审核结果需要幂等通知。",
            "business_rule",
            ["P2P"],
            "P2P",
        )
        converted = self.orchestrator.convert_knowledge_chunk(document.id)
        child_id = converted["documents"][0]["id"]
        suggestion = self.orchestrator.suggest_knowledge_split(child_id)
        self.assertEqual(
            suggestion["strategy"], "paragraph_sentence_clause_v1"
        )
        self.assertEqual(len(suggestion["parts"]), 2)
        self.assertTrue(suggestion["parts"][0].endswith("。"))
        self.assertEqual(
            "".join(suggestion["parts"]),
            self.orchestrator.store.get_knowledge(child_id).content,
        )
    def test_react_runtime_dynamically_selects_registered_tool(self):
        class FakeReactLLM:
            enabled = True

            def __init__(self):
                self.actions = [
                    {
                        "type": "tool",
                        "tool": "lookup",
                        "arguments": {"query": "设备借用 state"},
                        "reason": "Need state evidence",
                    },
                    {
                        "type": "finish",
                        "summary": "State evidence collected.",
                    },
                ]

            def generate_json(self, *args, **kwargs):
                return self.actions.pop(0)

        registry = ToolRegistry()
        registry.register(
            "lookup",
            "Lookup evidence",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            lambda arguments: {"evidence": arguments["query"]},
            ["research_agent"],
        )
        runtime = ReActRuntime(FakeReactLLM(), registry, max_steps=4)
        result = runtime.run("research_agent", "research 设备借用", [])
        self.assertEqual(result["mode"], "react")
        self.assertEqual(result["summary"], "State evidence collected.")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["tool"], "lookup")
        self.assertEqual(result["tool_calls"][0]["status"], "success")

    def test_tool_registry_enforces_permissions_and_react_budget(self):
        class RepeatingLLM:
            enabled = True

            def generate_json(self, *args, **kwargs):
                return {
                    "type": "tool",
                    "tool": "lookup",
                    "arguments": {"query": "same"},
                    "reason": "repeat",
                }

        registry = ToolRegistry()
        registry.register(
            "lookup",
            "Lookup evidence",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            lambda arguments: arguments,
            ["allowed_agent"],
        )
        with self.assertRaisesRegex(ValueError, "cannot call"):
            registry.execute(
                "blocked_agent", "lookup", {"query": "forbidden"}
            )
        runtime = ReActRuntime(RepeatingLLM(), registry, max_steps=4)
        result = runtime.run("allowed_agent", "bounded loop", [])
        self.assertEqual(result["react_steps"], 4)
        self.assertEqual(len(result["tool_calls"]), 1)
    def test_llm_client_parses_plain_and_fenced_json(self):
        self.assertEqual(
            OpenAICompatibleClient._parse_json_content('{"status":"ok"}'),
            {"status": "ok"},
        )
        fence = chr(96) * 3
        content = fence + 'json\n{"status":"ok"}\n' + fence
        self.assertEqual(
            OpenAICompatibleClient._parse_json_content(content),
            {"status": "ok"},
        )

if __name__ == "__main__":
    unittest.main()
