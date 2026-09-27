"""Invented equipment-borrowing fixture; not a real service contract.

All endpoints, field names, events and rules below exist only for local demos.
Knowledge sections are synthetic sections, not excerpts/pages of a source PDF.
"""
from typing import Any, Dict, List

from .models import DomainFacts, EventContract, InterfaceContract, ModuleSpec, StateTransition

DEMO_SOURCE = "CaseForge synthetic equipment borrowing guide v1"
EQUIPMENT_REQUIREMENT = """虚构设备借用系统提供 EQUIPMENT_BORROW 工单。
登录用户调用 CreateBorrowRequest 提交 equipment_id、borrow_days、request_id，借用天数为1到7天；相同 request_id 返回同一 borrow_id。
GetBorrowRequest 返回申请详情与当前可借库存，库存查询失败时提示重试。
CheckEquipmentAvailability 只查询库存，不改变申请状态；UpdateBorrowNote 只允许申请人在 DRAFT 状态编辑借用备注。
申请从 DRAFT 提交到 SUBMITTED，由 Librarian 审批到 READY 或 DECLINED；申请人不能审批自己的申请。
终态通过 demo.equipment.events 通知，payload.borrow_id 关联申请，重复事件不能重复处理。
演示需覆盖权限、审计、超时恢复和灰度兼容。所有名称和规则仅用于本仓库合成测试。"""

_SECTIONS = [
    ("DEMO-GUIDE-FLOW", "借用申请状态", "workflow", "DRAFT 由申请人提交到 SUBMITTED；Librarian 可批准到 READY 或拒绝到 DECLINED。READY 与 DECLINED 是终态，不允许再次审批。"),
    ("DEMO-GUIDE-PERMISSIONS", "借用权限与审计", "business_rule", "申请人只能查看自己的申请；Librarian 可以审批其他人的申请；服务端拒绝自审。演示审计服务记录操作者、borrow_id、动作与时间。"),
    ("DEMO-GUIDE-CREATE", "CreateBorrowRequest 创建申请", "api_contract", "CreateBorrowRequest 要求 equipment_id、borrow_days、request_id；borrow_days 为1至7的整数，重复 request_id 返回同一 borrow_id，不重复预留库存。"),
    ("DEMO-GUIDE-DETAIL", "GetBorrowRequest 详情查询", "api_contract", "GetBorrowRequest 根据 borrow_id 返回 equipment_summary 和当前库存；详情实时查询，查询超时显示重试，不将旧库存显示为实时值。"),
    ("DEMO-GUIDE-OPERATIONS", "设备借用 Check Edit 操作", "api_contract", "设备借用 Check 调用 CheckEquipmentAvailability 查询库存且不改变申请状态；Edit 调用 UpdateBorrowNote，仅允许申请人在 DRAFT 编辑借用备注。失败返回 error.message，成功后可重新查询备注。"),
    ("DEMO-GUIDE-EVENTS", "借用事件通知", "event_contract", "Kafka 演示 topic demo.equipment.events 发布 event=equipment.borrow.updated；payload.borrow_id 关联申请，状态 READY 或 DECLINED。重复消息不得重复处理，旧消息不得覆盖终态。"),
    ("DEMO-GUIDE-RELEASE", "演示发布兼容", "release_rule", "设备借用示例的灰度发布要求新旧客户端兼容，关闭新功能后仍可读取旧申请；异常可回滚，监控申请失败率。"),
    ("COMMON-DEMO-CHECKLIST", "合成测试通用检查", "test_standard", "通用测试覆盖接口参数、边界值、权限、重复提交、数据库写入失败、消息重试和发布回滚；未知业务契约应请求澄清。"),
]


def build_equipment_knowledge() -> List[Dict[str, Any]]:
    return [dict(id=identifier, title=title, doc_type=kind, content=content,
                 source=DEMO_SOURCE, page=index + 1, section="synthetic-section-{}".format(index + 1),
                 tags=["合成示例", "设备借用", kind] + {"DEMO-GUIDE-CREATE": ["创建", "关键字段", "幂等", "request_id"], "DEMO-GUIDE-DETAIL": ["详情", "实时查询", "最新信息", "区域"]}.get(identifier, []), chunk_index=index,
                 metadata={"domain": "ticket", "ticket_type": "COMMON" if identifier.startswith("COMMON-") else "EQUIPMENT_BORROW",
                           "synthetic": True})
            for index, (identifier, title, kind, content) in enumerate(_SECTIONS)]


def build_equipment_facts() -> DomainFacts:
    transitions = [
        StateTransition(from_state="DRAFT", action="提交借用申请", to_state="SUBMITTED", actor="申请人"),
        StateTransition(from_state="SUBMITTED", action="批准借用", to_state="READY", actor="Librarian", terminal=True),
        StateTransition(from_state="SUBMITTED", action="拒绝借用", to_state="DECLINED", actor="Librarian", terminal=True),
    ]
    for transition in transitions:
        transition.evidence_ids = ["DEMO-GUIDE-FLOW"]
    return DomainFacts(
        ticket_type="EQUIPMENT_BORROW", extra_actors=["申请人", "Librarian", "演示审计服务"],
        interfaces=[
            InterfaceContract(name="CreateBorrowRequest", purpose="创建设备借用申请",
                              required_fields=["equipment_id", "borrow_days", "request_id"],
                              success_condition="返回唯一 borrow_id；重复 request_id 返回同一申请",
                              failure_condition="借用天数不在1至7或缺少参数时拒绝且不预留库存", evidence_ids=["DEMO-GUIDE-CREATE"]),
            InterfaceContract(name="GetBorrowRequest", purpose="实时查询借用详情",
                              required_fields=["borrow_id"], success_condition="返回 equipment_summary 与最新库存",
                              failure_condition="查询超时明确提示重试", evidence_ids=["DEMO-GUIDE-DETAIL"]),
            InterfaceContract(name="CheckEquipmentAvailability", purpose="设备借用 Check 库存查询",
                              required_fields=["equipment_id"], success_condition="返回当前库存且不改变申请状态",
                              failure_condition="失败返回 error.message", evidence_ids=["DEMO-GUIDE-OPERATIONS"]),
            InterfaceContract(name="UpdateBorrowNote", purpose="Edit 修改借用备注",
                              required_fields=["borrow_id", "note"], success_condition="申请人可在 DRAFT 更新备注并重新查询",
                              failure_condition="其他角色或非 DRAFT 状态拒绝并返回 error.message", evidence_ids=["DEMO-GUIDE-OPERATIONS"]),
        ],
        state_transitions=transitions,
        events=[EventContract(topic="demo.equipment.events", event="equipment.borrow.updated",
                              statuses=["READY", "DECLINED"], correlation_key="payload.borrow_id",
                              delivery_risks=["重复消费", "乱序", "重试耗尽"], evidence_ids=["DEMO-GUIDE-EVENTS"])],
        permissions=["申请人只访问自己的申请", "禁止审批自己的申请", "仅 DRAFT 可修改借用备注", "库存查询不推进状态", "审计记录操作与时间"],
        required_case_types=["api_contract", "realtime_data", "review_operation", "state_transition", "permission", "event_consistency", "exception", "compatibility"],
        module_specs=[
            ModuleSpec(name="借用申请创建", keywords=["CreateBorrowRequest", "request_id", "创建"], objective="验证借用天数与重复提交", case_types=["api_contract", "exception"]),
            ModuleSpec(name="借用详情查询", keywords=["GetBorrowRequest", "详情", "实时"], objective="验证最新库存与失败反馈", case_types=["realtime_data"]),
            ModuleSpec(name="库存查询与备注", keywords=["Check", "Edit", "备注"], objective="验证只读库存查询和备注修改", case_types=["review_operation"]),
            ModuleSpec(name="借用状态流转", keywords=["DRAFT", "SUBMITTED", "READY", "DECLINED", "审批"], objective="验证合法路径和终态保护", case_types=["state_transition"]),
            ModuleSpec(name="借用权限与审计", keywords=["权限", "审计", "申请人"], objective="验证所有权和禁止自审", case_types=["permission"]),
            ModuleSpec(name="借用事件", keywords=["Kafka", "事件", "demo.equipment.events"], objective="验证关联键与重复消费", case_types=["event_consistency"]),
            ModuleSpec(name="失败恢复", keywords=["失败", "超时", "恢复"], objective="验证失败不重复写入", case_types=["exception"]),
            ModuleSpec(name="灰度兼容", keywords=["灰度", "兼容", "回滚"], objective="验证演示新旧版本兼容", case_types=["compatibility"]),
        ],
    )
