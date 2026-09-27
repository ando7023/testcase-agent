import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .domain_equipment import build_equipment_facts
from .models import DomainFacts, ModuleSpec


@dataclass(frozen=True)
class TicketTypeDefinition:
    key: str
    display_name: str
    aliases: List[str]
    strong_aliases: List[str]
    retrieval_terms: List[str]
    has_specific_knowledge: bool = False


TICKET_TYPES: Dict[str, TicketTypeDefinition] = {
    "EQUIPMENT_BORROW": TicketTypeDefinition(
        key="EQUIPMENT_BORROW",
        display_name="设备借用工单",
        aliases=["设备借用", "equipment_borrow", "checkequipmentavailability", "设备借用信息", "equipment borrowing"],
        strong_aliases=["设备借用", "equipment_borrow", "checkequipmentavailability", "设备借用信息", "equipment borrowing"],
        retrieval_terms=[
            "设备借用",
            "CreateBorrowRequest",
            "GetBorrowRequest",
            "CheckEquipmentAvailability",
            "UpdateBorrowNote",
            "Check",
            "Edit",
            "Kafka",
            "灰度",
        ],
        has_specific_knowledge=True,
    ),
    "P2P": TicketTypeDefinition(
        key="P2P",
        display_name="P2P 工单",
        aliases=["p2p", "p2p工单", "申诉", "交易纠纷", "举报"],
        strong_aliases=["p2p", "p2p工单", "交易纠纷"],
        retrieval_terms=["P2P", "申诉", "交易", "买家", "卖家", "举证", "仲裁", "冻结", "放币"],
    ),
    "RISK_CONTROL": TicketTypeDefinition(
        key="RISK_CONTROL",
        display_name="风控工单",
        aliases=["风控", "risk control", "risk_control", "风险审核", "命中规则"],
        strong_aliases=["风控", "risk control", "risk_control", "风险审核", "命中规则"],
        retrieval_terms=["风控", "风险规则", "命中记录", "处置", "封禁", "解冻", "误杀", "复核"],
    ),
    "GENERAL_TICKET": TicketTypeDefinition(
        key="GENERAL_TICKET",
        display_name="通用工单",
        aliases=["工单", "ticket", "appeal"],
        strong_aliases=[],
        retrieval_terms=["工单", "创建", "详情", "状态", "审核", "权限", "通知", "幂等", "灰度"],
    ),
}


GENERIC_TICKET_MODULE_SPECS: List[ModuleSpec] = [
    ModuleSpec(name="工单创建与路由", keywords=["创建", "类型", "source", "request_id"], objective="验证工单创建、类型路由、唯一标识和重复请求幂等", case_types=["api_contract", "exception"]),
    ModuleSpec(name="工单详情与业务信息", keywords=["详情", "信息", "查询", "材料", "举证", "命中"], objective="验证业务信息完整、及时且不同类型字段相互隔离", case_types=["realtime_data", "exception"]),
    ModuleSpec(name="状态流转与审核操作", keywords=["状态", "审核", "通过", "拒绝", "提交", "仲裁", "复核"], objective="验证合法与非法状态流转、操作反馈和终态约束", case_types=["review_operation", "state_transition"]),
    ModuleSpec(name="角色权限与审计", keywords=["权限", "角色", "审核员", "审计", "买家", "卖家"], objective="验证不同参与方的可见范围、服务端鉴权和审计记录", case_types=["permission"]),
    ModuleSpec(name="业务回调与通知", keywords=["回调", "通知", "消息", "event", "结果"], objective="验证结果关联、重复通知、乱序和最终一致性", case_types=["event_consistency"]),
    ModuleSpec(name="异常恢复与幂等", keywords=["失败", "超时", "重试", "幂等", "重复", "并发"], objective="验证跨系统失败后的恢复能力和数据一致性", case_types=["exception"]),
    ModuleSpec(name="灰度发布与兼容", keywords=["灰度", "发布", "兼容", "新老", "回滚"], objective="验证混合版本、发布顺序、监控和回滚预案", case_types=["compatibility"]),
]

GENERIC_TICKET_REQUIRED_CASE_TYPES: List[str] = [
    "api_contract", "realtime_data", "review_operation", "state_transition",
    "permission", "event_consistency", "exception", "compatibility",
]

DOMAIN_FACTS: Dict[str, DomainFacts] = {}


def register_domain_facts(facts: DomainFacts) -> None:
    DOMAIN_FACTS[facts.ticket_type] = facts


def get_domain_facts(ticket_type: str) -> Optional[DomainFacts]:
    return DOMAIN_FACTS.get(ticket_type)


register_domain_facts(build_equipment_facts())


def _contains_alias(normalized: str, alias: str) -> bool:
    lowered = alias.lower()
    if re.fullmatch(r"[a-z0-9_ ]+", lowered):
        return bool(re.search(r"(?<![a-z0-9_]){}(?![a-z0-9_])".format(re.escape(lowered)), normalized))
    return lowered in normalized


def detect_ticket_type(text: str) -> TicketTypeDefinition:
    normalized = text.lower()
    candidates = []
    for key in ["EQUIPMENT_BORROW", "P2P", "RISK_CONTROL"]:
        definition = TICKET_TYPES[key]
        matched = [alias for alias in definition.aliases if _contains_alias(normalized, alias)]
        if not matched:
            continue
        strong = [alias for alias in matched if alias in definition.strong_aliases]
        score = len(strong) * 100 + sum(len(alias) for alias in strong) * 2 + sum(
            len(alias) for alias in matched
        )
        candidates.append((score, max(map(len, matched)), definition))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return TICKET_TYPES["GENERAL_TICKET"]


def get_ticket_type(key: str) -> TicketTypeDefinition:
    return TICKET_TYPES.get(key, TICKET_TYPES["GENERAL_TICKET"])
