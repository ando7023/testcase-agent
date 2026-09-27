from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TestSkill:
    name: str
    triggers: List[str]
    instruction: str
    case_types: List[str]


SKILLS: Dict[str, TestSkill] = {
    "happy_path": TestSkill(
        "happy_path",
        ["新增", "创建", "提交", "保存", "查询", "登录", "支付"],
        "Cover the shortest valid business path and assert persisted state.",
        ["functional"],
    ),
    "boundary": TestSkill(
        "boundary",
        ["数量", "金额", "长度", "次数", "时间", "范围", "上限", "下限"],
        "Test minimum, maximum, just-inside, and just-outside values.",
        ["boundary"],
    ),
    "exception_recovery": TestSkill(
        "exception_recovery",
        ["失败", "超时", "重试", "异常", "网络", "回滚"],
        "Inject a dependency failure and verify recovery, idempotency, and user feedback.",
        ["exception"],
    ),
    "permission": TestSkill(
        "permission",
        ["角色", "权限", "管理员", "用户", "登录", "授权"],
        "Cover allowed, denied, expired-session, and cross-tenant access.",
        ["permission"],
    ),
    "state_transition": TestSkill(
        "state_transition",
        ["状态", "审核", "取消", "完成", "关闭", "发布"],
        "Validate legal and illegal transitions, concurrency, and repeated operations.",
        ["state_transition"],
    ),
    "ticket_creation_contract": TestSkill(
        "ticket_creation_contract",
        ["工单", "CreateBorrowRequest", "borrow_id", "request_id", "case_group"],
        "Validate required fields, enum routing, correlation IDs, duplicate requests, and response mapping.",
        ["api_contract"],
    ),
    "realtime_detail": TestSkill(
        "realtime_detail",
        ["实时查询", "详情", "GetBorrowRequest", "biz_equipment_info"],
        "Open details repeatedly and verify fresh data, field visibility, timeout behavior, and stale-data avoidance.",
        ["realtime_data"],
    ),
    "reviewer_operation": TestSkill(
        "reviewer_operation",
        ["审核员", "Check", "Edit", "申请人", "Librarian"],
        "Validate operation visibility by current reviewer, editable-field allowlist, and audit ownership.",
        ["permission", "review_operation"],
    ),
    "event_consistency": TestSkill(
        "event_consistency",
        ["Kafka", "topic", "event", "READY", "DECLINED", "消息"],
        "Validate event filtering, correlation, terminal status mapping, duplicate delivery, ordering, retry, and eventual consistency.",
        ["event_consistency"],
    ),
    "release_compatibility": TestSkill(
        "release_compatibility",
        ["灰度", "发布", "兼容", "新老系统", "历史APP", "回滚"],
        "Validate mixed-version behavior, release order, gray rollout, rollback, monitoring, and historical clients.",
        ["compatibility"],
    ),
}


def select_skills(text: str) -> List[TestSkill]:
    selected = []
    for skill in SKILLS.values():
        if any(trigger in text for trigger in skill.triggers):
            selected.append(skill)
    if not selected:
        selected = [SKILLS["happy_path"], SKILLS["boundary"], SKILLS["exception_recovery"]]
    return selected


def resolve_skills(text: str, names=None) -> List[TestSkill]:
    """None preserves legacy selection; an explicit list is authoritative."""
    if names is None:
        return select_skills(text)
    unknown = set(names) - SKILLS.keys()
    if unknown:
        raise ValueError("Unknown skills: {}".format(sorted(unknown)))
    return [SKILLS[name] for name in dict.fromkeys(names)]
