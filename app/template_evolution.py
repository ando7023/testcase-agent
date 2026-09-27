import hashlib
import re
from collections import Counter
from typing import Any, Dict, List

from .knowledge import tokenize
from .models import ProjectState, ScenarioRule, ScenarioTemplate, TestCase, utc_now_iso


RULE_WORDS = ["必须", "需要", "不得", "不能", "应当", "应该", "缺少", "遗漏", "验证"]
STOP_TERMS = {
    "测试",
    "用例",
    "工单",
    "场景",
    "需要",
    "必须",
    "生成",
    "验证",
}


class TemplateEvolutionAgent:
    name = "template_evolution"

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Any:
        action = payload.get("action")
        if action == "feedback":
            return self._from_feedback(payload)
        if action == "test_plan":
            return self._from_test_plan(payload)
        if action == "aggregate":
            return self._aggregate(payload)
        raise ValueError("Unknown template evolution action: {}".format(action))

    @staticmethod
    def _from_feedback(payload: Dict[str, Any]) -> ScenarioRule:
        project = ProjectState.model_validate(payload["project"])
        case = TestCase.model_validate(payload["case"])
        ticket_type = (
            project.analysis.ticket_types[0]
            if project.analysis and project.analysis.ticket_types
            else "COMMON"
        )
        reason = str(payload.get("reason", "")).strip()
        rule = reason or "{}场景应覆盖：{}".format(case.case_type, case.title)
        return ScenarioRule(
            id=TemplateEvolutionAgent._rule_id(ticket_type, case.case_type, rule),
            ticket_type=ticket_type,
            scene=case.title,
            rule=rule,
            case_type=case.case_type,
            source="feedback:{}:{}".format(project.id, case.id),
        )

    @staticmethod
    def _from_test_plan(payload: Dict[str, Any]) -> List[ScenarioRule]:
        ticket_type = str(payload.get("ticket_type") or "COMMON")
        source = str(payload.get("source") or "test_plan")
        content = str(payload.get("content") or "")
        sentences = [
            item.strip()
            for item in re.split(r"(?:\r?\n)+|[。；;!?！？]", content)
            if len(item.strip()) >= 6
        ]
        candidates = [
            sentence for sentence in sentences if any(word in sentence for word in RULE_WORDS)
        ]
        return [
            ScenarioRule(
                id=TemplateEvolutionAgent._rule_id(ticket_type, "functional", sentence),
                ticket_type=ticket_type,
                scene=sentence[:40],
                rule=sentence,
                case_type="functional",
                source=source,
            )
            for sentence in candidates[:50]
        ]

    @staticmethod
    def _aggregate(payload: Dict[str, Any]) -> List[ScenarioTemplate]:
        rules = [ScenarioRule.model_validate(item) for item in payload.get("rules", [])]
        previous = {
            item.id: item
            for item in [
                ScenarioTemplate.model_validate(value)
                for value in payload.get("templates", [])
            ]
        }
        groups: Dict[str, List[ScenarioRule]] = {}
        for rule in rules:
            key = "{}::{}".format(rule.ticket_type, rule.case_type)
            groups.setdefault(key, []).append(rule)

        templates = []
        for key, items in groups.items():
            ticket_type, case_type = key.split("::", 1)
            unique_rules = list(dict.fromkeys(item.rule for item in items))
            support_sources = {
                source
                for item in items
                for source in (item.sources or [item.source])
            }
            support_count = len(support_sources)
            target_status = "active" if support_count >= 2 else "candidate"
            template_id = "TPL-{}-{}".format(
                re.sub(r"[^A-Za-z0-9_]+", "-", ticket_type).strip("-"),
                re.sub(r"[^A-Za-z0-9_]+", "-", case_type).strip("-"),
            )
            existing = previous.get(template_id)
            changed = (
                not existing
                or existing.rules != unique_rules
                or existing.support_count != support_count
                or existing.status != target_status
            )
            templates.append(
                ScenarioTemplate(
                    id=template_id,
                    ticket_type=ticket_type,
                    name="{} · {} 通用场景模板".format(ticket_type, case_type),
                    rules=unique_rules,
                    common_terms=TemplateEvolutionAgent._common_terms(unique_rules),
                    support_count=support_count,
                    version=(existing.version + 1 if existing and changed else existing.version)
                    if existing
                    else 1,
                    status=target_status,
                    updated_at=utc_now_iso() if changed else existing.updated_at,
                )
            )
        return templates

    @staticmethod
    def _common_terms(rules: List[str]) -> List[str]:
        counts = Counter()
        for rule in rules:
            terms = {
                token
                for token in tokenize(rule)
                if len(token) >= 2 and token not in STOP_TERMS
            }
            counts.update(terms)
        threshold = 2 if len(rules) >= 2 else 1
        return [
            term
            for term, count in counts.most_common(12)
            if count >= threshold
        ][:8]

    @staticmethod
    def _rule_id(ticket_type: str, case_type: str, rule: str) -> str:
        digest = hashlib.sha1(
            "{}|{}|{}".format(ticket_type, case_type, rule).encode("utf-8")
        ).hexdigest()[:12]
        return "RULE-" + digest


def rules_as_knowledge(rules: List[ScenarioRule]) -> List[Dict[str, Any]]:
    return [
        {
            "id": rule.id,
            "title": "[单场景规则] {}".format(rule.scene[:60]),
            "content": rule.rule,
            "doc_type": "scenario_rule",
            "tags": [rule.ticket_type, rule.case_type, "场景规则"],
            "source": rule.source,
            "metadata": {"domain": "ticket", "ticket_type": rule.ticket_type},
        }
        for rule in rules
    ]


def templates_as_knowledge(
    templates: List[ScenarioTemplate],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": template.id,
            "title": "[自进化模板 v{}] {}".format(template.version, template.name),
            "content": "\n".join(
                ["适用关键词：{}".format("、".join(template.common_terms))]
                + ["- " + rule for rule in template.rules]
            ),
            "doc_type": "scenario_template",
            "tags": [template.ticket_type, "通用模板"] + template.common_terms,
            "source": "template_evolution",
            "metadata": {
                "domain": "ticket",
                "ticket_type": template.ticket_type,
                "version": template.version,
                "support_count": template.support_count,
            },
        }
        for template in templates
        if template.status == "active"
    ]
