import re
from typing import Any, Dict, List, Tuple


LEGACY_SCOPED_RULE = re.compile(r"^\[badcase沉淀\]\[([^]]+)\]\s*(.*)$")


class TeamMemory:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload

    @property
    def preferences(self) -> Dict[str, Any]:
        return self.payload.get("preferences", {})

    @property
    def learned_rules(self) -> List[str]:
        return [item for item in self.payload.get("learned_rules", []) if isinstance(item, str)]

    def _entries(self) -> List[Tuple[str, str]]:
        entries: List[Tuple[str, str]] = []
        for rule in self.learned_rules:
            match = LEGACY_SCOPED_RULE.match(rule)
            entries.append((match.group(2), match.group(1)) if match else (rule, "COMMON"))
        for item in self.payload.get("scoped_rules", []):
            if not isinstance(item, dict) or not str(item.get("rule", "")).strip():
                continue
            entries.append((str(item["rule"]), str(item.get("ticket_type") or "COMMON")))
        return entries

    def rules_for(self, ticket_type: str = "COMMON") -> List[str]:
        selected: List[str] = []
        for rule, scope in self._entries():
            if scope not in {"COMMON", ticket_type} or rule in selected:
                continue
            selected.append(rule)
        return selected

    def context(self, ticket_type: str = "COMMON") -> str:
        rules = self.rules_for(ticket_type)[-10:]
        return "\n".join("- " + rule for rule in rules) if rules else "No learned team rules yet."
