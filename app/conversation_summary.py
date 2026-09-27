"""Incremental, evidence-preserving decisions shared by both workspaces.

Unkeyed instructions are retained verbatim: we never guess that two business
rules contradict each other. Explicit `decision[key]: value` / `revoke[key]`
(also 决定/约束/问题/解决/撤销) support deterministic revisions without an LLM.
"""
import hashlib
import re

from .models import ConversationDecision, ConversationDecisionState


KEYED = re.compile(
    r"^(decision|constraint|question|resolve|revoke|决定|约束|问题|解决|撤销)"
    r"\[([^\]]+)\]\s*[:：]?\s*(.*)$", re.I | re.S,
)
CONSTRAINT = re.compile(r"\b(must|always|never|retain|constraint)\b|必须|不得|始终|确保|保留", re.I)


def advance(state, messages):
    # Old projects have no structured state; rebuild from their original log.
    if state.processed_count > len(messages):
        state = ConversationDecisionState()
    for message in messages[state.processed_count:]:
        if message.role == "assistant":
            for entry in state.entries:
                if not entry.source_version_id:
                    entry.source_version_id = message.version_id
            continue
        if message.mode not in {"chat", "continue", "targeted"}:
            continue
        content = message.content.strip()
        if not content or content == "人工编辑并确认当前模块树":
            continue
        match = KEYED.match(content)
        kind = "constraint" if CONSTRAINT.search(content) else "decision"
        status = "active"
        if match:
            action, key, value = match.groups()
            action = action.lower()
            if action in {"revoke", "撤销", "resolve", "解决"}:
                status = "revoked" if action in {"revoke", "撤销"} else "resolved"
                for entry in state.entries:
                    if entry.key == key and entry.status == "active":
                        entry.status = status
                content = value or content
            else:
                content = value.strip()
                if not content:
                    continue
                kind = ("constraint" if action in {"constraint", "约束"}
                        else "question" if action in {"question", "问题"} else "decision")
                for entry in state.entries:
                    if entry.key == key and entry.status == "active":
                        entry.status = "superseded"
        else:
            key = hashlib.sha256((message.target_module_id + "|" + content).encode()).hexdigest()[:16]
            if any(e.key == key and e.status == "active" for e in state.entries):
                continue
            if content.endswith(("?", "？")):
                kind = "question"
        state.entries.append(ConversationDecision(
            key=key, content=content, kind=kind, status=status,
            source_message_id=message.id, target_module_id=message.target_module_id,
            source_version_id=message.version_id,
        ))
    state.processed_count = len(messages)
    return state


def render(state, max_chars=10000):
    active = [entry for entry in state.entries if entry.status == "active"]
    active.sort(key=lambda entry: entry.kind != "constraint")
    lines = []
    used = 0
    omitted = 0
    for entry in active:
        line = "[{} key={} message={} version={} target={}] {}".format(
            entry.kind, entry.key, entry.source_message_id, entry.source_version_id,
            entry.target_module_id, entry.content,
        )
        if used + len(line) > max_chars:
            omitted += 1
            continue
        lines.append(line)
        used += len(line) + 1
    if omitted:
        lines.append("{} additional active decisions are retained in project memory; context budget exceeded.".format(omitted))
    return "\n".join(lines)


def restore(version, messages):
    if version.memory_state is not None:
        state = version.memory_state.model_copy(deep=True)
    else:
        # Compatibility with snapshots produced before structured summaries.
        end = next((i + 1 for i, message in enumerate(messages)
                    if message.version_id == version.id), 0)
        state = advance(ConversationDecisionState(), messages[:end])
    state.processed_count = len(messages)
    state.recent_start = len(messages)
    return state
