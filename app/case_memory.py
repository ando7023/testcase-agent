import json
import uuid
from functools import wraps

from .conversation_summary import advance, render, restore
from typing import List

from .models import (
    CaseConversationMessage,
    CaseSetVersion,
    ProjectState,
    TestCase,
)


def traced_case_memory(name):
    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if not self.tracer:
                return method(self, *args, **kwargs)
            project = args[0] if args else None
            with self.tracer.span(
                "memory.case_conversation." + name,
                kind="memory",
                attributes={"project_id": getattr(project, "id", "")},
            ) as span:
                result = method(self, *args, **kwargs)
                if span:
                    span.output_summary = "case conversation " + name
                return result
        return wrapped
    return decorator


class CaseConversationMemory:
    """Project-scoped episodic and version memory for case collaboration."""

    def __init__(self, recent_limit: int = 12, tracer=None) -> None:
        self.recent_limit = max(4, recent_limit)
        self.tracer = tracer

    @traced_case_memory("write_user")
    def remember_user(
        self,
        project: ProjectState,
        content: str,
        mode: str,
        target_module_id: str = "",
    ) -> CaseConversationMessage:
        message = CaseConversationMessage(
            id="CMSG-" + uuid.uuid4().hex[:10],
            role="user",
            content=content.strip(),
            mode=mode,
            target_module_id=target_module_id,
        )
        project.case_conversation.append(message)
        self._refresh_summary(project)
        return message

    @traced_case_memory("write_result")
    def remember_result(
        self,
        project: ProjectState,
        cases: List[TestCase],
        mode: str,
        instruction: str,
        target_module_id: str = "",
    ) -> CaseSetVersion:
        version = CaseSetVersion(
            id="CV-" + uuid.uuid4().hex[:10],
            mode=mode,
            instruction=instruction.strip(),
            target_module_id=target_module_id,
            cases=[item.model_copy(deep=True) for item in cases],
        )
        project.case_versions.append(version)
        module_count = len({item.module_id for item in cases})
        project.case_conversation.append(
            CaseConversationMessage(
                id="CMSG-" + uuid.uuid4().hex[:10],
                role="assistant",
                content="已完成{}，当前共 {} 条用例，覆盖 {} 个模块。".format(
                    self.mode_label(mode), len(cases), module_count
                ),
                mode=mode,
                target_module_id=target_module_id,
                version_id=version.id,
            )
        )
        self._refresh_summary(project)
        version.memory_state = project.case_decision_state.model_copy(deep=True)
        return version

    @traced_case_memory("read")
    def context(self, project: ProjectState) -> str:
        self._refresh_summary(project)
        recent = project.case_conversation[project.case_decision_state.recent_start:][-self.recent_limit :]
        payload = [
            {
                "role": item.role,
                "content": item.content,
                "mode": item.mode,
                "target_module_id": item.target_module_id,
                "version_id": item.version_id,
            }
            for item in recent
        ]
        return "Earlier case conversation summary:\n{}\nThe active decision ledger above supersedes conflicting historical turns. The current user instruction takes precedence.\nRecent turns:\n{}".format(
            project.case_memory_summary or "None.",
            json.dumps(payload, ensure_ascii=False),
        )

    def _refresh_summary(self, project: ProjectState) -> None:
        project.case_decision_state = advance(
            project.case_decision_state, project.case_conversation
        )
        project.case_memory_summary = render(project.case_decision_state)

    def restore_state(self, project, version) -> None:
        project.case_decision_state = restore(version, project.case_conversation)
        self._refresh_summary(project)

    @staticmethod
    def mode_label(mode: str) -> str:
        return {
            "full": "全量生成",
            "regenerate": "重新生成",
            "continue": "继续生成",
            "targeted": "指定模块生成",
            "chat": "对话式修改",
            "restore": "版本恢复",
            "human_edit": "人工修改",
            "human_review": "人工验收快照",
        }.get(mode, mode)
