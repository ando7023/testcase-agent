import json
import uuid
from functools import wraps

from .conversation_summary import advance, render, restore
from typing import List

from .models import (
    ModuleConversationMessage,
    ModuleTree,
    ModuleTreeVersion,
    ProjectState,
)


def traced_module_memory(name):
    def decorator(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if not self.tracer:
                return method(self, *args, **kwargs)
            project = args[0] if args else None
            with self.tracer.span(
                "memory.module_conversation." + name,
                kind="memory",
                attributes={"project_id": getattr(project, "id", "")},
            ) as span:
                result = method(self, *args, **kwargs)
                if span:
                    span.output_summary = "module conversation " + name
                return result
        return wrapped
    return decorator


class ModuleConversationMemory:
    """Project-scoped episodic and version memory for module collaboration."""

    def __init__(self, recent_limit: int = 12, tracer=None) -> None:
        self.recent_limit = max(4, recent_limit)
        self.tracer = tracer

    @traced_module_memory("write_user")
    def remember_user(
        self,
        project: ProjectState,
        content: str,
        mode: str,
        target_module_id: str = "",
    ) -> ModuleConversationMessage:
        message = ModuleConversationMessage(
            id="MSG-" + uuid.uuid4().hex[:10],
            role="user",
            content=content.strip(),
            mode=mode,
            target_module_id=target_module_id,
        )
        project.module_conversation.append(message)
        self._refresh_summary(project)
        return message

    @traced_module_memory("write_result")
    def remember_result(
        self,
        project: ProjectState,
        tree: ModuleTree,
        mode: str,
        instruction: str,
        target_module_id: str = "",
    ) -> ModuleTreeVersion:
        version = ModuleTreeVersion(
            id="MV-" + uuid.uuid4().hex[:10],
            mode=mode,
            instruction=instruction.strip(),
            target_module_id=target_module_id,
            module_tree=tree.model_copy(deep=True),
        )
        project.module_versions.append(version)
        names = "、".join(module.name for module in tree.modules)
        project.module_conversation.append(
            ModuleConversationMessage(
                id="MSG-" + uuid.uuid4().hex[:10],
                role="assistant",
                content="已完成{}，当前模块：{}".format(
                    self.mode_label(mode), names or "无"
                ),
                mode=mode,
                target_module_id=target_module_id,
                version_id=version.id,
            )
        )
        self._refresh_summary(project)
        version.memory_state = project.module_decision_state.model_copy(deep=True)
        return version

    @traced_module_memory("read")
    def context(self, project: ProjectState) -> str:
        self._refresh_summary(project)
        recent = project.module_conversation[project.module_decision_state.recent_start:][-self.recent_limit :]
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
        return "Earlier conversation summary:\n{}\nThe active decision ledger above supersedes conflicting historical turns. The current user instruction takes precedence.\nRecent turns:\n{}".format(
            project.module_memory_summary or "None.",
            json.dumps(payload, ensure_ascii=False),
        )

    def _refresh_summary(self, project: ProjectState) -> None:
        project.module_decision_state = advance(
            project.module_decision_state, project.module_conversation
        )
        project.module_memory_summary = render(project.module_decision_state)

    def restore_state(self, project, version) -> None:
        project.module_decision_state = restore(version, project.module_conversation)
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
        }.get(mode, mode)
