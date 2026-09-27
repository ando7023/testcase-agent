import json
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .llm import OpenAICompatibleClient


ToolHandler = Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: ToolHandler
    allowed_agents: Set[str]
    side_effect: str = "read"


class ToolRegistry:
    """Permissioned tool catalog shared by ReAct agents."""

    def __init__(self, tracer: Any = None) -> None:
        self._tools: Dict[str, RegisteredTool] = {}
        self.tracer = tracer

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: ToolHandler,
        allowed_agents: List[str],
        side_effect: str = "read",
    ) -> None:
        if name in self._tools:
            raise ValueError("Tool already registered: {}".format(name))
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            allowed_agents=set(allowed_agents),
            side_effect=side_effect,
        )

    def schemas_for(self, agent_name: str) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "side_effect": tool.side_effect,
            }
            for tool in self._tools.values()
            if agent_name in tool.allowed_agents
        ]

    def execute(
        self, agent_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError("Unknown tool: {}".format(tool_name))
        if agent_name not in tool.allowed_agents:
            raise ValueError(
                "Agent {} cannot call tool {}".format(agent_name, tool_name)
            )
        self._validate_arguments(tool, arguments)
        span_context = (
            self.tracer.span(
                "tool." + tool_name,
                kind="tool",
                attributes={
                    "agent": agent_name,
                    "tool": tool_name,
                    "side_effect": tool.side_effect,
                },
                input_value=arguments,
            )
            if self.tracer
            else nullcontext(None)
        )
        with span_context as span:
            result = tool.handler(arguments)
            if span:
                span.output_summary = self._summary(result)
            return result

    @staticmethod
    def _summary(value: Any) -> str:
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
        return rendered[:2000]

    @staticmethod
    def _validate_arguments(
        tool: RegisteredTool, arguments: Dict[str, Any]
    ) -> None:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        schema = tool.input_schema
        properties = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in arguments:
                raise ValueError(
                    "Tool {} requires argument {}".format(tool.name, field)
                )
        unknown = set(arguments) - set(properties)
        if unknown:
            raise ValueError(
                "Tool {} received unknown arguments: {}".format(
                    tool.name, sorted(unknown)
                )
            )
        expected_types = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for field, value in arguments.items():
            expected = properties.get(field, {}).get("type")
            python_type = expected_types.get(expected)
            if python_type and not isinstance(value, python_type):
                raise ValueError(
                    "Tool {} argument {} must be {}".format(
                        tool.name, field, expected
                    )
                )


class ReActRuntime:
    """Bounded Thought-Action-Observation loop using JSON actions."""

    def __init__(
        self,
        llm: OpenAICompatibleClient,
        registry: ToolRegistry,
        max_steps: int = 4,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.max_steps = max(1, max_steps)

    def run(
        self,
        agent_name: str,
        objective: str,
        fallback_plan: List[Tuple[str, Dict[str, Any]]],
        force_demo: bool = False,
    ) -> Dict[str, Any]:
        if force_demo or not self.llm.enabled:
            return self._run_fallback(agent_name, fallback_plan)

        tools = self.registry.schemas_for(agent_name)
        history: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []
        seen = set()
        finished_summary = ""
        for step in range(1, self.max_steps + 1):
            action = self.llm.generate_json(
                self._system_prompt(agent_name, tools),
                "Objective:\n{}\nPrevious actions and observations:\n{}".format(
                    objective,
                    json.dumps(history, ensure_ascii=False),
                ),
            )
            action_type = str(action.get("type", "")).lower()
            if action_type == "finish":
                finished_summary = str(action.get("summary", "")).strip()
                break
            if action_type != "tool":
                history.append(
                    {"step": step, "error": "Action type must be tool or finish"}
                )
                continue

            tool_name = str(action.get("tool", ""))
            arguments = action.get("arguments") or {}
            signature = "{}:{}".format(
                tool_name, json.dumps(arguments, sort_keys=True, ensure_ascii=False)
            )
            if signature in seen:
                history.append(
                    {
                        "step": step,
                        "tool": tool_name,
                        "error": "Duplicate tool call blocked",
                    }
                )
                continue
            seen.add(signature)
            call = {
                "step": step,
                "tool": tool_name,
                "arguments": arguments,
                "reason": str(action.get("reason", ""))[:300],
                "status": "success",
            }
            try:
                result = self.registry.execute(
                    agent_name, tool_name, arguments
                )
                observation = self._compact_result(result)
                call["observation"] = observation
            except (ValueError, RuntimeError) as exc:
                call["status"] = "error"
                call["observation"] = str(exc)
            calls.append(call)
            history.append(call)

        if not finished_summary:
            finished_summary = self._summary_from_calls(calls)
        return {
            "mode": "react",
            "summary": finished_summary,
            "observations": [
                call.get("observation", "")
                for call in calls
                if call.get("status") == "success"
            ],
            "tool_calls": calls,
            "react_steps": len(history),
            "max_steps": self.max_steps,
        }

    def _run_fallback(
        self,
        agent_name: str,
        fallback_plan: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        calls = []
        for step, (tool_name, arguments) in enumerate(
            fallback_plan[: self.max_steps], 1
        ):
            call = {
                "step": step,
                "tool": tool_name,
                "arguments": arguments,
                "reason": "deterministic fallback plan",
                "status": "success",
            }
            try:
                result = self.registry.execute(
                    agent_name, tool_name, arguments
                )
                call["observation"] = self._compact_result(result)
            except (ValueError, RuntimeError) as exc:
                call["status"] = "error"
                call["observation"] = str(exc)
            calls.append(call)
        return {
            "mode": "deterministic_tools",
            "summary": self._summary_from_calls(calls),
            "observations": [
                call.get("observation", "")
                for call in calls
                if call.get("status") == "success"
            ],
            "tool_calls": calls,
            "react_steps": len(calls),
            "max_steps": self.max_steps,
        }

    @staticmethod
    def _system_prompt(
        agent_name: str, tools: List[Dict[str, Any]]
    ) -> str:
        return """You are {}. Research before the next production agent runs.
Choose exactly one tool or finish. Return JSON only.
Tool action: {{"type":"tool","tool":"name","arguments":{{}},"reason":"why"}}
Finish action: {{"type":"finish","summary":"concise evidence-backed findings"}}
Do not repeat an identical call. Do not invent tool results. Available tools:
{}""".format(agent_name, json.dumps(tools, ensure_ascii=False))

    @staticmethod
    def _compact_result(result: Any, limit: int = 5000) -> str:
        if isinstance(result, str):
            value = result
        else:
            value = json.dumps(result, ensure_ascii=False, default=str)
        return value if len(value) <= limit else value[:limit] + "..."

    @staticmethod
    def _summary_from_calls(calls: List[Dict[str, Any]]) -> str:
        successful = [
            "{}: {}".format(call["tool"], call.get("observation", ""))
            for call in calls
            if call.get("status") == "success"
        ]
        return "\n".join(successful) if successful else "No tool evidence collected."
