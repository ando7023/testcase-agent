from typing import Any, Dict

from .agents import (
    Agent,
    CaseGenerationAgent as CaseGenerationStrategy,
    CaseReviewAgent as CaseReviewStrategy,
    CaseRevisionAgent as CaseRevisionService,
    ModuleCriticAgent as ModuleCriticStrategy,
    ModulePlanningAgent as ModulePlanningStrategy,
    RequirementUnderstandingAgent as RequirementUnderstandingStrategy,
    ToolResearchAgent,
)
from .llm import OpenAICompatibleClient
from .models import RequirementInput
from .tooling import ReActRuntime


class CoreAgent(Agent):
    """Agent marker with per-run tool telemetry consumed by AgentHarness."""

    last_tool_calls = []
    last_react_steps = 0

    def _capture_research(self, result: Dict[str, Any]) -> str:
        self.last_tool_calls = list(result.get("tool_calls", []))
        self.last_react_steps = int(result.get("react_steps", 0))
        return "{}\n{}".format(
            result.get("summary", ""),
            "\n".join(str(item) for item in result.get("observations", [])),
        ).strip()

    def _reset_telemetry(self) -> None:
        self.last_tool_calls = []
        self.last_react_steps = 0


class RequirementUnderstandingAgent(CoreAgent):
    name = "requirement_understanding"

    def __init__(
        self, llm: OpenAICompatibleClient, runtime: ReActRuntime
    ) -> None:
        super().__init__(llm)
        self.strategy = RequirementUnderstandingStrategy(llm)
        self.research = ToolResearchAgent(llm, runtime, self.name, "requirement")

    def run(self, payload: RequirementInput, context: Dict[str, Any]):
        self._reset_telemetry()
        research_payload = context.get("research_payload")
        if research_payload:
            research = self.research.run(research_payload, context)
            context = dict(context)
            context["react_research"] = self._capture_research(research)
        return self.strategy.run(payload, context)


class ModulePlanningAgent(CoreAgent):
    name = "module_planning"

    def __init__(self, llm: OpenAICompatibleClient) -> None:
        super().__init__(llm)
        self.strategy = ModulePlanningStrategy(llm)

    def run(self, payload: Any, context: Dict[str, Any]):
        self._reset_telemetry()
        return self.strategy.run(payload, context)


class TestCaseGenerationAgent(CoreAgent):
    name = "case_generation"

    def __init__(
        self, llm: OpenAICompatibleClient, runtime: ReActRuntime
    ) -> None:
        super().__init__(llm)
        self.strategy = CaseGenerationStrategy(llm)
        self.revision = CaseRevisionService(llm, self.strategy)
        self.research = ToolResearchAgent(llm, runtime, self.name, "case")

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]):
        self._reset_telemetry()
        if payload.get("action") == "revise":
            return self.revision.run(payload, context)
        research_payload = context.get("research_payload")
        if research_payload and context.get("selected_skills") is None:
            research = self.research.run(research_payload, context)
            context = dict(context)
            context["react_research"] = self._capture_research(research)
        return self.strategy.run(payload, context)


class QualityCriticAgent(CoreAgent):
    name = "quality_critic"

    def __init__(self, llm: OpenAICompatibleClient) -> None:
        super().__init__(llm)
        self.module_strategy = ModuleCriticStrategy(llm)
        self.case_strategy = CaseReviewStrategy(llm)

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]):
        self._reset_telemetry()
        target = payload.get("target")
        if target == "module":
            return self.module_strategy.run(payload, context)
        if target == "case":
            return self.case_strategy.run(payload, context)
        raise ValueError("Quality critic target must be module or case")


CORE_AGENT_NAMES = (
    RequirementUnderstandingAgent.name,
    ModulePlanningAgent.name,
    TestCaseGenerationAgent.name,
    QualityCriticAgent.name,
)
