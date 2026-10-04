"""Persisted, bounded contracts for the Agentic controller."""
from typing import Any, Dict, List, Optional
from typing_extensions import Literal
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from .models import utc_now_iso


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["invoke_agent", "invoke_tool", "request_input", "finish"]
    capability: str = Field(default="", max_length=80)
    reason: str = Field(min_length=1, max_length=1500)
    skills: List[str] = Field(default_factory=list, max_length=10)
    mode: Literal["full", "continue", "targeted", "chat"] = "full"
    instruction: str = Field(default="", max_length=4000)
    target_module_id: str = Field(default="", max_length=120)
    question: str = Field(default="", max_length=2000)


class SupervisorStep(BaseModel):
    index: int
    decision: Optional[SupervisorDecision] = None
    status: str = "running"
    observation: Dict[str, Any] = Field(default_factory=dict)
    before: str = ""
    after: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class SupervisorRun(BaseModel):
    id: str = Field(pattern=r"^AR-[a-f0-9]{32}$")
    project_id: str
    clarification_policy: Literal["strict", "evidence_only"] = "strict"
    goal: str = Field(min_length=1, max_length=4000)
    target: Literal["cases", "modules"] = "cases"
    mode: Literal["model", "deterministic"]
    status: Literal["running", "waiting_input", "waiting_confirmation", "completed", "budget_exhausted", "failed", "needs_attention"] = "running"
    max_steps: int = Field(default=12, ge=1, le=100)
    budget_extensions: List[int] = Field(default_factory=list)
    steps: List[SupervisorStep] = Field(default_factory=list)
    responses: List[str] = Field(default_factory=list)
    question: str = ""
    error: str = ""
    review_fingerprint: str = ""
    evidence: List[str] = Field(default_factory=list)
    degraded: bool = False
    issue_ledger: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    review_history: List[Dict[str, Any]] = Field(default_factory=list)
    repair_rounds: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=utc_now_iso)
