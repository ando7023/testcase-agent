"""Shared scope contract for short-requirement benchmarks; strict remains the default."""
from .llm import LLMError

POLICIES = {"strict", "evidence_only"}
BEHAVIOR_CONTRACT = """
Evaluation policy: evidence_only (behavior-level test design, NOT executable acceptance).
This policy scopes the level of detail; it never waives true defects, contradictions or failed reviews.
Use only explicit input requirements and their necessary logical consequences. Every atomic requirement,
module and case must link to input evidence. Do not create requirements from generic QA checklists or skills.
Missing API addresses, concrete field names, error codes, fixture construction or implementation details
are execution_detail gaps when the expected behavior is still determined. Record them; continue designing
observable behavior-level cases with abstract fixtures. Never fill them with invented contracts/proxy assertions.
Unrequested payment, age limits, privacy regulations, logging, retries etc. are out_of_scope, not mandatory tests.
Only behavior_blocker gaps (contradictions or ambiguity that changes the expected outcome) require business input.
Explain the affected requirement/scenario and why its outcome cannot be decided. Preserve unresolved blockers;
independent supported cases may be designed, but do not claim the blocked requirements were covered or finished.
Do not require functional/boundary/exception cases universally. Choose types only when supported by the input.
'Only registered subscribers may establish traces' is a necessary condition: an unregistered subscriber must not
establish a trace. It does NOT promise every registered subscriber succeeds; cancelled/expired is not unregistered.
For analysis: populate clarification_items with id, kind (execution_detail|out_of_scope|behavior_blocker), question,
requirement_ids, affected_scenarios, source_quote copied from raw input, and reason. Use an empty list if none.
Each ambiguities entry must have a corresponding clarification_items question. Never silently omit a real blocker.
For case critique clarifications: return clarification_kind and clarification_reason, exact evidence and affected
requirement_ids. An invented acceptance assertion is a defect, NOT an execution_detail clarification.
High/critical/error findings and review_incomplete still block. No hidden answers or simulated business facts.
Module confirmation is optional; proceed with a nonempty planned tree without claiming human acceptance.
Business input requires a concrete behavior blocker. Do not pause solely to confirm modules or expand scope.
"""


def policy_system(system, context):
    if context.get("clarification_policy", "strict") == "evidence_only":
        return system + "\n" + BEHAVIOR_CONTRACT
    return system


def validate_analysis(analysis, raw_requirement):
    """Fail closed on ungrounded model scope classifications; do not fabricate replacements."""
    ids = {r.id for r in analysis.atomic_requirements}
    if len(ids) != len(analysis.atomic_requirements):
        raise LLMError("Duplicate atomic requirement IDs", code="invalid_scope")
    for requirement in analysis.atomic_requirements:
        if not requirement.source_quote.strip() or requirement.source_quote not in raw_requirement:
            raise LLMError("Atomic requirement lacks original input evidence", code="invalid_scope")
    gap_ids = set()
    for gap in analysis.clarification_items:
        if (gap.id in gap_ids or not gap.id.strip() or not gap.question.strip() or not gap.reason.strip()
                or not gap.source_quote.strip() or not all(s.strip() for s in gap.affected_scenarios)
                or not set(gap.requirement_ids) <= ids or gap.source_quote not in raw_requirement):
            raise LLMError("Clarification lacks valid original input evidence or references", code="invalid_scope")
        gap_ids.add(gap.id)
    questions = {g.question.strip() for g in analysis.clarification_items}
    if any(a.strip() not in questions for a in analysis.ambiguities):
        raise LLMError("Ambiguity was not classified; cannot assume nonblocking", code="invalid_scope")


def behavior_blockers(project):
    if project.clarification_policy != "evidence_only" or not project.analysis:
        return []
    return [g for g in project.analysis.clarification_items if g.kind == "behavior_blocker"]


def scope_summary(project):
    gaps = project.analysis.clarification_items if project.analysis else []
    blockers = behavior_blockers(project)
    covered = {rid for case in project.cases for rid in case.requirement_ids}
    return {
        "clarification_policy": project.clarification_policy,
        "case_design_level": "behavior" if project.clarification_policy == "evidence_only" else "execution_oriented",
        "execution_readiness": "blocked" if blockers else "needs_preparation" if any(g.kind == "execution_detail" for g in gaps) else "not_assessed",
        "clarification_items": [g.model_dump() for g in gaps],
        "blocked_requirement_ids": sorted({rid for g in blockers for rid in g.requirement_ids}),
        "uncovered_requirement_ids": [r.id for r in project.analysis.atomic_requirements if r.id not in covered] if project.analysis else [],
    }
