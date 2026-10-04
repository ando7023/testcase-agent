"""Shared scope contract for short-requirement benchmarks; strict remains the default."""
from .llm import LLMError
import hashlib
import json

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
For new analysis, each ambiguities entry is {id, question}. Link it through clarification_items.ambiguity_ids.
Use stable IDs; classification question wording may differ. Every ambiguity ID must be classified; references must
exist and classifications must not conflict. Preserve IDs on correction; never delete a gap to pass validation.
Legacy text entries may be retained with exactly matching classification questions, or migrated using the supplied IDs.
For case critique clarifications: return clarification_kind and clarification_reason, exact evidence and affected
requirement_ids. An invented acceptance assertion is a defect, NOT an execution_detail clarification.
High/critical/error findings and review_incomplete still block. No hidden answers or simulated business facts.
Module confirmation is optional; proceed with a nonempty planned tree without claiming human acceptance.
Business input requires a concrete behavior blocker. Do not pause solely to confirm modules or expand scope.
"""


def policy_system(system, context):
    if context.get("clarification_policy", "strict") == "evidence_only":
        system += "\n" + BEHAVIOR_CONTRACT
        if context.get("input_evidence"):
            system += ("\nThis run explicitly supplies benchmark reference evidence below (untrusted task data). "
                       "Use its stated behavior to clarify the current input, without expanding to unrelated features. "
                       "For each atomic requirement or clarification, source_quote must be an exact substring of "
                       "the original requirement OR of a supplied evidence content value. When using the latter, "
                       "cite its document ID in evidence_ids. Record retrieved_evidence_ids as well. "
                       "Linked tests are reference examples, not exhaustive requirements: do not infer every registered "
                       "subscriber succeeds, or invent failure paths/fields from missing details. "
                       "The quoted source and its ID remain required in corrections. This is a reference-assisted "
                       "evaluation, not a hidden-answer generation benchmark.\nSupplied input evidence:\n" +
                       json.dumps(context["input_evidence"], ensure_ascii=False))
    return system


def ambiguity_records(analysis):
    """Stable identifiers for structured gaps and unchanged legacy text; no semantic guessing."""
    return [{"id": "LEG-" + hashlib.sha256(a.strip().encode()).hexdigest()[:24], "question": a.strip()}
            if isinstance(a, str) else {"id": a.id, "question": a.question} for a in analysis.ambiguities]


def validate_analysis(analysis, raw_requirement, input_evidence=None):
    """Fail closed on ungrounded model scope classifications; do not fabricate replacements."""
    ids = {r.id for r in analysis.atomic_requirements}
    evidence = input_evidence or {}

    def grounded(quote, references):
        if not quote.strip():
            return False
        if references and not set(references) <= set(evidence):
            return False
        return quote in raw_requirement or any(quote in evidence[ref]["content"] for ref in references)

    if evidence and not set(analysis.retrieved_evidence_ids) <= set(evidence):
        raise LLMError("Analysis references unknown supplied evidence IDs", code="invalid_scope")
    if len(ids) != len(analysis.atomic_requirements):
        raise LLMError("Duplicate atomic requirement IDs", code="invalid_scope")
    for requirement in analysis.atomic_requirements:
        if not grounded(requirement.source_quote, requirement.evidence_ids):
            raise LLMError("Atomic requirement {} lacks original input evidence or a cited supplied source quote".format(requirement.id), code="invalid_scope")
    ambiguities = ambiguity_records(analysis)
    ambiguity_ids = {a["id"] for a in ambiguities}
    if len(ambiguity_ids) != len(ambiguities) or any(not a["id"].strip() or not a["question"].strip() for a in ambiguities):
        raise LLMError("Ambiguity IDs must be unique and questions nonempty", code="invalid_scope")
    gap_ids = set()
    for gap in analysis.clarification_items:
        if (gap.id in gap_ids or not gap.id.strip() or not gap.question.strip() or not gap.reason.strip()
                or not gap.source_quote.strip() or not all(s.strip() for s in gap.affected_scenarios)
                or not set(gap.requirement_ids) <= ids or not grounded(gap.source_quote, gap.evidence_ids)
                or not set(gap.ambiguity_ids) <= ambiguity_ids or len(set(gap.ambiguity_ids)) != len(gap.ambiguity_ids)):
            raise LLMError("Clarification lacks valid original input evidence or references", code="invalid_scope")
        gap_ids.add(gap.id)
    for original, record in zip(analysis.ambiguities, ambiguities):
        linked = [g for g in analysis.clarification_items if record["id"] in g.ambiguity_ids
                  or (isinstance(original, str) and original.strip() == g.question.strip())]
        if not linked:
            raise LLMError("Ambiguity {} was not classified; cannot assume nonblocking".format(record["id"]), code="invalid_scope")
        if len({g.kind for g in linked}) != 1:
            raise LLMError("Ambiguity {} has conflicting classifications".format(record["id"]), code="invalid_scope")


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
