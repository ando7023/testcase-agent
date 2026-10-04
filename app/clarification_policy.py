"""Shared scope contract for short-requirement benchmarks; strict remains the default."""
from .llm import LLMError
import hashlib
import json
import re

VALIDATION_MESSAGES = {
    "unknown_evidence_id": "引用了本次未供应的证据 ID",
    "duplicate_requirement_id": "原子需求 ID 重复",
    "source_id_as_requirement": "文档 ID 不能作为原子需求 ID",
    "ungrounded_quote": "引文不在原始需求或所引用证据的正文中",
    "invalid_gap_id": "澄清项 ID 为空或重复",
    "empty_text": "必填文本或场景为空",
    "unknown_requirement_id": "应引用 atomic_requirements 的 ID，不能引用证据文档 ID",
    "invalid_ambiguity_id": "歧义 ID 为空或重复，或问题为空",
    "invalid_ambiguity_link": "歧义关联 ID 不存在或重复",
    "unclassified_ambiguity": "歧义尚未关联分类",
    "conflicting_classifications": "同一歧义存在 conflicting classifications",
}


def validation_diagnostics(issues):
    """Only fixed field paths and known codes may leave validation diagnostics."""
    if not isinstance(issues, list):
        return []
    return [{"field": i["field"], "code": i["code"]} for i in issues
            if isinstance(i, dict) and isinstance(i.get("code"), str) and i["code"] in VALIDATION_MESSAGES
            and isinstance(i.get("field"), str) and re.fullmatch(
                r"(?:atomic_requirements|clarification_items|ambiguities|retrieved_evidence_ids)(?:\[\d+\])?(?:\.[a-z_]+)?", i["field"])][:20]

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
An 'only eligible actors may perform an action' rule is a necessary condition: ineligible actors cannot perform it.
It does NOT promise every eligible actor succeeds. Generic policy examples are not input domain facts.
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
                       "Keep atomic requirement IDs distinct from supplied document IDs: requirement_ids references "
                       "atomic_requirements[].id, evidence_ids references the supplied document IDs. "
                       "Linked tests are reference examples with their own preconditions and postconditions, not "
                       "exhaustive requirements or a universal success guarantee. Their documented postconditions "
                       "are usable behavior-level oracles within their documented preconditions. "
                       "Do not turn the negation of an example precondition into a mandatory failure scenario: "
                       "an unavailable prerequisite is outside the reference example unless the original input "
                       "explicitly requires that scenario. Missing outcomes for such unrequested scenarios are "
                       "out_of_scope suggestions, not behavior_blockers. Do not ask for a second requirements "
                       "document to restate a supplied example's explicit outcome. Undefined concrete fixtures, "
                       "interfaces or internal meaning are execution_detail when that observable outcome is known. "
                       "If the original input explicitly requires an outcome that remains ambiguous or contradictory, "
                       "preserve it as a behavior_blocker. Do not invent failure paths or fields. "
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
    issues = []

    def issue(field, code):
        issues.append({"field": field, "code": code})

    def grounded(quote, references):
        if not quote.strip():
            return False
        if references and not set(references) <= set(evidence):
            return False
        return quote in raw_requirement or any(quote in evidence[ref]["content"] for ref in references)

    if evidence and not set(analysis.retrieved_evidence_ids) <= set(evidence):
        issue("retrieved_evidence_ids", "unknown_evidence_id")
    if len(ids) != len(analysis.atomic_requirements):
        issue("atomic_requirements", "duplicate_requirement_id")
    for index, requirement in enumerate(analysis.atomic_requirements):
        prefix = "atomic_requirements[{}].".format(index)
        if requirement.id in evidence:
            issue(prefix + "id", "source_id_as_requirement")
        if not set(requirement.evidence_ids) <= set(evidence):
            issue(prefix + "evidence_ids", "unknown_evidence_id")
        if not grounded(requirement.source_quote, requirement.evidence_ids):
            issue(prefix + "source_quote", "ungrounded_quote")
    ambiguities = ambiguity_records(analysis)
    ambiguity_ids = {a["id"] for a in ambiguities}
    if len(ambiguity_ids) != len(ambiguities) or any(not a["id"].strip() or not a["question"].strip() for a in ambiguities):
        issue("ambiguities", "invalid_ambiguity_id")
    gap_ids = set()
    for index, gap in enumerate(analysis.clarification_items):
        prefix = "clarification_items[{}].".format(index)
        if gap.id in gap_ids or not gap.id.strip():
            issue(prefix + "id", "invalid_gap_id")
        for field in ("question", "reason", "affected_scenarios"):
            value = getattr(gap, field)
            if not (all(s.strip() for s in value) if isinstance(value, list) else value.strip()):
                issue(prefix + field, "empty_text")
        if not set(gap.requirement_ids) <= ids:
            issue(prefix + "requirement_ids", "unknown_requirement_id")
        if not set(gap.evidence_ids) <= set(evidence):
            issue(prefix + "evidence_ids", "unknown_evidence_id")
        if not grounded(gap.source_quote, gap.evidence_ids):
            issue(prefix + "source_quote", "ungrounded_quote")
        if not set(gap.ambiguity_ids) <= ambiguity_ids or len(set(gap.ambiguity_ids)) != len(gap.ambiguity_ids):
            issue(prefix + "ambiguity_ids", "invalid_ambiguity_link")
        gap_ids.add(gap.id)
    for index, (original, record) in enumerate(zip(analysis.ambiguities, ambiguities)):
        linked = [g for g in analysis.clarification_items if record["id"] in g.ambiguity_ids
                  or (isinstance(original, str) and original.strip() == g.question.strip())]
        if not linked:
            issue("ambiguities[{}]".format(index), "unclassified_ambiguity")
        elif len({g.kind for g in linked}) != 1:
            issue("ambiguities[{}]".format(index), "conflicting_classifications")
    if issues:
        diagnostics = validation_diagnostics(issues)
        failure = LLMError("; ".join(i["field"] + ": " + i["code"] for i in diagnostics), code="invalid_scope")
        failure.validation_issues = diagnostics
        raise failure


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
        "clarification_scope_review": [r.model_dump() for r in project.analysis.clarification_scope_review] if project.analysis else [],
        "blocked_requirement_ids": sorted({rid for g in blockers for rid in g.requirement_ids}),
        "uncovered_requirement_ids": [r.id for r in project.analysis.atomic_requirements if r.id not in covered] if project.analysis else [],
    }
