"""Shared review gates: severity is not a substitute for evidence or disposition."""
import hashlib
from .models import ReviewReport

FIXABLE_CATEGORIES = {"case_type", "module_coverage", "assertion", "requirement_coverage", "semantic"}
ISSUE_TYPES = {"state_setup", "assertion_semantics", "observability", "data_isolation",
               "requirement_conflict", "redundancy", "missing_contract", "other"}


def contains_evidence(value, quote):
    """Match original text values, including newlines/quotes, not JSON field names."""
    if isinstance(value, str):
        return quote in value
    if isinstance(value, dict):
        return any(contains_evidence(item, quote) for item in value.values())
    if isinstance(value, list):
        return any(contains_evidence(item, quote) for item in value)
    return False


def is_blocking(finding, policy="strict"):
    if finding.severity in {"high", "critical", "error"} or finding.category == "review_incomplete":
        return True
    if finding.disposition == "clarification":
        if (policy == "evidence_only" and finding.clarification_basis_verified
                and finding.clarification_kind in {"execution_detail", "out_of_scope"}
                and finding.clarification_reason.strip()):
            return False
        return True
    if finding.category == "semantic" and finding.disposition == "suggestion":
        return False
    return finding.category in FIXABLE_CATEGORIES


def is_auto_fixable(finding):
    return (finding.category in FIXABLE_CATEGORIES
            and finding.disposition not in {"clarification", "suggestion"}
            and is_blocking(finding))


def issue_key(finding):
    # A stable issue family, not an assertion that two paraphrases are identical.
    identity = [finding.category, finding.case_id, finding.module_id,
                finding.issue_type or finding.detail or finding.message]
    return "ISS-" + hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:16]


def record_review(run, report, fingerprint):
    """Track recurrence without claiming absence proves a defect was resolved."""
    if any(f.category == "review_incomplete" for f in report.findings):
        return
    current = {}
    blocking = lambda f: is_blocking(f, getattr(run, "clarification_policy", "strict"))
    for finding in report.findings:
        finding.issue_id = finding.issue_id or issue_key(finding)
        key = finding.issue_id
        prior = run.issue_ledger.get(key, {})
        entry = current.get(key)
        if entry is None or (blocking(finding) and not entry["blocking"]):
            current[key] = {
                "finding": finding.model_dump(), "blocking": blocking(finding),
                "status": "open", "seen_reviews": prior.get("seen_reviews", 0) + 1,
                "reopened": prior.get("reopened", 0) + int(prior.get("status") == "not_observed"),
                "classification_changed": bool(prior and prior.get("blocking") != blocking(finding)),
                "last_fingerprint": fingerprint,
            }
    for key, entry in run.issue_ledger.items():
        if key not in current:
            entry["status"] = "not_observed"
    run.issue_ledger.update(current)
    run.review_history.append({"fingerprint": fingerprint, "score": report.score,
                               "findings": [f.model_dump() for f in report.findings]})
    run.review_history = run.review_history[-4:]


def restore_review_history(run):
    """Read older runs' saved reviews without granting a fresh-review credential."""
    if run.review_history:
        return
    prior = [step.observation.get("review") for step in run.steps
             if step.status == "success" and step.decision
             and step.decision.capability == "quality_critic" and step.observation.get("review")]
    for report in prior[-4:]:
        record_review(run, ReviewReport.model_validate(report), "")
