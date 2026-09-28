"""Content-bound human decisions, independent of model review/run history."""
import hashlib
import json


def case_fingerprint(case):
    payload = case.model_dump(exclude={"human_status", "review_status", "generated_by"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def current_feedback(project, case):
    record = next((item for item in reversed(project.feedback) if item.case_id == case.id), None)
    if not record:
        return None
    fingerprint = case_fingerprint(case)
    if record.case_fingerprint:
        return record if record.case_fingerprint == fingerprint else None
    # Legacy decisions are reusable only when an earlier snapshot proves the body.
    versions = [v for v in project.case_versions if v.created_at <= record.created_at]
    if not versions:
        return None
    version = max(enumerate(versions), key=lambda pair: (pair[1].created_at, pair[0]))[1]
    previous = next((item for item in version.cases if item.id == case.id), None)
    if previous is None or case_fingerprint(previous) != fingerprint:
        return None
    return record


def reconcile_feedback(project):
    for case in project.cases:
        record = current_feedback(project, case)
        case.human_status = record.action if record else "pending"


def acceptance_summary(project):
    decisions = [current_feedback(project, case) for case in project.cases]
    accepted = sum(item is not None and item.action in {"adopted", "edited"} for item in decisions)
    rejected = sum(item is not None and item.action == "rejected" for item in decisions)
    total = len(decisions)
    status = "accepted" if total and accepted == total else "partial" if accepted or rejected else "pending"
    return {"status": status, "total": total, "accepted": accepted, "rejected": rejected,
            "pending": total - accepted - rejected,
            "legacy_verified": sum(item is not None and not item.case_fingerprint for item in decisions)}


def example_content(case):
    steps = "\n".join("{}. {} => 预期：{}".format(i, step.action, step.expected)
                      for i, step in enumerate(case.steps, 1))
    return "前置条件：{}\n{}".format("；".join(case.preconditions), steps)


def example_document(project, case, ticket_type):
    fingerprint = case_fingerprint(case)
    record = current_feedback(project, case)
    version_id = record.case_version_id if record else ""
    if record and not record.case_fingerprint:
        versions = [v for v in project.case_versions if v.created_at <= record.created_at]
        version_id = max(enumerate(versions), key=lambda pair: (pair[1].created_at, pair[0]))[1].id
    return {
        "id": "EX-{}-{}-{}".format(project.id, case.id, fingerprint[:16]),
        "source_id": "case-example:{}:{}".format(project.id, case.id),
        "title": "[已采纳示例][{}] {}".format(case.case_type, case.title),
        "content": example_content(case), "doc_type": "case_example",
        "tags": [ticket_type, case.case_type, case.module_id], "source": "human_feedback",
        "metadata": {"scope": "project", "project_id": project.id, "case_id": case.id,
                     "ticket_type": ticket_type, "case_fingerprint": fingerprint,
                     "case_version_id": version_id,
                     "feedback_at": record.created_at if record else "",
                     "provenance": "feedback" if record and record.case_fingerprint else "legacy_snapshot"},
    }
