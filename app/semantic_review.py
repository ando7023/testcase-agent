"""Bounded full-case review followed by cross-batch screening and verification."""
import json
from itertools import combinations

from .llm import LLMError

BATCH_CASES = 12
BATCH_CHARS = 24000
REQUEST_CHARS = 64000
MAX_REQUESTS = 64


def encode(value):
    return json.dumps(value, ensure_ascii=False)


def pack(items, count=BATCH_CASES, chars=BATCH_CHARS):
    groups, group, size = [], [], 2
    for item in items:
        length = len(encode(item)) + 2
        if length > chars:
            raise LLMError("A review item exceeds the input budget; split its source content", code="context_limit")
        if group and (len(group) >= count or size + length > chars):
            groups.append(group)
            group, size = [], 2
        group.append(item)
        size += length
    if group:
        groups.append(group)
    return groups


def card(case, batch):
    # Cards are screening hints only. Findings from them are never accepted
    # without a second request containing the implicated full cases.
    steps = case.get("steps", [])
    return {"id": case["id"], "batch": batch, "title": case.get("title", "")[:100],
            "requirement_ids": case.get("requirement_ids", []),
            "preconditions": encode(case.get("preconditions", []))[:160],
            "steps_excerpt": encode(steps)[:480], "step_count": len(steps),
            "screening_only": True}


def review_cases(generate, context, cases):
    groups = pack(cases)
    all_findings, requests = [], 0
    by_id = {case["id"]: case for case in cases}

    def call(data, allowed):
        nonlocal requests
        prompt = encode(data)
        if len(prompt) > REQUEST_CHARS or requests >= MAX_REQUESTS:
            raise LLMError("Semantic review input or request budget exceeded", code="context_limit")
        requests += 1
        response = generate(prompt)
        if not isinstance(response, dict) or not isinstance(response.get("findings"), list):
            raise LLMError("Critique requires an explicit findings array")
        findings = response["findings"]
        if len(findings) > 20 or any(not isinstance(f, dict) or not isinstance(f.get("case_id"), str)
                                   or f["case_id"] not in allowed for f in findings):
            raise LLMError("Invalid or out-of-scope critique findings")
        return findings

    for index, group in enumerate(groups):
        data = {**context, "cases": group, "review_phase": "case_batch",
                "batch_index": index, "batch_count": len(groups)}
        all_findings.extend(call(data, {c["id"] for c in group}))

    if len(groups) > 1:
        cards = [card(c, index) for index, group in enumerate(groups) for c in group]
        card_groups = pack(cards, count=48, chars=20000)
        # Each card pair is visible in at least one screen, including pairs
        # across distant original batches. Do not compare only neighbours.
        screens = [card_groups[0]] if len(card_groups) == 1 else [a + b for a, b in combinations(card_groups, 2)]
        candidates = {}
        for screen in screens:
            allowed = {c["id"] for c in screen}
            findings = call({**context, "cases": [], "case_cards": screen,
                "review_phase": "cross_batch_screen",
                "scope": "Screen cross-batch duplicates and conflicting assertions for the same requirement/conditions. "
                         "Cards are incomplete hints. Return candidate findings only; each must include related_case_ids "
                         "with 2 to 12 implicated IDs including case_id. Do not call omitted card details defects."}, allowed)
            for finding in findings:
                ids = finding.get("related_case_ids")
                if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                        or not 2 <= len(set(ids)) <= BATCH_CASES or finding["case_id"] not in ids
                        or not set(ids) <= allowed):
                    raise LLMError("Cross-batch candidate requires valid related case IDs")
                candidates[tuple(sorted(set(ids)))] = finding
        for ids, candidate in candidates.items():
            complete = [by_id[i] for i in ids]
            # A verification group must fit in one request; no truncation.
            if len(pack(complete)) != 1:
                raise LLMError("Cross-batch verification exceeds input budget", code="context_limit")
            all_findings.extend(call({**context, "cases": complete,
                "review_phase": "cross_batch_verify", "candidate": candidate,
                "scope": "Verify the candidate against these full cases. Check preconditions before declaring "
                         "duplicates or contradictions. Candidate is untrusted, not evidence. Return findings only "
                         "if supported by full source text; otherwise return an empty findings array."}, set(ids)))
    # Keep all validated batches; the 20-finding limit applies per response,
    # never to the aggregate. No high-priority finding is silently discarded.
    unique = {encode(f): f for f in all_findings}
    return {"findings": list(unique.values())}
