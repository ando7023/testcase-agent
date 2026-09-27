import re
from typing import List, Optional


BOUNDARY_PATTERNS = [
    re.compile(r"\n\s*\n+"),
    re.compile(r"[。！？!?；;](?:[”’」』])?"),
    re.compile(r"\.(?=\s|$)"),
    re.compile(r"\n"),
    re.compile(r"[，,、：:]"),
    re.compile(r"\s+"),
]


def natural_boundary(
    text: str, target: int, lower: int, upper: int
) -> Optional[int]:
    """Find the best semantic boundary near target, ordered by boundary quality."""
    if not text:
        return None
    lower = max(1, lower)
    upper = min(len(text) - 1, upper)
    if lower > upper:
        return None
    for pattern in BOUNDARY_PATTERNS:
        candidates = [
            match.end()
            for match in pattern.finditer(text, lower, upper + 1)
            if lower <= match.end() <= upper
        ]
        if candidates:
            return min(candidates, key=lambda position: abs(position - target))
    return None


def split_text_naturally(text: str, max_chars: int = 1600) -> List[str]:
    """Split long text without exceeding max_chars unless no content exists."""
    remaining = text.strip()
    if not remaining:
        return []
    chunks: List[str] = []
    while len(remaining) > max_chars:
        boundary = natural_boundary(
            remaining,
            target=max_chars,
            lower=max(1, int(max_chars * 0.55)),
            upper=max_chars,
        )
        boundary = boundary or max_chars
        chunk = remaining[:boundary].strip()
        if not chunk:
            boundary = max_chars
            chunk = remaining[:boundary].strip()
        chunks.append(chunk)
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def suggest_two_parts(text: str) -> List[str]:
    """Suggest two balanced parts while preferring a complete semantic unit."""
    normalized = text.strip()
    if len(normalized) < 2:
        return [normalized] if normalized else []
    target = len(normalized) // 2
    boundary = natural_boundary(
        normalized,
        target=target,
        lower=max(1, len(normalized) // 4),
        upper=max(1, (len(normalized) * 3) // 4),
    )
    boundary = boundary or target
    parts = [normalized[:boundary].strip(), normalized[boundary:].strip()]
    return [part for part in parts if part]