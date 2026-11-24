"""Strip the lines a PDF repeats on every page before chunking.

Real documents carry running headers and footers: the NIST guideline repeats
``NIST SP 800-63B DIGITAL IDENTITY GUIDELINES...`` plus a "free of charge" URL
line on 56 of its first 60 pages. Left in place, every chunk starts with the
same boilerplate, so the embedder sees near-identical text on every page of a
document and the distinctive sentence on that page is a smaller share of the
vector.

The rule is frequency-based, not layout-based: a line (or, after digits are
normalised, a line *shape*) that reappears near the top or the bottom of a large
share of pages is boilerplate. Digits are normalised because the usual header
carries a page number, so exact matching alone misses it. Only the first and
last few lines of a page are considered, because a numbered body line such as
"Section 1 ..." would otherwise look identical to its neighbours.
"""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\d+")
_NON_WORD = re.compile(r"[^0-9a-z]+")
EDGE_LINES = 3


def _key(line: str) -> str:
    """Page-number-insensitive identity of a line."""
    lowered = _NON_WORD.sub(" ", line.lower())
    return _DIGITS.sub("#", lowered).strip()


def _edge_lines(text: str) -> list[str]:
    """The lines close enough to the top or bottom to be a header or footer."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []
    return lines[:EDGE_LINES] + (lines[-EDGE_LINES:] if len(lines) > EDGE_LINES else [])


def find_repeated_lines(
    pages: list[str],
    min_page_share: float = 0.4,
    max_chars: int = 200,
    min_pages: int = 5,
) -> set[str]:
    """Keys of edge lines that appear on at least ``min_page_share`` of the pages."""
    if len(pages) < min_pages:
        return set()

    pages_with: dict[str, int] = {}
    for text in pages:
        keys_on_page = set()
        for stripped in _edge_lines(text):
            if len(stripped) > max_chars:
                continue
            key = _key(stripped)
            if key:
                keys_on_page.add(key)
        for key in keys_on_page:
            pages_with[key] = pages_with.get(key, 0) + 1

    threshold = max(2, int(len(pages) * min_page_share))
    return {key for key, count in pages_with.items() if count >= threshold}


def strip_repeated_lines(
    pages: list[str],
    min_page_share: float = 0.4,
    max_chars: int = 200,
    min_pages: int = 5,
) -> tuple[list[str], dict]:
    """Remove repeated lines from every page, never emptying a page.

    A page whose remaining text is nearly gone keeps its original text: dropping
    a header is a quality win, dropping the only line on a mostly-blank page is
    not.
    """
    repeated = find_repeated_lines(pages, min_page_share, max_chars, min_pages)
    if not repeated:
        return pages, {"patterns": 0, "lines_removed": 0, "chars_removed": 0}

    cleaned: list[str] = []
    lines_removed = 0
    chars_removed = 0
    for text in pages:
        original = text or ""
        kept: list[str] = []
        for line in original.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) <= max_chars and _key(stripped) in repeated:
                lines_removed += 1
                chars_removed += len(line)
                continue
            kept.append(line)

        candidate = "\n".join(kept).strip()
        if len(candidate) < 40 and len(original.strip()) > len(candidate):
            cleaned.append(original)
            continue
        cleaned.append(candidate)

    return cleaned, {
        "patterns": len(repeated),
        "lines_removed": lines_removed,
        "chars_removed": chars_removed,
    }
