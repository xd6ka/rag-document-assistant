"""Running headers and footers must disappear before chunking, and nothing else may."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import find_repeated_lines, strip_repeated_lines  # noqa: E402

FOOTER = "This publication is available free of charge from: https://doi.org/10.6028/NIST.SP.800-63b"
BODIES = [
    "the claimant transfers a secret to the verifier",
    "authentication establishes the subject claim",
    "identity proofing resolves a person to a record",
    "verifiers decide whether to accept an assertion",
    "federation lets one party assert on another behalf",
    "lifecycle management covers revocation duties",
]


def _page(header: str, body: str) -> str:
    return f"{header}\n{body}\n{FOOTER}"


def test_repeated_header_and_footer_are_removed_but_body_stays():
    pages = [_page("NIST SP 800-63B DIGITAL IDENTITY GUIDELINES: AUTHENTICATION", BODIES[i]) for i in range(6)]
    cleaned, stats = strip_repeated_lines(pages)

    assert stats["patterns"] == 2
    for page, body in zip(cleaned, BODIES):
        assert "DIGITAL IDENTITY GUIDELINES" not in page
        assert "free of charge" not in page
        assert body in page


def test_page_number_variants_count_as_the_same_header():
    pages = [f"H. R. 3684—{i}\n{BODIES[i - 1]}." for i in range(1, 7)]
    cleaned, stats = strip_repeated_lines(pages)

    assert stats["lines_removed"] == 6
    assert all(page.startswith(BODIES[i][:20]) for i, page in enumerate(cleaned))


def test_mid_page_numbered_lines_are_not_treated_as_headers():
    tokens = [a + b for a in "abcdef" for b in "abcdef"]
    pages = []
    for i in range(6):
        lines = [f"paragraph about {tokens[i * 6 + j]}" for j in range(6)]
        lines.insert(3, f"Section {i}. Subsection text that keeps running for a while.")
        pages.append("\n".join(lines))
    cleaned, stats = strip_repeated_lines(pages)
    assert stats["lines_removed"] == 0
    assert all(f"Section {i}." in page for i, page in enumerate(cleaned))


def test_short_document_is_left_alone():
    pages = ["TITLE\nbody one", "TITLE\nbody two", "TITLE\nbody three"]
    cleaned, stats = strip_repeated_lines(pages)
    assert stats["lines_removed"] == 0
    assert cleaned == pages


def test_a_page_that_would_be_emptied_keeps_its_text():
    header = "NIST SP 800-63B DIGITAL IDENTITY GUIDELINES"
    pages = [_page(header, BODIES[i]) for i in range(6)] + [f"{header}\n"]
    cleaned, _ = strip_repeated_lines(pages)
    assert cleaned[-1].strip() == header


def test_unrelated_pages_are_untouched():
    pages = [f"page {word} has its own wording entirely" for word in BODIES]
    assert find_repeated_lines(pages) == set()
    cleaned, stats = strip_repeated_lines(pages)
    assert cleaned == pages
    assert stats == {"patterns": 0, "lines_removed": 0, "chars_removed": 0}
