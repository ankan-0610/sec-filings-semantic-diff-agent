from __future__ import annotations

import logging
import re

from agent.state import AgentState
from services.embedder import chunk_by_tokens

logger = logging.getLogger(__name__)

# Section-level patterns keyed by snake_case section key.
# These patterns look for SEC Item headings and then extract text until the next Item heading.
#
# Notes:
# - We keep patterns reasonably general to handle formatting differences across filings.
# - If we fail to locate a section, we fall back to a 512-token window.
SECTION_PATTERNS: dict[str, list[str]] = {
    # Item 1A — Risk Factors
    "risk_factors": [
        r"item\s+1a\s*[,\.]?\s+risk\s+factors",
    ],
    # Item 7 — Management’s Discussion and Analysis of Financial Condition and Results of Operations
    "mda": [
        r"item\s+7\s*[,\.]?\s*management['’]?\s+discussion\s+and\s+analysis",
        r"item\s+7\s*[,\.]?\s*management\s+discussion\s+and\s+analysis",
    ],
    # Item 3 — Legal Proceedings
    "legal": [
        r"item\s+3\s*[,\.]?\s*legal\s+proceedings",
    ],
    # Item 1 — Business
    "business": [
        r"item\s+1\s*[,\.]?\s*business",
    ],
    # Item 8 — Financial Statements and Supplementary Data
    "financials": [
        r"item\s+8\s*[,\.]?\s*financial\s+statements\s+and\s+supplementary\s+data",
    ],
}


def _compile_item_pattern(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for p in patterns:
        compiled.append(re.compile(p, flags=re.IGNORECASE))
    return compiled


def _find_section_spans(document_text: str) -> dict[str, tuple[int, int]]:
    """
    Returns extracted spans (start, end) in document_text for each section_key found.
    If a section_key is not found, it is absent from the dict.
    """
    # Find all headings for all sections, then delimit by the next heading position.
    headings: list[tuple[int, int, str]] = []  # (start_pos, end_pos, section_key)

    for section_key, patterns in SECTION_PATTERNS.items():
        compiled = _compile_item_pattern(patterns)
        for pat in compiled:
            for match in pat.finditer(document_text):
                headings.append((match.start(), match.end(), section_key))
                # Avoid duplicate headings for multiple patterns matching same location.
                # We'll rely on delimiter logic later.

    if not headings:
        return {}

    # Sort by position, then by end position.
    headings.sort(key=lambda x: (x[0], x[1]))

    # For each heading, end is the start of the next heading (any section).
    spans: dict[str, tuple[int, int]] = {}
    for idx, (start_pos, heading_end, section_key) in enumerate(headings):
        # If we already have a span for this section, prefer the earliest occurrence.
        if section_key in spans:
            continue

        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(document_text)
        # Extract content after the heading title.
        content_start = heading_end
        content_end = max(content_start, next_start)
        spans[section_key] = (content_start, content_end)

    return spans


def _normalize_section_key(section_key: str) -> str:
    key = section_key.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def _fallback_section_text(document_text: str) -> str:
    # Fall back to a 512-token window, as required.
    chunks = chunk_by_tokens(document_text, window_tokens=512, overlap_tokens=64)
    if not chunks:
        return ""
    return chunks[0]


def parse_node(state: AgentState) -> AgentState:
    document_text = state["document_text"]

    state["sections"] = {}
    extracted_spans = _find_section_spans(document_text)

    for section_key in SECTION_PATTERNS.keys():
        normalized = _normalize_section_key(section_key)

        if section_key in extracted_spans:
            start, end = extracted_spans[section_key]
            raw_section = document_text[start:end]
            # Collapse whitespace for stability.
            cleaned = re.sub(r"[ \t]+", " ", raw_section)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            state["sections"][normalized] = cleaned
        else:
            logger.debug("No pattern match for section %s; using fallback.", section_key)
            state["sections"][normalized] = _fallback_section_text(document_text)

    return state
