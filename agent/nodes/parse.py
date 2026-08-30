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
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*1a[^\n]{0,160}?\b(?:risk\s+factors)\b",
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*1\s*(?:[\.,:;\-–—]|\s+)?\s*a[^\n]{0,160}?\b(?:risk\s+factors)\b",
        r"\b(?:risk\s+factors)\b",
    ],
    # Item 7 — Management’s Discussion and Analysis of Financial Condition and Results of Operations
    "mda": [
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*7[^\n]{0,220}?\bmanagement['’]?\s+discussion\s+and\s+analysis\b",
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*7[^\n]{0,220}?\bmanagement\s+discussion\s+and\s+analysis\b",
        r"\bmanagement\s+discussion\s+and\s+analysis\b",
    ],
    # Item 3 — Legal Proceedings
    "legal": [
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*3[^\n]{0,140}?\blegal\s+proceedings\b",
        r"\blegal\s+proceedings\b",
    ],
    # Item 1 — Business
    "business": [
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*1[^\n]{0,140}?\bbusiness\b",
        r"\bbusiness\b",
    ],
    # Item 8 — Financial Statements and Supplementary Data
    "financials": [
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*8[^\n]{0,220}?\bfinancial\s+statements\s+and\s+supplementary\s+data\b",
        r"\bfinancial\s+statements\s+and\s+supplementary\s+data\b",
    ],
}


def _normalize_document_for_heading_detection(document_text: str) -> str:
    """Normalize SEC text into a more parse-friendly form without destroying the narrative."""
    normalized = document_text.replace("\\", " ")
    normalized = normalized.replace("’", "'")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")
    normalized = normalized.replace("\xa0", " ")

    # Drop inline-XBRL taxonomy and footnote noise that often survives as text like:
    # "https://fasb.org/us-gaap/2021-01-31#OtherAssetsNoncurrent"
    normalized = re.sub(r"(?i)https?://\S+", " ", normalized)
    normalized = re.sub(r"(?i)\b(?:ntrue|nfalse|nfy|np1y|np0y|n\d{4}|n[a-z0-9]{2,})\b", " ", normalized)
    normalized = re.sub(r"(?i)fasb\.org(?:/us-gaap/\d{4}#|/\d{4}-\d{2}-\d{2}#)?", " ", normalized)
    normalized = re.sub(r"(?i)(?:us-gaap|dei|xbrli|ix|xbrldt|iso4217|srt|ifrs-full|ifrs|us-ias)[:\w-]*", " ", normalized)
    normalized = re.sub(r"(?i)\s*:\s*/\s*/\s*", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_likely_xbrl_noise(section_text: str) -> bool:
    """Reject taxonomy/fact-tag fragments that look like XBRL metadata, not prose."""
    if not section_text or len(section_text.strip()) < 80:
        return True

    lower_text = section_text.lower()
    xbrl_markers = re.findall(r"(?:fasb\.org|us-gaap|xbrli|xbrldt|dei|srt|ifrs-full|://|/\s*/)", lower_text)
    alpha_words = re.findall(r"[a-z]{4,}", lower_text)
    weird_fact_tokens = re.findall(r"\bn\d+\b|\bn[a-z0-9]{2,}\b", lower_text)

    if xbrl_markers and len(alpha_words) < 40:
        return True
    if xbrl_markers and len(weird_fact_tokens) > 20:
        return True
    if "fasb.org" in lower_text and "the company" not in lower_text and "management" not in lower_text:
        return True
    return False


def _is_reference_only_section(section_text: str) -> bool:
    """Reject section bodies that are just a reference to the heading, not the section content itself."""
    if not section_text:
        return True

    lower_text = section_text.lower()
    reference_phrases = [
        "incorporated herein by reference",
        "under the heading",
        "forward-looking statements",
        "this form 10-k under the heading",
        "see part",
        "referenced herein",
    ]

    if any(p in lower_text for p in reference_phrases):
        return True

    # If the candidate is only a short sentence about the section name and contains no real narrative,
    # it is usually a cross-reference, not body text.
    alpha_words = re.findall(r"[a-z]{4,}", lower_text)
    if len(alpha_words) < 20 and "risk factors" in lower_text or "management" in lower_text or "legal proceedings" in lower_text:
        return True

    return False


def _compile_item_pattern(patterns: list[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for p in patterns:
        compiled.append(re.compile(p, flags=re.IGNORECASE))
    return compiled


def _section_title_start(document_text: str, match_start: int, match_end: int) -> int:
    """Move left to include a nearby Item/Part heading before the section title."""
    lookback_start = max(0, match_start - 250)
    prefix = document_text[lookback_start:match_end]
    item_match = re.search(
        r"(?:part\s+[ivx]+\s*,?\s*)?(?:item|i)\s*(?:1a|1|3|7|8)\b",
        prefix,
        flags=re.IGNORECASE,
    )
    if item_match:
        return lookback_start + item_match.start()
    return match_start


def _find_section_spans(document_text: str) -> dict[str, tuple[int, int]]:
    """
    Returns extracted spans (start, end) in document_text for each section_key found.
    We prefer the highest-quality narrative body, not the first XBRL-heavy candidate.
    """
    headings: list[tuple[int, int, str]] = []

    for section_key, patterns in SECTION_PATTERNS.items():
        compiled = _compile_item_pattern(patterns)
        for pat in compiled:
            for match in pat.finditer(document_text):
                start = _section_title_start(document_text, match.start(), match.end())
                headings.append((start, match.end(), section_key))

    if not headings:
        logger.debug("No SEC section headings matched in normalized document.")
        return {}

    headings.sort(key=lambda x: (x[0], x[1]))

    spans: dict[str, tuple[int, int]] = {}
    for section_key in SECTION_PATTERNS.keys():
        matches = [h for h in headings if h[2] == section_key]
        if not matches:
            continue

        best_start = -1
        best_end = -1
        best_quality = -1.0

        for idx, (start_pos, heading_end, _) in enumerate(matches):
            next_start = matches[idx + 1][0] if idx + 1 < len(matches) else len(document_text)
            content_start = max(heading_end, start_pos)
            content_end = max(content_start, next_start)
            body = document_text[content_start:content_end]
            cleaned = re.sub(r"[ \t]+", " ", body)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

            if not cleaned:
                continue

            if _is_likely_xbrl_noise(cleaned) or _is_reference_only_section(cleaned):
                logger.debug("Discarding non-content candidate for section %s: %s", section_key, cleaned[:180])
                continue

            quality = len(re.findall(r"[a-z]{4,}", cleaned.lower())) * 2 + len(cleaned)
            if quality > best_quality:
                best_start = content_start
                best_end = content_end
                best_quality = quality

        if best_start >= 0 and best_end >= 0:
            spans[section_key] = (best_start, best_end)
            logger.debug("Selected section %s span %d:%d with quality %.1f.", section_key, best_start, best_end, best_quality)

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
    detection_text = _normalize_document_for_heading_detection(document_text)

    # Match against the raw document text because the true heading anchors are preserved there,
    # while the heavily normalized version can remove the item/part references we need to locate.
    match_text = document_text if document_text else detection_text

    state["sections"] = {}
    extracted_spans = _find_section_spans(match_text)

    for section_key in SECTION_PATTERNS.keys():
        normalized = _normalize_section_key(section_key)

        if section_key in extracted_spans:
            start, end = extracted_spans[section_key]
            raw_section = match_text[start:end]
            cleaned = re.sub(r"[ \t]+", " ", raw_section)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

            if _is_likely_xbrl_noise(cleaned) or _is_reference_only_section(cleaned):
                logger.debug("Section %s body deemed non-content (noise or cross-reference); using fallback.", section_key)
                state["sections"][normalized] = _fallback_section_text(match_text)
            else:
                state["sections"][normalized] = cleaned
                logger.debug("Section %s extracted with %d characters.", section_key, len(cleaned))
        else:
            logger.debug("No pattern match for section %s; using fallback.", section_key)
            state["sections"][normalized] = _fallback_section_text(match_text)

    return state
