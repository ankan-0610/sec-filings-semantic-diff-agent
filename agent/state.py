from __future__ import annotations

from enum import Enum
from typing import NotRequired, TypedDict

import numpy as np


class DriftChangeType(str, Enum):
    content_added = "content_added"
    content_removed = "content_removed"
    content_reframed = "content_reframed"
    new_risk_or_uncertainty = "new_risk_or_uncertainty"
    other = "other"


class DiffResult(TypedDict):
    section_key: str
    cosine_score: float


class DriftEvent(TypedDict):
    section_key: str
    filing_id_old: int | None
    filing_id_new: int
    cosine_score: float
    is_significant: bool
    confidence: float
    summary: str
    change_type: DriftChangeType


class AgentState(TypedDict):
    # Identity / filing selection
    ticker: str
    cik: str
    target_form: str  # "10-K" or "10-Q"

    # Set by fetch_node
    new_filing_found: bool
    latest_accession: str
    previous_accession: str | None

    # Needed to craft prompts
    latest_period_of_report: str | None
    previous_period_of_report: str | None

    # Fetched filing text (for parsing + LLM prompts)
    document_text: str

    # Node intermediates
    sections: dict[str, str]  # section_key -> text
    embeddings: dict[str, np.ndarray]  # section_key -> vector

    diffs: list[DiffResult]  # cosine per section
    drift_events: list[DriftEvent]  # LLM assessed events

    errors: list[str]

    # DB ids needed by nodes for persistence
    filing_id_new: int
    filing_id_old: NotRequired[int | None]
