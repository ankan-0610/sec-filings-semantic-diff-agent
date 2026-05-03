from __future__ import annotations

import logging
from math import sqrt

import numpy as np

from agent.state import AgentState, DiffResult
from config import settings
from db.queries import ensure_schema, get_section_embeddings_for_filing

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
        # For safety: embeddings should have same shape; if not, fall back to 0.
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def diff_node(state: AgentState) -> AgentState:
    """
    diff_node:
      - loads previous filing section vectors from DB
      - computes cosine similarity per current section
      - stores results in state["diffs"]
    """
    ensure_schema()

    filing_id_old = state.get("filing_id_old")
    current_embeddings = state.get("embeddings", {})

    if not current_embeddings:
        state["diffs"] = []
        return state

    # If we have no previous filing, we can't compute drift.
    if filing_id_old is None:
        state["diffs"] = []
        return state

    previous_embeddings = get_section_embeddings_for_filing(int(filing_id_old))

    diffs: list[DiffResult] = []
    for section_key, current_vec in current_embeddings.items():
        prev_vec = previous_embeddings.get(section_key)
        cosine = cosine_similarity(current_vec, prev_vec) if prev_vec is not None else 0.0

        # Skip trivial/cosmetic changes for downstream LLM.
        if cosine <= settings.TRIVIAL_THRESHOLD:
            diffs.append(DiffResult(section_key=section_key, cosine_score=cosine))

    state["diffs"] = diffs
    return state
