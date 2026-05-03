from __future__ import annotations

import logging

import numpy as np

from agent.state import AgentState
from db.queries import ensure_schema, insert_section_vector
from services.embedder import embed_section

logger = logging.getLogger(__name__)


async def embed_node(state: AgentState) -> AgentState:
    """
    embed_node:
      - embeds each section via services/embedder.py conventions
      - persists section-level embeddings into section_vectors table
    """
    ensure_schema()

    section_texts = state.get("sections", {})
    if not section_texts:
        # Nothing to embed; keep embeddings empty.
        state["embeddings"] = {}
        return state

    filing_id_new = state["filing_id_new"]

    embeddings: dict[str, np.ndarray] = {}
    for section_key, text in section_texts.items():
        vector = embed_section(text)
        # Store in-memory for diff/reason nodes.
        embeddings[section_key] = vector

        # Persist: store section-level vector as a single "chunk".
        insert_section_vector(
            filing_id=filing_id_new,
            section_key=section_key,
            chunk_index=0,
            chunk_text=text,
            embedding=vector,
        )

    state["embeddings"] = embeddings
    return state
