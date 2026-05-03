from __future__ import annotations

import logging

from agent.state import AgentState
from db.queries import ensure_schema, insert_drift_event

logger = logging.getLogger(__name__)


async def persist_node(state: AgentState) -> AgentState:
    """
    persist_node:
      - writes drift events to drift_events table
    """
    ensure_schema()

    drift_events = state.get("drift_events", [])
    ticker = state["ticker"]
    filing_id_new = state["filing_id_new"]

    for event in drift_events:
        insert_drift_event(
            ticker=ticker,
            section_key=event["section_key"],
            filing_id_old=event["filing_id_old"],
            filing_id_new=filing_id_new,
            cosine_score=event["cosine_score"],
            llm_summary=event["summary"],
            is_significant=event["is_significant"],
        )

    return state
