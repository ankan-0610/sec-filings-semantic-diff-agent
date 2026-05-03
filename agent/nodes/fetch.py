from __future__ import annotations

import logging

from agent.state import AgentState
from db.queries import ensure_schema, get_latest_filing, insert_filing, upsert_company
from services.edgar import select_and_download_latest

logger = logging.getLogger(__name__)


async def fetch_node(state: AgentState) -> AgentState:
    """
    fetch_node → parse_node → embed_node → diff_node → reason_node → persist_node

    Populates:
      - new_filing_found
      - latest_accession / previous_accession
      - filing_id_new / filing_id_old
      - document_text
      - latest_period_of_report / previous_period_of_report
    """
    # Ensure schema exists before any inserts/selects.
    ensure_schema()

    # Defaults so the "no new filing" conditional branch can still safely
    # reach persist_node without KeyErrors.
    state["new_filing_found"] = False
    state["filing_id_new"] = 0
    state["document_text"] = ""

    # Start with empty outputs for downstream nodes.
    state["sections"] = {}
    state["embeddings"] = {}
    state["diffs"] = []
    state["drift_events"] = []
    state["errors"] = state.get("errors", [])

    ticker = state["ticker"]
    cik = state["cik"]
    target_form = state["target_form"]

    # Persist company row (idempotent).
    upsert_company(ticker=ticker, cik=cik, name=None)

    previous = get_latest_filing(ticker=ticker, form_type=target_form)
    if previous is None:
        state["previous_accession"] = None
        state["filing_id_old"] = None  # type: ignore[assignment]
        state["previous_period_of_report"] = None
    else:
        state["previous_accession"] = previous["accession_number"]
        state["filing_id_old"] = previous["id"]
        state["previous_period_of_report"] = previous["period_of_report"]

    try:
        selection = await select_and_download_latest(
            ticker=ticker,
            cik=cik,
            target_form=target_form,
        )
    except Exception as exc:  # noqa: BLE001 - append and continue pipeline
        logger.exception("fetch_node failed for %s (%s): %s", ticker, target_form, exc)
        state["errors"].append(f"fetch_node: {exc!r}")
        # If we can't fetch, we treat it as no new filing to avoid blowing up the graph.
        state["new_filing_found"] = False
        return state

    if selection is None:
        state["new_filing_found"] = False
        return state

    # Insert the fetched filing.
    filing_id_new = insert_filing(
        ticker=ticker,
        accession_number=selection["accession_number"],
        form_type=selection["form_type"],
        period_of_report=selection["period_of_report"],
        filed_date=selection["filed_date"],
        raw_text=selection["cleaned_text"],
    )

    state["new_filing_found"] = True
    state["latest_accession"] = selection["accession_number"]
    state["latest_period_of_report"] = selection["period_of_report"]
    state["filing_id_new"] = filing_id_new
    state["document_text"] = selection["cleaned_text"]

    return state
