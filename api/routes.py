from __future__ import annotations

import logging
from typing import Literal, cast

from fastapi import APIRouter, HTTPException

from agent.graph import build_agent_graph
from agent.state import AgentState
from db.queries import get_company
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

_agent_graph = build_agent_graph()


class DriftEventOut(BaseModel):
    section_key: str
    filing_id_old: int | None
    filing_id_new: int
    cosine_score: float
    is_significant: bool
    confidence: float
    summary: str
    change_type: str


class RunResult(BaseModel):
    status: Literal["no_new_filing", "completed", "error"]
    ticker: str
    cik: str
    filing_period: str | None = None
    drift_events: list[DriftEventOut] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


async def _run_for_form(*, ticker: str, cik: str, target_form: str) -> dict:
    initial_state: AgentState = cast(AgentState, {
        "ticker": ticker,
        "cik": cik,
        "target_form": target_form,
        "new_filing_found": False,
        "latest_accession": "",
        "previous_accession": None,
        "latest_period_of_report": None,
        "previous_period_of_report": None,
        "document_text": "",
        "sections": {},
        "embeddings": {},
        "diffs": [],
        "drift_events": [],
        "errors": [],
        "filing_id_new": 0,
        "filing_id_old": None,
    })
    return await _agent_graph.ainvoke(initial_state)


@router.post("/run/{ticker}", response_model=RunResult)
async def run_for_ticker(ticker: str) -> RunResult:
    company = get_company(ticker=ticker)
    if company is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker in companies table: {ticker}")

    # Run both 10-Q and 10-K; merge drift events.
    drift_events: list[DriftEventOut] = []
    errors: list[str] = []
    filing_period: str | None = None
    any_new_filing = False

    for form_type in ("10-Q", "10-K"):
        state = await _run_for_form(ticker=ticker, cik=company["cik"], target_form=form_type)
        any_new_filing = any_new_filing or bool(state.get("new_filing_found"))

        if state.get("latest_period_of_report") and filing_period is None:
            filing_period = state.get("latest_period_of_report")

        errors.extend(state.get("errors", []))

        for ev in state.get("drift_events", []):
            drift_events.append(
                DriftEventOut(
                    section_key=str(ev["section_key"]),
                    filing_id_old=ev.get("filing_id_old"),
                    filing_id_new=int(ev["filing_id_new"]),
                    cosine_score=float(ev["cosine_score"]),
                    is_significant=bool(ev["is_significant"]),
                    confidence=float(ev["confidence"]),
                    summary=str(ev["summary"]),
                    change_type=str(ev["change_type"].value if hasattr(ev["change_type"], "value") else ev["change_type"]),
                )
            )

    if errors and not drift_events and not any_new_filing:
        return RunResult(
            status="error",
            ticker=ticker,
            cik=company["cik"],
            filing_period=None,
            drift_events=[],
            errors=errors,
        )

    if not any_new_filing:
        return RunResult(
            status="no_new_filing",
            ticker=ticker,
            cik=company["cik"],
            filing_period=None,
            drift_events=[],
            errors=errors,
        )

    return RunResult(
        status="completed",
        ticker=ticker,
        cik=company["cik"],
        filing_period=filing_period,
        drift_events=drift_events,
        errors=errors,
    )
