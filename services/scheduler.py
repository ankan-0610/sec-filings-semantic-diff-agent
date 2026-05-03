from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from agent.graph import build_agent_graph
from agent.state import AgentState
from typing import cast

from db.queries import get_scheduler_state, list_companies, set_scheduler_state, utc_now

from config import settings

logger = logging.getLogger(__name__)

TargetForm = Literal["10-K", "10-Q"]
SCHEDULER_JOB_NAME = "nightly_edgar_scan"

_agent_graph = build_agent_graph()


async def run_for_ticker(ticker: str, cik: str, target_form: TargetForm) -> dict:
    # Initial state must satisfy AgentState keys required by TypeDict.
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

    result_state = await _agent_graph.ainvoke(initial_state)
    # result_state is the updated AgentState dict.
    return result_state


async def _run_one_company(ticker: str, cik: str) -> None:
    # Run both quarterly and annual pipelines.
    for form_type in ("10-Q", "10-K"):
        with suppress(Exception):
            await run_for_ticker(ticker=ticker, cik=cik, target_form=form_type)


async def run_all_companies() -> None:
    companies = list_companies()
    if not companies:
        logger.info("No companies in DB; skipping scan.")
        return

    sem = asyncio.Semaphore(3)

    async def _bounded_run(t: str, cik: str) -> None:
        async with sem:
            await _run_one_company(t, cik)

    await asyncio.gather(*[_bounded_run(c["ticker"], c["cik"]) for c in companies])


def make_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=settings.SCHEDULER_TIMEZONE)
    trigger = CronTrigger(
        hour=settings.SCHEDULER_CRON_HOUR,
        minute=settings.SCHEDULER_CRON_MINUTE,
        timezone=settings.SCHEDULER_TIMEZONE,
    )

    scheduler.add_job(
        nightly_edgar_scan,
        trigger=trigger,
        id=SCHEDULER_JOB_NAME,
        name=SCHEDULER_JOB_NAME,
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )
    return scheduler


async def nightly_edgar_scan() -> None:
    try:
        logger.info("Starting nightly EDGAR scan job.")
        await run_all_companies()
        set_scheduler_state(
            SCHEDULER_JOB_NAME,
            last_run=utc_now(),
            last_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("nightly_edgar_scan failed: %s", exc)
        set_scheduler_state(
            SCHEDULER_JOB_NAME,
            last_error=str(exc),
            last_run=utc_now(),
        )
