from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, TypedDict

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from config import settings
from db.queries import has_accession


class FilingSelection(TypedDict):
    ticker: str
    cik_padded: str
    accession_number: str  # accession number as in EDGAR (with dashes)
    accession_no_dashes: str  # accession number without dashes
    primary_doc: str
    form_type: str
    period_of_report: str | None
    filed_date: str | None
    cleaned_text: str


@dataclass(frozen=True)
class EdgarDocInfo:
    accession_number: str
    accession_no_dashes: str
    primary_doc: str
    form_type: str
    period_of_report: str | None
    filed_date: str | None


def pad_cik(cik: str) -> str:
    # EDGAR expects 10-digit CIK for submissions and archive paths.
    digits = re.sub(r"\D+", "", cik)
    return digits.zfill(10)


def _headers() -> dict[str, str]:
    return {"User-Agent": settings.EDGAR_USER_AGENT}


def _data_submissions_url(cik_padded: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik_padded}.json"


def _archives_filing_url(cik_padded: str, accession_no_dashes: str, primary_doc: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/"
        f"{accession_no_dashes}/{primary_doc}"
    )


@retry(wait=wait_exponential_jitter(initial=0.5, max=10), stop=stop_after_attempt(5))
async def fetch_submissions_json(cik_padded: str) -> dict[str, Any]:
    await asyncio.sleep(settings.EDGAR_RATE_LIMIT_SLEEP)
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(_data_submissions_url(cik_padded), headers=_headers())
        resp.raise_for_status()
        # data.sec.gov returns JSON
        return resp.json()


def _pick_latest_doc(recent: dict[str, Any], target_form: str) -> EdgarDocInfo | None:
    """
    Select the most recent (first element in recent.* arrays) that matches target_form
    and has not been processed yet (by accession number).
    """
    accessions: list[str] = recent.get("accessionNumber") or []
    forms: list[str] = recent.get("form") or []
    primary_docs: list[str] = recent.get("primaryDocument") or []
    report_dates: list[str] = recent.get("reportDate") or []
    filing_dates: list[str] = recent.get("filingDate") or []

    # EDGAR arrays are aligned by index; most recent is at index 0.
    for i, (acc, form) in enumerate(zip(accessions, forms)):
        if form != target_form:
            continue
        if has_accession(acc):
            continue

        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        accession_no_dashes = acc.replace("-", "")
        period_of_report = report_dates[i] if i < len(report_dates) else None
        filed_date = filing_dates[i] if i < len(filing_dates) else None
        if not primary_doc:
            continue

        return EdgarDocInfo(
            accession_number=acc,
            accession_no_dashes=accession_no_dashes,
            primary_doc=primary_doc,
            form_type=form,
            period_of_report=period_of_report,
            filed_date=filed_date,
        )

    return None


@retry(wait=wait_exponential_jitter(initial=0.5, max=10), stop=stop_after_attempt(5))
async def fetch_filing_html(
    cik_padded: str,
    accession_no_dashes: str,
    primary_doc: str,
) -> str:
    await asyncio.sleep(settings.EDGAR_RATE_LIMIT_SLEEP)
    async with httpx.AsyncClient(timeout=40) as client:
        url = _archives_filing_url(cik_padded, accession_no_dashes, primary_doc)
        # Follow redirects from the SEC (some archive paths redirect between
        # zero-padded and non-padded CIK directories). Allow httpx to follow
        # them automatically so we get the final HTML content.
        resp = await client.get(url, headers=_headers(), follow_redirects=True)
        resp.raise_for_status()
        html = resp.text

    # Choose parser dynamically: some SEC filings are XML/XHTML and will raise
    # an XMLParsedAsHTMLWarning when parsed with the HTML parser. Detect XML
    # by a leading XML declaration and parse as XML in that case to avoid the
    # warning and get more reliable parsing.
    html_start = html.lstrip()[:200].lower()
    parser = "lxml-xml" if html_start.startswith("<?xml") else "lxml"
    soup = BeautifulSoup(html, parser)
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # SEC filings can be huge; preserve paragraph-ish boundaries.
    text = soup.get_text(separator="\\n")
    # Collapse excessive whitespace.
    text = re.sub(r"[ \\t]+", " ", text)
    text = re.sub(r"\\n{3,}", "\\n\\n", text).strip()
    return text


async def select_and_download_latest(
    *,
    ticker: str,
    cik: str,
    target_form: str,
) -> FilingSelection | None:
    cik_padded = pad_cik(cik)

    submissions = await fetch_submissions_json(cik_padded)
    filings = submissions.get("filings", {})
    recent = filings.get("recent", {})
    if not isinstance(recent, dict):
        return None

    selected = _pick_latest_doc(recent, target_form=target_form)
    if selected is None:
        return None

    cleaned_text = await fetch_filing_html(
        cik_padded=cik_padded,
        accession_no_dashes=selected.accession_no_dashes,
        primary_doc=selected.primary_doc,
    )

    return FilingSelection(
        ticker=ticker,
        cik_padded=cik_padded,
        accession_number=selected.accession_number,
        accession_no_dashes=selected.accession_no_dashes,
        primary_doc=selected.primary_doc,
        form_type=selected.form_type,
        period_of_report=selected.period_of_report,
        filed_date=selected.filed_date,
        cleaned_text=cleaned_text,
    )
