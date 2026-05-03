from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np

from db.connection import conn, db_lock

SCHEMA_DEFAULT_PATH = "db/schema.sql"


class CompanyRow(TypedDict):
    ticker: str
    cik: str
    name: str | None
    added_at: str


class FilingRow(TypedDict):
    id: int
    ticker: str
    accession_number: str
    form_type: str | None
    period_of_report: str | None
    filed_date: str | None
    raw_text: str | None
    fetched_at: str


class SchedulerStateRow(TypedDict):
    job_name: str
    last_run: str | None
    last_error: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: str) -> str:
    schema_path = Path(path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema SQL not found: {path}")
    return schema_path.read_text(encoding="utf-8")


def ensure_schema(schema_sql_path: str = SCHEMA_DEFAULT_PATH) -> None:
    """
    Initializes DB schema if the core table is missing.
    Safe to call repeatedly.
    """
    with db_lock:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            ("companies",),
        )
        row = cur.fetchone()
        if row is not None:
            return

        schema_sql = _read_text(schema_sql_path)
        conn.executescript(schema_sql)
        conn.commit()


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32, copy=False)
    # Store a 1D vector as-is. Shape is not persisted; callers should treat it as a flat array.
    return embedding.tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def utc_now() -> str:
    return utc_now_iso()


def upsert_company(ticker: str, cik: str, name: str | None = None) -> None:
    with db_lock:
        conn.execute(
            """
            INSERT INTO companies (ticker, cik, name)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
              cik=excluded.cik,
              name=excluded.name
            """,
            (ticker, cik, name),
        )
        conn.commit()


def list_companies() -> list[CompanyRow]:
    with db_lock:
        cur = conn.execute(
            "SELECT ticker, cik, name, added_at FROM companies ORDER BY added_at DESC"
        )
        rows = cur.fetchall()

    return [
        CompanyRow(
            ticker=str(r[0]),
            cik=str(r[1]),
            name=r[2],
            added_at=str(r[3]),
        )
        for r in rows
    ]


def get_company(ticker: str) -> CompanyRow | None:
    with db_lock:
        cur = conn.execute(
            "SELECT ticker, cik, name, added_at FROM companies WHERE ticker=?",
            (ticker,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    return CompanyRow(ticker=str(row[0]), cik=str(row[1]), name=row[2], added_at=str(row[3]))


def has_accession(accession_number: str) -> bool:
    with db_lock:
        cur = conn.execute(
            "SELECT 1 FROM filings WHERE accession_number=? LIMIT 1",
            (accession_number,),
        )
        return cur.fetchone() is not None


def get_latest_filing(ticker: str, form_type: str | None = None) -> FilingRow | None:
    with db_lock:
        if form_type is None:
            cur = conn.execute(
                """
                SELECT id, ticker, accession_number, form_type, period_of_report, filed_date, raw_text, fetched_at
                FROM filings
                WHERE ticker=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (ticker,),
            )
        else:
            cur = conn.execute(
                """
                SELECT id, ticker, accession_number, form_type, period_of_report, filed_date, raw_text, fetched_at
                FROM filings
                WHERE ticker=? AND form_type=?
                ORDER BY id DESC
                LIMIT 1
                """,
                (ticker, form_type),
            )

        row = cur.fetchone()

    if row is None:
        return None

    return FilingRow(
        id=int(row[0]),
        ticker=str(row[1]),
        accession_number=str(row[2]),
        form_type=row[3],
        period_of_report=row[4],
        filed_date=row[5],
        raw_text=row[6],
        fetched_at=str(row[7]),
    )


def insert_filing(
    *,
    ticker: str,
    accession_number: str,
    form_type: str | None,
    period_of_report: str | None,
    filed_date: str | None,
    raw_text: str | None,
) -> int:
    with db_lock:
        cur = conn.execute(
            """
            INSERT INTO filings (
              ticker, accession_number, form_type, period_of_report, filed_date, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, accession_number, form_type, period_of_report, filed_date, raw_text),
        )
        conn.commit()
        last_rowid = cur.lastrowid
        if last_rowid is None:
            raise RuntimeError("Failed to retrieve inserted row id (lastrowid is None).")
        return int(last_rowid)


def insert_section_vector(
    *,
    filing_id: int,
    section_key: str,
    chunk_index: int,
    chunk_text: str,
    embedding: np.ndarray,
) -> None:
    with db_lock:
        conn.execute(
            """
            INSERT INTO section_vectors (
              filing_id, section_key, chunk_index, chunk_text, embedding
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                filing_id,
                section_key,
                int(chunk_index),
                chunk_text,
                _serialize_embedding(embedding),
            ),
        )
        conn.commit()


def get_section_embeddings_for_filing(filing_id: int) -> dict[str, np.ndarray]:
    with db_lock:
        cur = conn.execute(
            """
            SELECT section_key, embedding
            FROM section_vectors
            WHERE filing_id=?
            """,
            (int(filing_id),),
        )
        rows = cur.fetchall()

    out: dict[str, np.ndarray] = {}
    for row in rows:
        key = str(row[0])
        emb = _deserialize_embedding(row[1])
        out[key] = emb
    return out


def get_section_text_for_filing(filing_id: int) -> dict[str, str]:
    with db_lock:
        cur = conn.execute(
            """
            SELECT section_key, chunk_text
            FROM section_vectors
            WHERE filing_id=?
            """,
            (int(filing_id),),
        )
        rows = cur.fetchall()

    out: dict[str, str] = {}
    for row in rows:
        key = str(row[0])
        text = str(row[1]) if row[1] is not None else ""
        out[key] = text
    return out


def insert_drift_event(
    *,
    ticker: str,
    section_key: str | None,
    filing_id_old: int | None,
    filing_id_new: int,
    cosine_score: float,
    llm_summary: str,
    is_significant: bool,
) -> None:
    with db_lock:
        filing_id_old_param: object = (
            int(filing_id_old) if filing_id_old is not None else None
        )

        conn.execute(
            """
            INSERT INTO drift_events (
              ticker, section_key, filing_id_new, filing_id_old, cosine_score, llm_summary, is_significant
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                section_key,
                int(filing_id_new),
                filing_id_old_param,
                float(cosine_score),
                llm_summary,
                1 if is_significant else 0,
            ),
        )
        conn.commit()


def set_scheduler_state(
    job_name: str, *, last_run: str | None = None, last_error: str | None = None
) -> None:
    with db_lock:
        conn.execute(
            """
            INSERT INTO scheduler_state (job_name, last_run, last_error)
            VALUES (?, ?, ?)
            ON CONFLICT(job_name) DO UPDATE SET
              last_run=excluded.last_run,
              last_error=excluded.last_error
            """,
            (job_name, last_run, last_error),
        )
        conn.commit()


def get_scheduler_state(job_name: str) -> SchedulerStateRow | None:
    with db_lock:
        cur = conn.execute(
            """
            SELECT job_name, last_run, last_error
            FROM scheduler_state
            WHERE job_name=?
            """,
            (job_name,),
        )
        row = cur.fetchone()

    if row is None:
        return None

    return SchedulerStateRow(
        job_name=str(row[0]),
        last_run=row[1],
        last_error=row[2],
    )
