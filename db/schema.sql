-- Watchlist
CREATE TABLE companies (
    ticker TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    name TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One row per filing fetched
CREATE TABLE filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    accession_number TEXT UNIQUE NOT NULL,
    form_type TEXT,           -- 10-K or 10-Q
    period_of_report TEXT,    -- e.g. 2024-09-30
    filed_date TEXT,
    raw_text TEXT,            -- full cleaned text (optional, for audit)
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- One row per section per filing
CREATE TABLE section_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_id INTEGER REFERENCES filings(id),
    section_key TEXT,         -- e.g. "risk_factors", "mda", "legal"
    chunk_index INTEGER,      -- if section is split into chunks
    chunk_text TEXT,
    embedding BLOB,           -- numpy float32 array serialised with tobytes()
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Significant semantic drift events
CREATE TABLE drift_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    section_key TEXT,
    filing_id_new INTEGER REFERENCES filings(id),
    filing_id_old INTEGER REFERENCES filings(id),
    cosine_score REAL,        -- 1.0 = identical, 0.0 = completely different
    llm_summary TEXT,         -- LLM-generated description of what changed
    is_significant BOOLEAN,   -- LLM verdict: true drift vs cosmetic
    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scheduler state (per scheduled job name)
CREATE TABLE scheduler_state (
    job_name TEXT PRIMARY KEY,
    last_run TEXT,          -- ISO-8601 UTC timestamp
    last_error TEXT        -- error message (if any)
);
