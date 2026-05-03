# SEC Filing Semantic Agent

Lightweight agent that monitors SEC EDGAR filings for companies, computes embedding-based semantic drift between filings, and surfaces drift events via a FastAPI API.

Quickstart

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your API keys (optional but required for LLM reasoning):

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY if using LLM reasoning
```

3. Start the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Add companies to the `companies` table in the SQLite DB (see `db/schema.sql`) then call the POST endpoint to run a scan for a ticker:

```bash
curl -X POST http://localhost:8000/run/TSLA
```

Notes

- The scheduler runs nightly at 02:00 UTC by default and will populate the database with filings and drift events.
- Embeddings are stored in the SQLite DB as raw float32 bytes.
- See `db/schema.sql` for the schema.
