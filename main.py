from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import router
from db.queries import ensure_schema
from services.scheduler import make_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_schema()
    scheduler = make_scheduler()
    scheduler.start()
    logger.info("APScheduler started.")

    try:
        yield
    finally:
        logger.info("Shutting down APScheduler...")
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="SEC Filing Semantic Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
