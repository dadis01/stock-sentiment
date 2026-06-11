"""
FastAPI analyzer service for the Stock Sentiment platform.

Endpoints:
    GET  /health  — liveness + pending article count + DB connectivity
    POST /analyze — score an article by id, store result, return sentiment

A background thread polls the DB every 10 seconds for unanalyzed articles
and scores them automatically — no shared in-memory queue needed, so this
works correctly across separate Heroku dynos.

Run standalone:
    uvicorn data_analyzer.analyzer:app --host 0.0.0.0 --port 8000
"""

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analyzer.sentiment import load_finbert, analyze as score_text
from database.db import get_connection, insert_sentiment

load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    article_id: int


class AnalyzeResponse(BaseModel):
    label: str
    score: float
    price: float


# ---------------------------------------------------------------------------
# Core scoring logic
# ---------------------------------------------------------------------------

def _score_article(article_id: int) -> dict[str, Any]:
    """Read headline from DB, run sentiment, fetch price, persist result."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ticker, headline FROM articles WHERE id = %s;",
                (article_id,),
            )
            row = cur.fetchone()

    if row is None:
        raise ValueError(f"article_id={article_id} not found in DB.")

    ticker, headline = row

    sentiment = score_text(headline)
    label = sentiment["label"]
    score = sentiment["score"]

    price = 0.0
    try:
        info = yf.Ticker(ticker).fast_info
        try:
            price = float(info.last_price)
        except (KeyError, TypeError, AttributeError):
            pass
        if not price:
            try:
                price = float(info.regular_market_price)
            except (KeyError, TypeError, AttributeError):
                pass
    except Exception:
        pass
    if not price:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception as exc:
            logger.warning(f"Could not fetch price for {ticker}: {exc}")

    insert_sentiment(article_id, ticker, label, score, price)
    logger.info(f"Analyzed article_id={article_id} ({ticker}): {label} {score:.4f} @ ${price}")
    return {"label": label, "score": score, "price": price, "ticker": ticker}


# ---------------------------------------------------------------------------
# DB-polling background worker (replaces shared in-memory queue)
# ---------------------------------------------------------------------------

def _pending_count() -> int:
    """Return how many articles have no sentiment row yet."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM articles a
                    LEFT JOIN sentiments s ON s.article_id = a.id
                    WHERE s.id IS NULL;
                    """
                )
                return cur.fetchone()[0]
    except Exception:
        return -1


def _fetch_pending_ids(batch: int = 10) -> list[int]:
    """Return up to `batch` article ids that have no sentiment yet."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.id FROM articles a
                    LEFT JOIN sentiments s ON s.article_id = a.id
                    WHERE s.id IS NULL
                    ORDER BY a.created_at ASC
                    LIMIT %s;
                    """,
                    (batch,),
                )
                return [row[0] for row in cur.fetchall()]
    except Exception as exc:
        logger.warning(f"_fetch_pending_ids error: {exc}")
        return []


def _db_poll_worker() -> None:
    """Daemon thread: poll DB every 10 s and score any unanalyzed articles."""
    logger.info("DB-poll worker started.")
    while True:
        try:
            ids = _fetch_pending_ids(batch=10)
            for article_id in ids:
                try:
                    _score_article(article_id)
                except Exception as exc:
                    logger.error(f"Failed to score article_id={article_id}: {exc}")
        except Exception as exc:
            logger.error(f"Poll worker error: {exc}")
        time.sleep(10)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_finbert()
    worker = threading.Thread(target=_db_poll_worker, daemon=True, name="db-poll-worker")
    worker.start()
    logger.info("Analyzer service ready.")
    yield
    logger.info("Analyzer service shutting down.")


app = FastAPI(title="Stock Sentiment Analyzer", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    db_status = "connected"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
    except Exception as exc:
        logger.warning(f"Health check DB error: {exc}")
        db_status = "error"

    return {
        "status": "ok",
        "queue_size": _pending_count(),
        "db": db_status,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        result = _score_article(request.article_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"POST /analyze error: {exc}")
        raise HTTPException(status_code=500, detail="Analysis failed.")

    return AnalyzeResponse(
        label=result["label"],
        score=result["score"],
        price=result["price"],
    )
