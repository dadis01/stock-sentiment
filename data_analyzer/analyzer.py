"""
FastAPI analyzer service for the Stock Sentiment platform.

Endpoints:
    GET  /health  — liveness + queue size + DB connectivity
    POST /analyze — score an article by id, store result, return sentiment

A background thread drains article_queue from the collector module,
calling the same scoring logic automatically.

Run standalone:
    uvicorn data_analyzer.analyzer:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analyzer.sentiment import load_finbert, analyze as score_text  # noqa: F401
from database.db import get_connection, insert_sentiment, get_sentiments_by_ticker

# Import the shared queue from the collector (may not exist in test/standalone mode).
try:
    from data_collector.collector import article_queue
except ImportError:
    import queue as _queue_mod
    article_queue = _queue_mod.Queue()

load_dotenv()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""
    article_id: int


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze."""
    label: str
    score: float
    price: float


# ---------------------------------------------------------------------------
# Core scoring logic (used by endpoint and background worker)
# ---------------------------------------------------------------------------

def _score_article(article_id: int) -> dict[str, Any]:
    """
    Read the article headline from DB, run FinBERT, fetch live price, persist result.

    Parameters
    ----------
    article_id : primary key in articles table

    Returns
    -------
    dict with keys: label, score, price, ticker
    """
    # Fetch headline and ticker from DB
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

    # Run FinBERT
    sentiment = score_text(headline)
    label = sentiment["label"]
    score = sentiment["score"]

    # Fetch live price — fast_info raises KeyError internally, so access each attr separately
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

    # Persist
    insert_sentiment(article_id, ticker, label, score, price)
    logger.info(f"Analyzed article_id={article_id} ({ticker}): {label} {score:.4f} @ ${price}")
    return {"label": label, "score": score, "price": price, "ticker": ticker}


# ---------------------------------------------------------------------------
# Background queue worker
# ---------------------------------------------------------------------------

def _queue_worker() -> None:
    """
    Background daemon thread that drains article_queue.

    Blocks on queue.get() and calls _score_article() for each article_id.
    Errors are logged but do not kill the thread.
    """
    logger.info("Queue worker thread started.")
    while True:
        try:
            article_id = article_queue.get(timeout=5)
            logger.debug(f"Queue worker picked up article_id={article_id}.")
            _score_article(article_id)
        except Exception:
            # timeout or scoring error — continue loop
            pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FinBERT and start queue worker on startup."""
    load_finbert()
    worker = threading.Thread(target=_queue_worker, daemon=True, name="queue-worker")
    worker.start()
    logger.info("Analyzer service ready.")
    yield
    logger.info("Analyzer service shutting down.")


app = FastAPI(title="Stock Sentiment Analyzer", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """
    Return service liveness, current queue depth, and DB connectivity status.

    Returns
    -------
    dict: {"status": "ok", "queue_size": int, "db": "connected" | "error"}
    """
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
        "queue_size": article_queue.qsize(),
        "db": db_status,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Score the headline of a single article and persist the sentiment result.

    Parameters (JSON body)
    ----------------------
    article_id : int — primary key of the article to analyze

    Returns
    -------
    AnalyzeResponse: label, score, price
    """
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
