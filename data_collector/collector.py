"""
Data collector process for the Stock Sentiment platform.

Runs on a 30-minute schedule:
  1. Fetches top-5 headlines per ticker from NewsAPI.
  2. Inserts each article into the DB via database.db.
  3. Places article_id onto a shared queue.Queue for the analyzer.
  4. Fetches and stores the current stock price via yfinance.

Run standalone:
    python data_collector/collector.py
"""

import os
import queue
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import schedule
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from newsapi import NewsApiClient

# Allow running as a top-level script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from database.db import insert_article, insert_price

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# Shared queue consumed by the analyzer background thread (when imported as a module).
article_queue: queue.Queue = queue.Queue()


def _get_newsapi_client() -> NewsApiClient:
    """Instantiate and return a NewsApiClient using NEWSAPI_KEY env var."""
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY is not set — NewsAPI calls will fail.")
    return NewsApiClient(api_key=NEWSAPI_KEY)


def fetch_headlines(ticker: str, client: NewsApiClient) -> list[dict]:
    """
    Fetch the top 5 English-language headlines for *ticker* from NewsAPI.

    Parameters
    ----------
    ticker : stock symbol used as the search query
    client : authenticated NewsApiClient instance

    Returns
    -------
    list of article dicts as returned by NewsAPI (may be empty on error)
    """
    try:
        response = client.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=5,
        )
        articles = response.get("articles", [])
        logger.info(f"Fetched {len(articles)} headlines for {ticker}.")
        return articles
    except Exception as exc:
        logger.error(f"fetch_headlines failed for {ticker}: {exc}")
        return []


def fetch_and_store_price(ticker: str) -> None:
    """
    Fetch the current market price for *ticker* via yfinance and persist it.

    Uses the 'regularMarketPrice' field from the fast_info object; falls back
    to the last closing price if the live price is unavailable.
    """
    price = None
    try:
        # fast_info raises KeyError internally for some attributes — access explicitly
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
        except Exception:
            pass

    if price:
        try:
            insert_price(ticker, price)
            logger.info(f"Stored price for {ticker}: {price}")
        except Exception as exc:
            logger.error(f"fetch_and_store_price failed to insert for {ticker}: {exc}")
    else:
        logger.warning(f"Could not fetch price for {ticker} — skipping.")


def collect_once() -> None:
    """
    Run one full collection cycle: headlines + prices for all tracked tickers.

    For each ticker:
      - Fetches up to 5 headlines from NewsAPI
      - Inserts each article into the DB
      - Puts each article_id onto article_queue
      - Fetches and stores the current price
    """
    client = _get_newsapi_client()
    for ticker in TICKERS:
        articles = fetch_headlines(ticker, client)
        for art in articles:
            headline = art.get("title") or ""
            url = art.get("url")
            published_raw = art.get("publishedAt")
            published_at: datetime | None = None
            if published_raw:
                try:
                    published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                except ValueError:
                    published_at = None

            try:
                article_id = insert_article(ticker, headline, url, published_at)
                article_queue.put(article_id)
                logger.debug(f"Queued article_id={article_id} for {ticker}.")
            except Exception as exc:
                logger.error(f"Failed to insert/queue article for {ticker}: {exc}")

        fetch_and_store_price(ticker)


def run_scheduler() -> None:
    """
    Schedule collect_once() every 30 minutes and block forever.

    Runs an initial collection immediately on startup, then repeats on schedule.
    """
    logger.info("Collector starting — running initial collection.")
    collect_once()
    schedule.every(30).minutes.do(collect_once)
    logger.info("Collector scheduled every 30 minutes.")
    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    run_scheduler()
