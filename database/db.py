"""
PostgreSQL database access layer for the Stock Sentiment platform.

All public functions use context managers for connection/cursor lifecycle.
Environment variable DATABASE_URL is read via python-dotenv.
"""

import os
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:secret@localhost:5432/stocksentiment")


def get_connection():
    """
    Return a new psycopg2 connection using DATABASE_URL.

    Raises psycopg2.OperationalError if the database is unreachable.
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except psycopg2.OperationalError as exc:
        logger.error(f"Cannot connect to database: {exc}")
        raise


def init_db() -> None:
    """
    Create the articles, sentiments, and prices tables if they do not exist.

    Safe to call multiple times (uses IF NOT EXISTS).
    """
    create_articles = """
        CREATE TABLE IF NOT EXISTS articles (
            id          SERIAL PRIMARY KEY,
            ticker      VARCHAR(10) NOT NULL,
            headline    TEXT        NOT NULL,
            url         TEXT,
            published_at TIMESTAMP,
            created_at  TIMESTAMP   DEFAULT NOW()
        );
    """
    create_sentiments = """
        CREATE TABLE IF NOT EXISTS sentiments (
            id          SERIAL PRIMARY KEY,
            article_id  INTEGER     REFERENCES articles(id),
            ticker      VARCHAR(10) NOT NULL,
            label       VARCHAR(20),
            score       FLOAT,
            price       FLOAT,
            analyzed_at TIMESTAMP   DEFAULT NOW()
        );
    """
    create_prices = """
        CREATE TABLE IF NOT EXISTS prices (
            id         SERIAL PRIMARY KEY,
            ticker     VARCHAR(10) NOT NULL,
            price      FLOAT,
            fetched_at TIMESTAMP   DEFAULT NOW()
        );
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_articles)
                cur.execute(create_sentiments)
                cur.execute(create_prices)
            conn.commit()
        logger.info("Database tables initialised successfully.")
    except Exception as exc:
        logger.error(f"init_db failed: {exc}")
        raise


def insert_article(
    ticker: str,
    headline: str,
    url: Optional[str],
    published_at: Optional[datetime],
) -> int:
    """
    Insert a news article row and return the new article id.

    Parameters
    ----------
    ticker      : stock symbol, e.g. 'AAPL'
    headline    : article title text
    url         : source URL (may be None)
    published_at: publication datetime (may be None)

    Returns
    -------
    int — the SERIAL primary key of the inserted row
    """
    sql = """
        INSERT INTO articles (ticker, headline, url, published_at)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ticker, headline, url, published_at))
                article_id = cur.fetchone()[0]
            conn.commit()
        logger.debug(f"Inserted article id={article_id} for {ticker}.")
        return article_id
    except Exception as exc:
        logger.error(f"insert_article failed: {exc}")
        raise


def insert_sentiment(
    article_id: int,
    ticker: str,
    label: str,
    score: float,
    price: float,
) -> None:
    """
    Insert a sentiment analysis result linked to an article row.

    Parameters
    ----------
    article_id : FK into articles.id
    ticker     : stock symbol
    label      : 'positive', 'negative', or 'neutral'
    score      : confidence score in [0, 1]
    price      : live stock price at analysis time
    """
    sql = """
        INSERT INTO sentiments (article_id, ticker, label, score, price)
        VALUES (%s, %s, %s, %s, %s);
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (article_id, ticker, label, score, price))
            conn.commit()
        logger.debug(f"Inserted sentiment for article_id={article_id}: {label} {score:.4f}")
    except Exception as exc:
        logger.error(f"insert_sentiment failed: {exc}")
        raise


def insert_price(ticker: str, price: float) -> None:
    """
    Insert a price snapshot for a ticker into the prices table.

    Parameters
    ----------
    ticker : stock symbol
    price  : current market price
    """
    sql = "INSERT INTO prices (ticker, price) VALUES (%s, %s);"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (ticker, price))
            conn.commit()
        logger.debug(f"Inserted price for {ticker}: {price}")
    except Exception as exc:
        logger.error(f"insert_price failed: {exc}")
        raise


def get_sentiments_by_ticker(ticker: str) -> list[dict]:
    """
    Return all sentiment rows for a given ticker, newest first.

    Each dict contains: id, article_id, ticker, label, score, price, analyzed_at.
    """
    sql = """
        SELECT s.id, s.article_id, s.ticker, s.label, s.score, s.price, s.analyzed_at
        FROM sentiments s
        WHERE s.ticker = %s
        ORDER BY s.analyzed_at DESC;
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception as exc:
        logger.error(f"get_sentiments_by_ticker failed: {exc}")
        raise


def get_latest_headlines(ticker: str, limit: int = 10) -> list[dict]:
    """
    Return the most recent articles for a ticker joined with their sentiment scores.

    Each dict contains: headline, url, published_at, label, score, price, analyzed_at.
    Rows without a matching sentiment are excluded.
    """
    sql = """
        SELECT a.headline, a.url, a.published_at,
               s.label, s.score, s.price, s.analyzed_at
        FROM articles a
        JOIN sentiments s ON s.article_id = a.id
        WHERE a.ticker = %s
        ORDER BY s.analyzed_at DESC
        LIMIT %s;
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (ticker, limit))
                rows = [dict(r) for r in cur.fetchall()]
        return rows
    except Exception as exc:
        logger.error(f"get_latest_headlines failed: {exc}")
        raise


def get_all_tickers_summary() -> dict:
    """
    Return a summary dict keyed by ticker with avg sentiment score and latest price.

    Shape: { "AAPL": {"avg_score": 0.72, "latest_price": 185.4}, ... }
    """
    sql_sentiment = """
        SELECT ticker,
               AVG(score)  AS avg_score,
               COUNT(*)    AS total
        FROM sentiments
        GROUP BY ticker;
    """
    sql_price = """
        SELECT DISTINCT ON (ticker) ticker, price
        FROM prices
        ORDER BY ticker, fetched_at DESC;
    """
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql_sentiment)
                sentiment_rows = {r["ticker"]: dict(r) for r in cur.fetchall()}
                cur.execute(sql_price)
                price_rows = {r["ticker"]: r["price"] for r in cur.fetchall()}

        result = {}
        all_tickers = set(sentiment_rows) | set(price_rows)
        for ticker in all_tickers:
            result[ticker] = {
                "avg_score": sentiment_rows.get(ticker, {}).get("avg_score"),
                "latest_price": price_rows.get(ticker),
                "total_analyzed": sentiment_rows.get(ticker, {}).get("total", 0),
            }
        return result
    except Exception as exc:
        logger.error(f"get_all_tickers_summary failed: {exc}")
        raise
