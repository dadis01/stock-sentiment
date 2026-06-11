"""
Integration tests for the Stock Sentiment platform.

Tests the full pipeline end-to-end using:
  - Monkeypatched SQLite DB (no real PostgreSQL required)
  - Mocked NewsAPI, FinBERT, and yfinance (no real external calls)
"""

import queue
import re
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# SQLite shim (same as test_db.py — kept local to avoid cross-file coupling)
# ---------------------------------------------------------------------------

_SQLITE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS articles (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker       TEXT NOT NULL,
        headline     TEXT NOT NULL,
        url          TEXT,
        published_at TIMESTAMP,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS sentiments (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id  INTEGER REFERENCES articles(id),
        ticker      TEXT NOT NULL,
        label       TEXT,
        score       REAL,
        price       REAL,
        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS prices (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker     TEXT NOT NULL,
        price      REAL,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""


def _translate_sql(sql: str) -> str:
    """Translate PostgreSQL-specific SQL to SQLite-compatible SQL."""
    sql = sql.replace("%s", "?")
    sql = re.sub(r"\bSERIAL\b", "INTEGER", sql, flags=re.IGNORECASE)
    sql = re.sub(r"DEFAULT\s+NOW\(\)", "DEFAULT CURRENT_TIMESTAMP", sql, flags=re.IGNORECASE)
    if re.search(r"DISTINCT\s+ON", sql, re.IGNORECASE):
        sql = re.sub(r"DISTINCT\s+ON\s*\([^)]+\)\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"ORDER\s+BY\s+\w+\s*,\s*\w+\s+DESC", "GROUP BY ticker", sql, flags=re.IGNORECASE)
    return sql


class _FakeConn:
    def __init__(self, sqlite_conn):
        self._conn = sqlite_conn
        sqlite_conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self._conn.cursor(), as_dict=cursor_factory is not None)

    def commit(self):
        self._conn.commit()


class _FakeCursor:
    def __init__(self, cur, as_dict=False):
        self._cur = cur
        self._as_dict = as_dict
        self._is_returning = False
        self._returning_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cur.close()
        return False

    def execute(self, sql, params=()):
        sql = _translate_sql(sql)
        if "RETURNING" in sql.upper():
            sql = sql[: sql.upper().index("RETURNING")].rstrip(", ")
            self._cur.execute(sql, params)
            self._returning_id = self._cur.lastrowid
            self._is_returning = True
        else:
            self._is_returning = False
            self._cur.execute(sql, params)

    def fetchone(self):
        if self._is_returning:
            self._is_returning = False
            return (self._returning_id,)
        row = self._cur.fetchone()
        if row is not None and self._as_dict:
            return dict(row)
        return row

    def fetchall(self):
        rows = self._cur.fetchall()
        if self._as_dict:
            return [dict(r) for r in rows]
        return rows

    def close(self):
        self._cur.close()


@pytest.fixture()
def shared_sqlite():
    """One in-memory SQLite DB shared across both test functions."""
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.executescript(_SQLITE_SCHEMA)
    return _FakeConn(raw)


# ---------------------------------------------------------------------------
# Integration test 1 — full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline(shared_sqlite):
    """
    Verify: article inserted → placed on queue → analyzer scores → sentiment row written.

    Mocks:
      - NewsAPI client (returns one headline for AAPL)
      - get_connection in both db.py and analyzer.py (returns SQLite fake)
      - FinBERT (returns canned sentiment)
      - yfinance (returns canned price)
    """
    import data_collector.collector as collector_mod
    import data_analyzer.analyzer as analyzer_mod
    from database.db import get_sentiments_by_ticker

    test_queue: queue.Queue = queue.Queue()

    mock_newsapi = MagicMock()
    mock_newsapi.get_everything.return_value = {
        "articles": [
            {
                "title": "Apple Q4 earnings beat expectations",
                "url": "https://example.com/aapl",
                "publishedAt": "2024-01-15T09:00:00Z",
            }
        ]
    }

    mock_yf_ticker = MagicMock()
    mock_yf_ticker.fast_info.last_price = 185.0

    # Collect — inserts article into shared_sqlite and puts id on test_queue.
    with (
        patch("database.db.get_connection", return_value=shared_sqlite),
        patch.object(collector_mod, "_get_newsapi_client", return_value=mock_newsapi),
        patch.object(collector_mod, "article_queue", test_queue),
        patch.object(collector_mod, "fetch_and_store_price"),
    ):
        collector_mod.collect_once()

    assert not test_queue.empty(), "Expected article_id on queue after collect_once()"
    article_id = test_queue.get()
    assert isinstance(article_id, int) and article_id > 0

    # Analyze — _score_article uses its own imported get_connection reference,
    # so we must patch BOTH the source module and the analyzer's local reference.
    with (
        patch("database.db.get_connection", return_value=shared_sqlite),
        patch("data_analyzer.analyzer.get_connection", return_value=shared_sqlite),
        patch("data_analyzer.analyzer.insert_sentiment") as mock_insert_sentiment,
        patch("data_analyzer.analyzer.score_text", return_value={"label": "positive", "score": 0.92}),
        patch("data_analyzer.analyzer.yf.Ticker", return_value=mock_yf_ticker),
    ):
        result = analyzer_mod._score_article(article_id)

    assert result["label"] == "positive"
    assert abs(result["score"] - 0.92) < 1e-5
    assert abs(result["price"] - 185.0) < 1e-2

    # Verify insert_sentiment was called with correct args.
    mock_insert_sentiment.assert_called_once_with(article_id, "AAPL", "positive", 0.92, 185.0)


# ---------------------------------------------------------------------------
# Integration test 2 — web app data access
# ---------------------------------------------------------------------------

def test_web_app_reads_results(shared_sqlite):
    """
    Verify that get_latest_headlines and get_all_tickers_summary return
    the correct structure after seeding the test DB with known data.
    """
    from database.db import (
        insert_article,
        insert_sentiment,
        insert_price,
        get_latest_headlines,
        get_all_tickers_summary,
    )

    with patch("database.db.get_connection", return_value=shared_sqlite):
        a_id = insert_article(
            "MSFT",
            "Microsoft Azure revenue surges",
            "https://example.com/msft",
            datetime(2024, 2, 10, 8, 30),
        )
        insert_sentiment(a_id, "MSFT", "positive", 0.88, 420.0)
        insert_price("MSFT", 420.0)

        headlines = get_latest_headlines("MSFT", limit=10)
        summary = get_all_tickers_summary()

    # Headlines structure
    assert isinstance(headlines, list)
    assert len(headlines) >= 1
    h = headlines[0]
    assert "headline" in h
    assert "label" in h
    assert "score" in h
    assert "price" in h
    assert h["label"] == "positive"

    # Summary structure
    assert isinstance(summary, dict)
    assert "MSFT" in summary
    msft = summary["MSFT"]
    assert "avg_score" in msft
    assert "latest_price" in msft
    assert abs(msft["avg_score"] - 0.88) < 1e-5
    assert abs(msft["latest_price"] - 420.0) < 1e-2
