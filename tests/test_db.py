"""
Unit tests for database/db.py.

Uses an in-memory SQLite connection (via monkeypatching get_connection) so that
no real PostgreSQL instance is required.  A compatibility shim translates
psycopg2-style SQL to SQLite syntax.
"""

import re
import sqlite3
from datetime import datetime
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# SQLite shim
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
    # %s → ? placeholders
    sql = sql.replace("%s", "?")
    # SERIAL → INTEGER (DDL)
    sql = re.sub(r"\bSERIAL\b", "INTEGER", sql, flags=re.IGNORECASE)
    # DEFAULT NOW() → DEFAULT CURRENT_TIMESTAMP (DDL)
    sql = re.sub(r"DEFAULT\s+NOW\(\)", "DEFAULT CURRENT_TIMESTAMP", sql, flags=re.IGNORECASE)
    # DISTINCT ON (col) → remove (SELECT will return all rows; caller dict-keys by ticker)
    # Also rewrite ORDER BY col, timestamp DESC → GROUP BY col to get one row per ticker
    if re.search(r"DISTINCT\s+ON", sql, re.IGNORECASE):
        sql = re.sub(r"DISTINCT\s+ON\s*\([^)]+\)\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"ORDER\s+BY\s+\w+\s*,\s*\w+\s+DESC", "GROUP BY ticker", sql, flags=re.IGNORECASE)
    return sql


class _FakeConn:
    """
    Wraps sqlite3.Connection to emulate a psycopg2 connection.

    Supports cursor_factory keyword so db.py's RealDictCursor calls work.
    """

    def __init__(self, sqlite_conn: sqlite3.Connection):
        self._conn = sqlite_conn
        sqlite_conn.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self, cursor_factory=None):
        """Return a _FakeCursor; as_dict=True when cursor_factory is provided."""
        as_dict = cursor_factory is not None
        return _FakeCursor(self._conn.cursor(), as_dict=as_dict)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


class _FakeCursor:
    """
    Wraps sqlite3.Cursor.

    Handles:
      - %s → ? placeholder translation
      - RETURNING id via lastrowid
      - PostgreSQL DDL syntax (SERIAL, NOW())
      - DISTINCT ON rewrite
      - dict rows when as_dict=True (emulates RealDictCursor)
    """

    def __init__(self, cur: sqlite3.Cursor, as_dict: bool = False):
        self._cur = cur
        self._as_dict = as_dict
        self._is_returning = False
        self._returning_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cur.close()
        return False

    def execute(self, sql: str, params=()):
        """Execute SQL after translating PostgreSQL syntax to SQLite."""
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
def sqlite_conn():
    """Create a fresh in-memory SQLite DB and return a _FakeConn."""
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.executescript(_SQLITE_SCHEMA)
    return _FakeConn(raw)


@pytest.fixture(autouse=True)
def patch_db(sqlite_conn):
    """Monkeypatch database.db.get_connection to return the SQLite fake."""
    with patch("database.db.get_connection", return_value=sqlite_conn):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from database.db import (  # noqa: E402
    init_db,
    insert_article,
    insert_sentiment,
    get_sentiments_by_ticker,
    get_all_tickers_summary,
)


def test_init_db_creates_tables(sqlite_conn):
    """init_db() should execute without error (tables already exist via fixture)."""
    init_db()  # must not raise


def test_insert_article_returns_id(sqlite_conn):
    """insert_article() must return a positive integer id."""
    article_id = insert_article(
        ticker="AAPL",
        headline="Apple unveils new chip",
        url="https://example.com/apple",
        published_at=datetime(2024, 1, 15, 12, 0),
    )
    assert isinstance(article_id, int)
    assert article_id > 0


def test_insert_sentiment_saves_correctly(sqlite_conn):
    """insert_sentiment() should persist a row readable via get_sentiments_by_ticker."""
    article_id = insert_article("TSLA", "Tesla beats estimates", None, None)
    insert_sentiment(
        article_id=article_id,
        ticker="TSLA",
        label="positive",
        score=0.93,
        price=245.50,
    )
    rows = get_sentiments_by_ticker("TSLA")
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "positive"
    assert abs(row["score"] - 0.93) < 1e-5
    assert abs(row["price"] - 245.50) < 1e-2


def test_get_sentiments_by_ticker_returns_list(sqlite_conn):
    """get_sentiments_by_ticker() must return a list (empty if no data)."""
    result = get_sentiments_by_ticker("GOOGL")
    assert isinstance(result, list)
    assert result == []

    article_id = insert_article("GOOGL", "Google Cloud grows", None, None)
    insert_sentiment(article_id, "GOOGL", "positive", 0.85, 140.0)
    result = get_sentiments_by_ticker("GOOGL")
    assert len(result) == 1


def test_get_all_tickers_summary_returns_dict(sqlite_conn):
    """get_all_tickers_summary() must return a dict with per-ticker avg_score."""
    a1 = insert_article("AAPL", "Record sales", None, None)
    a2 = insert_article("AAPL", "New model launch", None, None)
    insert_sentiment(a1, "AAPL", "positive", 0.8, 180.0)
    insert_sentiment(a2, "AAPL", "positive", 0.9, 181.0)

    sqlite_conn._conn.execute("INSERT INTO prices (ticker, price) VALUES (?, ?)", ("AAPL", 181.0))
    sqlite_conn._conn.commit()

    summary = get_all_tickers_summary()
    assert isinstance(summary, dict)
    assert "AAPL" in summary
    assert abs(summary["AAPL"]["avg_score"] - 0.85) < 1e-5
