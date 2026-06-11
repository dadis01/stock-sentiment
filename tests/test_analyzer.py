"""
Unit tests for data_analyzer/analyzer.py.

Uses FastAPI TestClient and mocks all external dependencies:
  - FinBERT model (data_analyzer.sentiment.analyze)
  - yfinance
  - psycopg2 / database.db functions
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_finbert():
    """Patch load_finbert so it doesn't download the model during tests."""
    with patch("data_analyzer.analyzer.load_finbert"):
        yield


@pytest.fixture()
def client(mock_finbert):
    """
    Return a TestClient for the FastAPI app.

    load_finbert is patched so the model never downloads.
    The queue worker thread starts but idles silently.
    """
    from data_analyzer.analyzer import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health endpoint tests
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_ok(client):
    """GET /health must return HTTP 200 with status='ok'."""
    with patch("data_analyzer.analyzer.get_connection") as mock_conn:
        mock_conn.return_value.__enter__ = lambda s: s
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value.cursor.return_value.__enter__ = lambda s: s
        mock_conn.return_value.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value.cursor.return_value.execute = MagicMock()

        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_shows_db_connected(client):
    """GET /health must report db='connected' when psycopg2 succeeds."""
    with patch("data_analyzer.analyzer.get_connection") as mock_conn:
        ctx = MagicMock()
        ctx.__enter__ = lambda s: s
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.cursor.return_value.__enter__ = lambda s: s
        ctx.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = ctx

        resp = client.get("/health")

    assert resp.json()["db"] == "connected"


def test_health_shows_db_error_when_connection_fails(client):
    """GET /health must report db='error' if psycopg2 raises."""
    with patch("data_analyzer.analyzer.get_connection", side_effect=Exception("db down")):
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["db"] == "error"


# ---------------------------------------------------------------------------
# /analyze endpoint tests
# ---------------------------------------------------------------------------

def _make_db_context(headline: str = "AAPL hits record", ticker: str = "AAPL"):
    """
    Return a mock psycopg2 connection context that returns one article row.
    """
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = (ticker, headline)

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur
    return mock_conn


def test_analyze_scores_headline(client):
    """POST /analyze must call the sentiment scorer and return a label."""
    with (
        patch("data_analyzer.analyzer.get_connection", return_value=_make_db_context()),
        patch("data_analyzer.analyzer.score_text", return_value={"label": "positive", "score": 0.91}) as mock_score,
        patch("data_analyzer.analyzer.yf.Ticker") as mock_yf,
        patch("data_analyzer.analyzer.insert_sentiment"),
    ):
        mock_yf.return_value.fast_info.last_price = 185.0
        resp = client.post("/analyze", json={"article_id": 1})

    assert resp.status_code == 200
    mock_score.assert_called_once_with("AAPL hits record")
    assert resp.json()["label"] == "positive"


def test_analyze_fetches_yfinance_price(client):
    """POST /analyze must call yf.Ticker to get the live price."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.fast_info.last_price = 195.75

    with (
        patch("data_analyzer.analyzer.get_connection", return_value=_make_db_context()),
        patch("data_analyzer.analyzer.score_text", return_value={"label": "neutral", "score": 0.5}),
        patch("data_analyzer.analyzer.yf.Ticker", return_value=mock_ticker_obj) as mock_yf,
        patch("data_analyzer.analyzer.insert_sentiment"),
    ):
        resp = client.post("/analyze", json={"article_id": 7})

    assert resp.status_code == 200
    assert abs(resp.json()["price"] - 195.75) < 1e-3
    mock_yf.assert_called_once_with("AAPL")


def test_analyze_writes_sentiment_to_db(client):
    """POST /analyze must call insert_sentiment with the correct arguments."""
    with (
        patch("data_analyzer.analyzer.get_connection", return_value=_make_db_context("Tesla drops", "TSLA")),
        patch("data_analyzer.analyzer.score_text", return_value={"label": "negative", "score": 0.88}),
        patch("data_analyzer.analyzer.yf.Ticker") as mock_yf,
        patch("data_analyzer.analyzer.insert_sentiment") as mock_insert,
    ):
        mock_yf.return_value.fast_info.last_price = 220.0
        resp = client.post("/analyze", json={"article_id": 3})

    assert resp.status_code == 200
    mock_insert.assert_called_once_with(3, "TSLA", "negative", 0.88, 220.0)


def test_analyze_returns_correct_shape(client):
    """POST /analyze response must contain label, score, and price keys."""
    with (
        patch("data_analyzer.analyzer.get_connection", return_value=_make_db_context()),
        patch("data_analyzer.analyzer.score_text", return_value={"label": "neutral", "score": 0.6}),
        patch("data_analyzer.analyzer.yf.Ticker") as mock_yf,
        patch("data_analyzer.analyzer.insert_sentiment"),
    ):
        mock_yf.return_value.fast_info.last_price = 100.0
        resp = client.post("/analyze", json={"article_id": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"label", "score", "price"}
    assert isinstance(body["label"], str)
    assert isinstance(body["score"], float)
    assert isinstance(body["price"], float)


def test_analyze_returns_404_for_missing_article(client):
    """POST /analyze must return 404 when article_id does not exist in DB."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = lambda s: s
    mock_cur.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchone.return_value = None  # article not found

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cur

    with patch("data_analyzer.analyzer.get_connection", return_value=mock_conn):
        resp = client.post("/analyze", json={"article_id": 9999})

    assert resp.status_code == 404
