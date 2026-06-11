"""
Unit tests for data_collector/collector.py.

All external calls (NewsAPI, yfinance, DB inserts) are mocked.
"""

import queue
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

import data_collector.collector as collector_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article(title: str = "Test headline", url: str = "https://x.com", published: str = "2024-01-15T10:00:00Z") -> dict:
    """Return a minimal NewsAPI article dict."""
    return {"title": title, "url": url, "publishedAt": published}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_newsapi_fetches_headlines():
    """fetch_headlines() should call client.get_everything and return articles list."""
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {
        "articles": [_make_article("Apple surges"), _make_article("Apple dips")]
    }
    result = collector_mod.fetch_headlines("AAPL", mock_client)
    mock_client.get_everything.assert_called_once_with(
        q="AAPL", language="en", sort_by="publishedAt", page_size=5
    )
    assert len(result) == 2
    assert result[0]["title"] == "Apple surges"


def test_newsapi_returns_empty_on_error():
    """fetch_headlines() should return [] if the client raises an exception."""
    mock_client = MagicMock()
    mock_client.get_everything.side_effect = Exception("Network error")
    result = collector_mod.fetch_headlines("AAPL", mock_client)
    assert result == []


def test_article_inserted_for_each_ticker():
    """collect_once() must call insert_article once per fetched headline per ticker."""
    headlines = [_make_article(f"Headline {i}") for i in range(3)]
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"articles": headlines}

    with (
        patch.object(collector_mod, "_get_newsapi_client", return_value=mock_client),
        patch("data_collector.collector.insert_article", return_value=42) as mock_insert,
        patch("data_collector.collector.insert_price"),
        patch.object(collector_mod, "fetch_and_store_price"),
    ):
        collector_mod.collect_once()

    # 5 tickers × 3 headlines each = 15 insert calls
    assert mock_insert.call_count == 5 * 3


def test_article_id_put_on_queue():
    """collect_once() must put every returned article_id onto article_queue."""
    headlines = [_make_article()]
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"articles": headlines}

    test_queue: queue.Queue = queue.Queue()

    with (
        patch.object(collector_mod, "_get_newsapi_client", return_value=mock_client),
        patch("data_collector.collector.insert_article", return_value=99),
        patch.object(collector_mod, "article_queue", test_queue),
        patch.object(collector_mod, "fetch_and_store_price"),
    ):
        collector_mod.collect_once()

    # 5 tickers × 1 headline = 5 queue entries, all with id=99
    assert test_queue.qsize() == 5
    while not test_queue.empty():
        assert test_queue.get() == 99


def test_price_fetched_on_collection():
    """collect_once() must call fetch_and_store_price once per ticker."""
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"articles": []}

    with (
        patch.object(collector_mod, "_get_newsapi_client", return_value=mock_client),
        patch.object(collector_mod, "fetch_and_store_price") as mock_price,
    ):
        collector_mod.collect_once()

    assert mock_price.call_count == len(collector_mod.TICKERS)
    called_tickers = [c.args[0] for c in mock_price.call_args_list]
    assert set(called_tickers) == set(collector_mod.TICKERS)


def test_fetch_and_store_price_uses_yfinance():
    """fetch_and_store_price() should read yfinance fast_info and call insert_price."""
    mock_ticker = MagicMock()
    mock_ticker.fast_info.last_price = 185.50

    with (
        patch("data_collector.collector.yf.Ticker", return_value=mock_ticker),
        patch("data_collector.collector.insert_price") as mock_insert_price,
    ):
        collector_mod.fetch_and_store_price("AAPL")

    mock_insert_price.assert_called_once_with("AAPL", 185.50)
