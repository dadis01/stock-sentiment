"""
Streamlit dashboard for the Stock Sentiment platform.

Four sections:
  1. Stock selector + action buttons
  2. Sentiment overview bar chart (Plotly)
  3. Latest headlines table with colour-coded labels
  4. Monitoring metrics panel

Run:
    streamlit run web_app/app.py
"""

import os
import sys
from pathlib import Path

from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from database.db import (
    get_all_tickers_summary,
    get_latest_headlines,
    get_sentiments_by_ticker,
)

load_dotenv()

ANALYZER_URL = os.getenv("ANALYZER_URL", "http://localhost:8000")
TICKERS = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
_icon = Image.open(Path(__file__).parent / "icon.png")
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon=_icon,
    layout="wide",
    menu_items={},
)

st.title("Stock Market Sentiment Analysis Platform")

# Hide Deploy button and hamburger menu.
st.markdown(
    """
    <style>
    .stDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Print button + Running label injected via component so it can reach window.top.
components.html(
    """
    <style>
    body { margin: 0; background: transparent; }
    #wrap {
        position: fixed;
        top: 0; right: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        z-index: 999999;
    }
    #print-btn {
        background: white;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 14px;
        cursor: pointer;
        color: #333;
    }
    #print-btn:hover { background: #f0f0f0; }
    #running-label {
        display: none;
        font-size: 13px;
        color: #e67e22;
        font-weight: 600;
    }
    </style>
    <div id="wrap">
        <span id="running-label">⏳ Running...</span>
        <button id="print-btn" onclick="window.top.print()">🖨️ Print</button>
    </div>
    <script>
    (function() {
        const label = document.getElementById('running-label');
        try {
            const observer = new MutationObserver(function() {
                const running = window.top.document.querySelector('[data-testid="stStatusWidget"]');
                label.style.display = running ? 'inline' : 'none';
            });
            observer.observe(window.top.document.body, { childList: true, subtree: true });
        } catch(e) {}
    })();
    </script>
    """,
    height=50,
)

# ---------------------------------------------------------------------------
# Section 1 — Stock selector
# ---------------------------------------------------------------------------
st.header("1. Stock Selector")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_ticker = st.selectbox(
        "Choose a ticker to inspect:",
        options=TICKERS,
        index=0,
    )

with col2:
    if st.button("Analyze now"):
        """
        Trigger on-demand analysis for the most recent unanalyzed article
        of the selected ticker by POSTing to /analyze.
        """
        # Find the latest article_id for this ticker that has no sentiment yet.
        try:
            from database.db import get_connection
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT a.id FROM articles a
                        LEFT JOIN sentiments s ON s.article_id = a.id
                        WHERE a.ticker = %s AND s.id IS NULL
                        ORDER BY a.created_at DESC
                        LIMIT 1;
                        """,
                        (selected_ticker,),
                    )
                    row = cur.fetchone()

            if row:
                article_id = row[0]
                resp = requests.post(
                    f"{ANALYZER_URL}/analyze",
                    json={"article_id": article_id},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"Article {article_id} → {data['label']} "
                        f"(score {data['score']:.3f}, price ${data['price']:.2f})"
                    )
                else:
                    st.error(f"Analyzer returned {resp.status_code}: {resp.text}")
            else:
                st.info(f"No unanalyzed articles found for {selected_ticker}.")
        except Exception as exc:
            logger.error(f"Analyze now failed: {exc}")
            st.error(f"Error contacting analyzer: {exc}")

with col3:
    if st.button("Refresh data"):
        """Force a full page rerun to reload all DB data."""
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Sentiment overview bar chart
# ---------------------------------------------------------------------------
st.header("2. Sentiment Overview")


def _label_to_color(label: str | None) -> str:
    """Map a sentiment label string to a Plotly color hex."""
    mapping = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
    return mapping.get((label or "").lower(), "#95a5a6")


try:
    summary = get_all_tickers_summary()
    tickers_with_data = [t for t in TICKERS if t in summary and summary[t]["avg_score"] is not None]

    if tickers_with_data:
        avg_scores = [float(summary[t]["avg_score"] or 0) for t in tickers_with_data]

        # Determine dominant label per ticker from recent sentiments
        label_per_ticker = []
        for t in tickers_with_data:
            rows = get_sentiments_by_ticker(t)
            if rows:
                # Most common label across all rows
                from collections import Counter
                counts = Counter(r["label"] for r in rows)
                label_per_ticker.append(counts.most_common(1)[0][0])
            else:
                label_per_ticker.append("neutral")

        bar_colors = [_label_to_color(lbl) for lbl in label_per_ticker]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=tickers_with_data,
                    y=avg_scores,
                    marker_color=bar_colors,
                    text=[f"{s:.3f}" for s in avg_scores],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Average Sentiment Score per Ticker",
            xaxis_title="Ticker",
            yaxis_title="Avg Score",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data yet. Run the collector and analyzer to populate the DB.")
except Exception as exc:
    logger.error(f"Sentiment chart error: {exc}")
    st.error(f"Could not load sentiment data: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Headlines table
# ---------------------------------------------------------------------------
st.header(f"3. Latest Headlines — {selected_ticker}")

LABEL_COLORS = {"positive": "green", "negative": "red", "neutral": "gray"}


def _colorize_label(label: str) -> str:
    """Return an HTML-colored span for the label string."""
    color = LABEL_COLORS.get((label or "").lower(), "gray")
    return f'<span style="color:{color};font-weight:bold">{label}</span>'


try:
    headlines = get_latest_headlines(selected_ticker, limit=10)
    if headlines:
        df = pd.DataFrame(headlines)
        # Rename columns for display
        df = df.rename(
            columns={
                "headline": "Headline",
                "label": "Label",
                "score": "Score",
                "price": "Price ($)",
                "analyzed_at": "Analyzed At",
            }
        )
        # Format float columns
        if "Score" in df.columns:
            df["Score"] = df["Score"].apply(lambda x: f"{x:.3f}" if x is not None else "—")
        if "Price ($)" in df.columns:
            df["Price ($)"] = df["Price ($)"].apply(lambda x: f"{x:.2f}" if x is not None else "—")

        display_cols = ["Headline", "Label", "Score", "Price ($)", "Analyzed At"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info(f"No analyzed headlines found for {selected_ticker} yet.")
except Exception as exc:
    logger.error(f"Headlines table error: {exc}")
    st.error(f"Could not load headlines: {exc}")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Monitoring panel
# ---------------------------------------------------------------------------
st.header("4. Monitoring")

m1, m2, m3, m4 = st.columns(4)


def _count_rows(table: str) -> int:
    """Return total row count for a given table name."""
    from database.db import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            return cur.fetchone()[0]


# Total articles collected
try:
    total_articles = _count_rows("articles")
except Exception:
    total_articles = "N/A"

# Total articles analyzed
try:
    total_analyzed = _count_rows("sentiments")
except Exception:
    total_analyzed = "N/A"

# System health
try:
    health_resp = requests.get(f"{ANALYZER_URL}/health", timeout=5)
    health_data = health_resp.json() if health_resp.status_code == 200 else {}
    health_status = health_data.get("status", "unknown").upper()
    queue_size = health_data.get("queue_size", "?")
    health_display = f"{health_status} (queue: {queue_size})"
except Exception:
    health_display = "UNREACHABLE"

# Latest price for selected ticker
try:
    from database.db import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT price FROM prices WHERE ticker = %s ORDER BY fetched_at DESC LIMIT 1;",
                (selected_ticker,),
            )
            price_row = cur.fetchone()
    latest_price = f"${price_row[0]:.2f}" if price_row else "N/A"
except Exception:
    latest_price = "N/A"

with m1:
    st.metric("Total Articles Collected", total_articles)
with m2:
    st.metric("Total Articles Analyzed", total_analyzed)
with m3:
    st.metric("Analyzer Health", health_display)
with m4:
    st.metric(f"{selected_ticker} Latest Price", latest_price)
