"""
Sentiment analysis wrapper.

Primary: HuggingFace Inference API (ProsusAI/finbert) — no local model, no torch.
Fallback: VADER lexicon-based analysis when API is unavailable.
"""

import os
from typing import Optional

import requests
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_hf_api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
_vader: Optional[SentimentIntensityAnalyzer] = None


def load_finbert() -> None:
    """Warm up VADER and verify HF API token is set (non-fatal if missing)."""
    global _vader
    _vader = SentimentIntensityAnalyzer()
    token = os.getenv("HF_API_TOKEN", "")
    if token:
        logger.info("HuggingFace API token found — will use FinBERT via Inference API.")
    else:
        logger.warning("HF_API_TOKEN not set — falling back to VADER for all requests.")
    logger.info("Sentiment analyser ready.")


def _vader_analyze(text: str) -> dict:
    """VADER compound → positive/negative/neutral label."""
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    compound = _vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        label, score = "positive", (compound + 1) / 2
    elif compound <= -0.05:
        label, score = "negative", (1 - compound) / 2
    else:
        label, score = "neutral", 0.5
    return {"label": label, "score": round(score, 4)}


def analyze(text: str) -> dict:
    """
    Run sentiment analysis on a single text string.

    Returns dict with keys:
        label : str   — 'positive', 'negative', or 'neutral'
        score : float — confidence in [0, 1]
    """
    token = os.getenv("HF_API_TOKEN", "")
    if token:
        try:
            resp = requests.post(
                _hf_api_url,
                headers={"Authorization": f"Bearer {token}"},
                json={"inputs": text[:512]},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                # API returns [[{label, score}, ...]] sorted by score desc
                top = data[0][0] if isinstance(data[0], list) else data[0]
                label = top["label"].lower()
                score = float(top["score"])
                logger.debug(f"HF API sentiment: {label} ({score:.4f})")
                return {"label": label, "score": score}
            else:
                logger.warning(f"HF API returned {resp.status_code} — falling back to VADER.")
        except Exception as exc:
            logger.warning(f"HF API error: {exc} — falling back to VADER.")

    result = _vader_analyze(text)
    logger.debug(f"VADER sentiment: {result['label']} ({result['score']:.4f})")
    return result
