"""
FinBERT sentiment model wrapper.

Uses ProsusAI/finbert from Hugging Face Transformers.
Provides a module-level singleton loaded via load_finbert().
"""

from typing import Optional

from loguru import logger
from transformers import pipeline, Pipeline

_finbert_pipeline: Optional[Pipeline] = None


def load_finbert() -> Pipeline:
    """
    Load the ProsusAI/finbert sentiment pipeline and cache it as a module singleton.

    Subsequent calls return the already-loaded pipeline without reloading.

    Returns
    -------
    transformers.Pipeline — text-classification pipeline ready for inference
    """
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info("Loading mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis — this may take a moment on first run.")
        try:
            _finbert_pipeline = pipeline(
                "text-classification",
                model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
                top_k=1,
            )
            logger.info("Sentiment model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load FinBERT: {exc}")
            raise
    return _finbert_pipeline


def analyze(text: str) -> dict:
    """
    Run FinBERT sentiment analysis on a single text string.

    Parameters
    ----------
    text : the headline or news text to score

    Returns
    -------
    dict with keys:
        label : str  — 'positive', 'negative', or 'neutral'
        score : float — confidence score in [0, 1]

    Raises RuntimeError if the model has not been loaded yet.
    """
    if _finbert_pipeline is None:
        raise RuntimeError("FinBERT model is not loaded. Call load_finbert() first.")

    try:
        # pipeline returns [[{"label": "...", "score": ...}]] when top_k=1
        result = _finbert_pipeline(text[:512])  # FinBERT max tokens ~512
        top = result[0][0] if isinstance(result[0], list) else result[0]
        label = top["label"].lower()
        score = float(top["score"])
        logger.debug(f"Sentiment: {label} ({score:.4f}) for text: {text[:60]}...")
        return {"label": label, "score": score}
    except Exception as exc:
        logger.error(f"analyze() failed: {exc}")
        raise
