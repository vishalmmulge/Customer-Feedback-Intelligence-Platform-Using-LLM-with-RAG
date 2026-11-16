"""Lightweight rule-based sentiment analysis.

A full transformer model (e.g. via sentence-transformers / HuggingFace) would
require an internet download and a GPU for reasonable speed. For this MVP we
use a transparent, dependency-free lexicon so the pipeline works offline and
the results are easy to audit. Every word carries a small weight; the final
score is clamped to [-1, 1] and mapped to positive / neutral / negative.

This is intentionally simple and explainable — swap in a trained model later
without changing the rest of the pipeline.
"""

import re
from typing import Dict, Tuple

# Small sentiment lexicon with per-word weights. Positive > 0, negative < 0.
_POSITIVE_WORDS: Dict[str, float] = {
    "good": 1.0, "great": 1.0, "excellent": 1.5, "amazing": 1.5, "love": 1.5,
    "happy": 1.0, "satisfied": 1.0, "pleased": 1.0, "impressed": 1.0,
    "awesome": 1.5, "fantastic": 1.5, "wonderful": 1.5, "best": 1.0,
    "perfect": 1.5, "recommend": 1.0, "easy": 0.5, "fast": 0.5,
    "helpful": 1.0, "friendly": 0.5, "smooth": 0.5, "intuitive": 0.5,
    "reliable": 0.5, "value": 0.5, "thanks": 0.5, "thank": 0.5,
    "improved": 0.5, "improvement": 0.5,
}
_NEGATIVE_WORDS: Dict[str, float] = {
    "bad": -1.0, "terrible": -1.5, "awful": -1.5, "horrible": -1.5,
    "hate": -1.5, "worst": -1.5, "broken": -1.0, "error": -0.5, "bug": -0.5,
    "slow": -0.5, "expensive": -0.5, "confusing": -0.5, "difficult": -0.5,
    "hard": -0.5, "disappointed": -1.0, "frustrated": -1.0,
    "frustrating": -1.0, "annoying": -0.5, "waste": -1.0, "useless": -1.0,
    "poor": -0.5, "fail": -0.5, "failed": -0.5, "issue": -0.5,
    "problem": -0.5, "complaint": -1.0, "angry": -1.0, "upset": -1.0,
}

# Thresholds for label assignment.
_POS_THRESHOLD: float = 0.10
_NEG_THRESHOLD: float = -0.10
# Normalisation divisor keeps the score within a readable range.
_NORMALISER: float = 3.0


class SentimentService:
    """Scores feedback text on a [-1, 1] scale and assigns a label."""

    def analyze(self, text: str) -> Tuple[str, float]:
        """Return ``(label, score)`` where label in {positive, neutral, negative}."""
        if not text:
            return "neutral", 0.0

        words = re.findall(r"\w+", text.lower())
        score = 0.0
        for word in words:
            score += _POSITIVE_WORDS.get(word, 0.0)
            score += _NEGATIVE_WORDS.get(word, 0.0)

        # Clamp to [-1, 1] for stability.
        score = max(-1.0, min(1.0, score / _NORMALISER))

        if score > _POS_THRESHOLD:
            label = "positive"
        elif score < _NEG_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return label, round(score, 3)
