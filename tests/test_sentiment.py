"""Tests for the rule-based sentiment service."""

import pytest

from backend.services.sentiment import SentimentService


@pytest.fixture
def svc():
    return SentimentService()


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("I love this great product", "positive"),
        ("Amazing and fantastic experience", "positive"),
        ("This is terrible and awful", "negative"),
        ("Worst thing ever, I hate it", "negative"),
        ("It is okay, nothing special", "neutral"),
        ("", "neutral"),
    ],
)
def test_analyze_labels(svc, text, expected_label):
    label, score = svc.analyze(text)
    assert label == expected_label
    assert -1.0 <= score <= 1.0


def test_analyze_clamps_score(svc):
    """Extremely positive text stays within [-1, 1]."""
    label, score = svc.analyze("amazing fantastic excellent great wonderful perfect best love love love")
    assert label == "positive"
    assert score <= 1.0
