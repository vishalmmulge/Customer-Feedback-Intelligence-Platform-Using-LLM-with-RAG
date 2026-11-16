"""Tests for the rule-based topic classifier."""

import pytest

from backend.services.topic import TopicService


@pytest.fixture
def svc():
    return TopicService()


@pytest.mark.parametrize(
    "text,expected_topic",
    [
        ("The app keeps crashing with an error", "bug_issue"),
        ("I would like a dark mode feature", "feature_request"),
        ("The subscription is too expensive", "pricing"),
        ("The interface is confusing and hard to use", "usability"),
        ("It's slow and laggy, please improve speed", "performance"),
        ("The support agent was helpful", "customer_support"),
        ("Nothing relevant here", "general"),
        ("", "general"),
    ],
)
def test_classify(svc, text, expected_topic):
    assert svc.classify(text) == expected_topic
