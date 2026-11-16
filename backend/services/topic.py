"""Rule-based topic classification.

Maps feedback text to a topic bucket using keyword matching. The topic map is
explicit and editable, which makes the output explainable to stakeholders
(important for customer-feedback use cases). ``general`` is the fallback when
no category keywords match.

Future work: replace with zero-shot classification or a trained classifier.
"""

from typing import Dict

# topic -> keywords. Matching is case-insensitive substring search.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "bug_issue": [
        "bug", "error", "crash", "broken", "issue", "problem",
        "doesn't work", "not working", "fail", "failure", "glitch",
    ],
    "feature_request": [
        "feature", "request", "would like", "wish", "add",
        "improve", "enhancement", "suggestion", "missing",
    ],
    "pricing": [
        "price", "pricing", "expensive", "cost", "billing",
        "subscription", "charge", "fee", "refund", "overpriced",
    ],
    "usability": [
        "easy", "difficult", "hard", "confusing", "intuitive",
        "usability", "interface", "ui", "ux", "navigation",
    ],
    "performance": [
        "slow", "fast", "speed", "lag", "performance", "loading",
        "delay", "sluggish", "latency",
    ],
    "customer_support": [
        "support", "help", "agent", "representative", "contact",
        "service", "response", "reply", "team",
    ],
}

FALLBACK_TOPIC: str = "general"


class TopicService:
    """Classifies a feedback string into one topic bucket."""

    def classify(self, text: str) -> str:
        if not text:
            return FALLBACK_TOPIC

        text_lower = text.lower()
        best_topic = FALLBACK_TOPIC
        best_score = 0

        for topic, keywords in TOPIC_KEYWORDS.items():
            if topic == FALLBACK_TOPIC:
                continue
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic
