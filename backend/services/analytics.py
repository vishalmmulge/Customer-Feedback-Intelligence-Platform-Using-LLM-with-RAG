"""Aggregate analytics over cleaned feedback records.

Pure Pandas computations — no models, no network. Used by the dashboard and
the ``/analytics`` endpoints.
"""

from typing import Any, Dict, List

from ..models.schemas import AnalyticsSummary, FeedbackRecord


def _distribution(values: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in values:
        key = str(v) if v is not None else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


class AnalyticsService:
    def summary(self, records: List[FeedbackRecord]) -> AnalyticsSummary:
        total = len(records)
        ratings = [r.rating for r in records if r.rating is not None]
        avg_rating = (
            round(sum(ratings) / len(ratings), 2) if ratings else None
        )

        return AnalyticsSummary(
            total_feedback=total,
            average_rating=avg_rating,
            sentiment_distribution=_distribution(
                [r.sentiment for r in records]
            ),
            topic_distribution=_distribution([r.topic for r in records]),
            channel_distribution=_distribution([r.channel for r in records]),
        )
