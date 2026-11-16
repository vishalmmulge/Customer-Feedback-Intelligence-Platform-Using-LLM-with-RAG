"""Analytics API routes: overview, sentiment, topics, trends, and summary.

All aggregations are computed directly from SQLite so no statistics are
hardcoded or fabricated.
"""

from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request

from ..core.database import Database
from ..core.logging import logger
from ..models.schemas import AnalyticsSummary

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _get_db(request: Request) -> Database:
    return request.app.state.database


def _bucket_date(timestamp: Optional[str]) -> str:
    """Bucket a timestamp ISO string into a YYYY-MM-DD date for trends."""
    if not timestamp:
        return "unknown"
    try:
        return timestamp[:10]
    except Exception:
        return "unknown"


@router.get("/summary", response_model=AnalyticsSummary)
def analytics_summary(request: Request):
    """Aggregate sentiment, topic, source, product, and rating statistics."""
    db: Database = _get_db(request)
    summary = AnalyticsSummary(
        total_feedback=db.total_count(),
        average_rating=db.average_rating(),
        sentiment_distribution=db.sentiment_distribution(),
        topic_distribution=db.topic_distribution(),
        source_distribution=db.source_distribution(),
        product_distribution=db.product_distribution(),
        rating_distribution=db.rating_distribution(),
    )
    logger.debug("Computed analytics summary")
    return summary


@router.get("/overview")
def analytics_overview(request: Request):
    """High-level overview metrics for the dashboard."""
    db: Database = _get_db(request)
    return {
        "total_feedback": db.total_count(),
        "average_rating": db.average_rating(),
        "sentiment_distribution": db.sentiment_distribution(),
        "topic_distribution": db.topic_distribution(),
        "top_topics": sorted(
            db.topic_distribution().items(), key=lambda x: x[1], reverse=True
        )[:10],
    }


@router.get("/sentiment")
def analytics_sentiment(request: Request):
    """Sentiment distribution with percentages."""
    db: Database = _get_db(request)
    dist = db.sentiment_distribution()
    total = sum(dist.values()) or 1
    return {
        "distribution": dist,
        "percentages": {
            k: round(v / total * 100, 2) for k, v in dist.items()
        },
        "total": total,
    }


@router.get("/topics")
def analytics_topics(request: Request):
    """Topic distribution with percentages."""
    db: Database = _get_db(request)
    dist = db.topic_distribution()
    total = sum(dist.values()) or 1
    return {
        "distribution": dist,
        "percentages": {
            k: round(v / total * 100, 2) for k, v in dist.items()
        },
        "total": total,
    }


@router.get("/trends")
def analytics_trends(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
):
    """Feedback volume trend over time (buckets by day)."""
    db: Database = _get_db(request)
    # Load all records and bucket by date.
    records = db.list_records(page=1, page_size=100_000)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

    daily: Dict[str, int] = {}
    daily_sentiment: Dict[str, Dict[str, int]] = {}

    for r in records:
        day = _bucket_date(r.timestamp or r.processed_at)
        if day == "unknown":
            continue
        if day < cutoff:
            continue
        daily[day] = daily.get(day, 0) + 1
        sent = r.sentiment or "unknown"
        daily_sentiment.setdefault(day, {})
        daily_sentiment[day][sent] = daily_sentiment[day].get(sent, 0) + 1

    # Fill in missing days with zero.
    trend = []
    try:
        start = datetime.strptime(cutoff, "%Y-%m-%d")
        end = datetime.now(timezone.utc).date()
        current = start
        while current.date() <= end:
            d = current.date().isoformat()
            trend.append(
                {
                    "date": d,
                    "count": daily.get(d, 0),
                    "sentiment": daily_sentiment.get(d, {}),
                }
            )
            current += timedelta(days=1)
    except Exception:
        trend = [{"date": d, "count": c, "sentiment": daily_sentiment.get(d, {})} for d, c in sorted(daily.items())]

    return {"days": days, "trend": trend}
