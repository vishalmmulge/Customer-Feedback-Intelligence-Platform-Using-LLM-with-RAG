"""Tests for the preprocessing service (CSV ingestion, cleaning, validation)."""

import io
import pandas as pd
import pytest

from backend.core.exceptions import ValidationError
from backend.services.preprocessing import PreprocessingService


@pytest.fixture
def svc():
    return PreprocessingService()


def test_clean_drops_empty_feedback(svc):
    """Rows with missing/blank feedback_text are dropped."""
    df = pd.DataFrame(
        {
            "feedback_text": ["great product", "", None, "   ", "another one"],
            "rating": [5, 1, 2, 3, 4],
        }
    )
    cleaned = svc.clean(df)
    assert len(cleaned) == 2
    assert set(cleaned["feedback_text"]) == {"great product", "another one"}


def test_clean_preserves_valid_rows(svc):
    """A fully valid DataFrame passes through unchanged (modulo whitespace)."""
    df = pd.DataFrame(
        {
            "feedback_text": ["  hello world  ", "good"],
            "customer_id": ["c1", "c2"],
            "rating": [5, 4],
        }
    )
    cleaned = svc.clean(df)
    assert len(cleaned) == 2
    assert cleaned["feedback_text"].iloc[0] == "hello world"


def test_validate_missing_required_column(svc):
    """Missing feedback_text raises ValidationError."""
    df = pd.DataFrame({"name": ["a", "b"]})
    with pytest.raises(ValidationError) as exc:
        svc.validate_columns(df)
    assert "feedback_text" in exc.value.message


def test_validate_columns_ok(svc):
    df = pd.DataFrame({"feedback_text": ["x"], "rating": [1]})
    assert svc.validate_columns(df) == []


def test_to_records_maps_fields(svc):
    df = pd.DataFrame(
        {
            "feedback_text": ["nice"],
            "customer_id": ["c1"],
            "product": ["Dashboard"],
            "channel": ["email"],
            "rating": [5],
            "timestamp": ["2024-01-01"],
        }
    )
    cleaned = svc.clean(df)
    records = svc.to_records(cleaned)
    assert len(records) == 1
    assert records[0].feedback_text == "nice"
    assert records[0].rating == 5


def test_process_end_to_end(svc):
    csv_bytes = b"customer_id,product,channel,rating,feedback_text\nc1,Dashboard,email,5,great\nc2,,chat,1,broken\n"
    records, errors = svc.process(io.BytesIO(csv_bytes))
    assert len(records) == 2
    assert records[0].feedback_text == "great"
    assert records[1].feedback_text == "broken"
