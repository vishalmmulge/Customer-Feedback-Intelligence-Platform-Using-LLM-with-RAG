"""CSV ingestion, cleaning, validation, and duplicate detection.

The preprocessing service is the first stage of the pipeline. It:

1. Reads a CSV upload into a Pandas DataFrame.
2. **Auto-detects** which column is the feedback text (and optional columns
   like customer_id, rating, timestamp, product, channel, source) by scanning
   column names for common keywords. No manual mapping required.
3. Validates that usable feedback text is present.
4. Cleans whitespace / missing values and drops rows with no feedback text.
5. Detects and removes duplicate feedback text within the upload.
6. Assigns unique IDs, source tags, and processing timestamps.
7. Converts clean rows into :class:`FeedbackRecord` domain objects.

Nothing here requires an external model or network call, so it is fast and
fully testable offline.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core.exceptions import ValidationError
from ..core.logging import logger
from ..models.schemas import FeedbackRecord

# Internal schema -> list of keyword hints to look for in CSV column names.
# Order matters: more specific keywords are checked first.
COLUMN_HINTS: Dict[str, List[str]] = {
    "feedback_text": [
        "feedback_text", "feedback text", "feedback", "text", "comment",
        "comments", "review", "reviews", "message", "messages", "content",
        "description", "detail", "details", "issue", "issues", "complaint",
        "complaints", "suggestion", "suggestions", "response", "responses",
        "body", "note", "notes", "remark", "remarks",
    ],
    "customer_id": [
        "customer_id", "customer id", "customer", "user_id", "user id",
        "userid", "user", "client_id", "client id", "client", "account_id",
        "account id", "account", "respondent_id", "respondent",
    ],
    "rating": [
        "rating", "ratings", "score", "scores", "stars", "star", "value",
        "values", "nps", "sentiment_score",
    ],
    "timestamp": [
        "timestamp", "time stamp", "date", "date/time", "datetime",
        "created_at", "created at", "submitted_at", "submitted at",
        "date_time", "date time", "time", "created", "submitted",
    ],
    "product": [
        "product", "products", "item", "items", "app", "application",
        "service", "services", "feature", "features", "platform",
    ],
    "channel": [
        "channel", "channels", "source", "sources", "platform", "platforms",
        "medium", "mediums", "origin", "origins", "via",
    ],
    "source": [
        "source", "sources", "origin", "origins", "channel", "channels",
        "platform", "platforms",
    ],
}

# Fallback column used when nothing matches for feedback text.
FEEDBACK_TEXT_COLUMN = "feedback_text"

# Default channel / source used when the upload does not provide one.
DEFAULT_CHANNEL: str = "unknown"
DEFAULT_SOURCE: str = "csv"


class PreprocessingService:
    """Validates and cleans raw CSV feedback into structured records.

    Automatically detects column names so users never need to provide a
    manual mapping. Works with any CSV that contains a column of feedback
    text, regardless of what that column is called.
    """

    def load_csv(self, file_path_or_buffer) -> pd.DataFrame:
        """Read a CSV from a path or file-like object into a DataFrame."""
        try:
            df = pd.read_csv(file_path_or_buffer)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValidationError(f"Failed to read CSV: {exc}")
        return df

    def _detect_column(
        self, df_columns: List[str], target: str
    ) -> Optional[str]:
        """Find the CSV column that best matches ``target``.

        Returns the original CSV column name, or ``None`` if no match.
        Matching is case-insensitive and checks for keyword containment.
        """
        lower_cols = {c.lower().strip(): c for c in df_columns}
        hints = COLUMN_HINTS.get(target, [target])

        # Exact match first.
        for hint in hints:
            if hint.lower().strip() in lower_cols:
                return lower_cols[hint.lower().strip()]

        # Substring / keyword match second.
        for hint in hints:
            hint_lower = hint.lower().strip()
            for col_lower, col_orig in lower_cols.items():
                if hint_lower in col_lower or col_lower in hint_lower:
                    return col_orig

        return None

    def auto_map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Return a mapping of detected CSV columns to internal schema names.

        Only columns that exist in the DataFrame and match a hint are
        included. Already-mapped (exact name match) columns are skipped.
        """
        mapping: Dict[str, str] = {}
        used_targets: set = set()

        for target in COLUMN_HINTS:
            detected = self._detect_column(list(df.columns), target)
            if detected is None:
                continue
            if detected == target:
                # Already correctly named; no mapping needed.
                used_targets.add(target)
                continue
            if target in used_targets:
                # Don't overwrite an already-assigned target.
                continue
            mapping[detected] = target
            used_targets.add(target)

        return mapping

    def validate_columns(self, df: pd.DataFrame) -> List[str]:
        """Return a list of missing required columns (empty if valid)."""
        missing = [c for c in ["feedback_text"] if c not in df.columns]
        if missing:
            raise ValidationError(
                f"Missing required column(s): {', '.join(missing)}",
                details={"missing": missing, "present": list(df.columns)},
            )
        return missing

    @staticmethod
    def _feedback_hash(text: str) -> str:
        """Stable hash of normalized feedback text for duplicate detection."""
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw DataFrame and return a normalized copy."""
        df = df.copy()

        # Normalize string columns: strip whitespace, coerce nan/None/blank to None.
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "str":
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({"nan": None, "None": None})
                df[col] = df[col].mask(df[col] == "", None)
                df[col] = df[col].astype(object)

        before = len(df)

        # Drop rows without usable feedback text.
        ft = df["feedback_text"].astype(object)
        df = df[~(ft.fillna("").str.strip() == "")]

        # Duplicate detection on normalized feedback text.
        df["_feedback_hash"] = df["feedback_text"].astype(object).apply(self._feedback_hash)
        before_dedup = len(df)
        df = df.drop_duplicates(subset="_feedback_hash", keep="first")
        dupes = before_dedup - len(df)
        if dupes:
            logger.info(f"Dropped {dupes} duplicate feedback row(s)")

        after = len(df)
        dropped = before - after
        if dropped:
            logger.info(f"Dropped {dropped} row(s) with empty/missing feedback_text")

        # Apply defaults for optional columns.
        now_iso = datetime.now(timezone.utc).isoformat()

        if "customer_id" not in df.columns or df["customer_id"].isna().all():
            import uuid

            df["customer_id"] = [f"cust-{uuid.uuid4().hex[:8]}" for _ in range(len(df))]
        else:
            import uuid

            df["customer_id"] = df["customer_id"].fillna(
                lambda _: f"cust-{uuid.uuid4().hex[:8]}"
            )

        for col, default in (
            ("product", None),
            ("channel", DEFAULT_CHANNEL),
            ("source", DEFAULT_SOURCE),
            ("timestamp", None),
        ):
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)
                df[col] = df[col].astype(object)

        # Coerce rating to nullable integer where possible.
        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")

        # Assign processing timestamp and unique IDs.
        df["processed_at"] = now_iso
        import uuid

        df["feedback_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Drop helper column.
        df = df.drop(columns=["_feedback_hash"])

        return df.reset_index(drop=True)

    def to_records(self, df: pd.DataFrame) -> List[FeedbackRecord]:
        """Convert a cleaned DataFrame into a list of domain records."""
        records: List[FeedbackRecord] = []
        for _, row in df.iterrows():
            rating = None
            if "rating" in df.columns and pd.notna(row.get("rating")):
                try:
                    rating = int(row["rating"])
                except (ValueError, TypeError):
                    rating = None

            records.append(
                FeedbackRecord(
                    id=str(row.get("feedback_id") or str(uuid4())),
                    feedback_id=str(row.get("feedback_id") or str(uuid4())),
                    customer_id=str(row.get("customer_id") or ""),
                    feedback_text=str(row["feedback_text"]),
                    source=self._opt_str(row.get("source"), DEFAULT_SOURCE),
                    product=self._opt_str(row.get("product")),
                    channel=self._opt_str(row.get("channel"), DEFAULT_CHANNEL),
                    rating=rating,
                    timestamp=self._opt_str(row.get("timestamp")),
                    processed_at=self._opt_str(row.get("processed_at")),
                )
            )
        return records

    @staticmethod
    def _opt_str(value: Any, default: Optional[str] = None) -> Optional[str]:
        """Coerce a cell to a string, treating nan/None/blank as missing."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return str(value)

    def process(
        self,
        file_path_or_buffer,
        source: str = DEFAULT_SOURCE,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[FeedbackRecord], List[Dict[str, Any]]]:
        """End-to-end: load, validate, clean, convert.

        **Auto-detection:** If no ``column_mapping`` is provided (or it is
        empty), the service automatically scans the CSV column names and
        maps them to the internal schema (``feedback_text``, ``customer_id``,
        ``rating``, ``timestamp``, ``product``, ``channel``, ``source``).

        Returns the accepted records plus any row-level errors encountered.
        """
        df = self.load_csv(file_path_or_buffer)

        # Apply explicit column mapping first, if provided.
        if column_mapping:
            rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_map:
                df = df.rename(columns=rename_map)
        else:
            # Auto-detect columns: map CSV names -> internal schema names.
            auto_map = self.auto_map_columns(df)
            if auto_map:
                df = df.rename(columns=auto_map)
                logger.info(f"Auto-mapped columns: {auto_map}")

        self.validate_columns(df)
        cleaned = self.clean(df)
        # Override source column with the explicit source argument.
        cleaned["source"] = source
        records = self.to_records(cleaned)
        return records, []


# Re-export uuid4 for tests that reference it directly.
from uuid import uuid4  # noqa: E402
