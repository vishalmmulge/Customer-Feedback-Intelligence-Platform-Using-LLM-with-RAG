"""SQLite persistence layer.

Provides a small, dependency-free abstraction over the ``sqlite3`` standard
library. All feedback records are durably stored so that container recreation
does not lose application data.

Table: feedback
    feedback_id TEXT PRIMARY KEY,
    customer_id TEXT,
    feedback_text TEXT NOT NULL,
    source TEXT,
    product TEXT,
    timestamp TEXT,
    rating INTEGER,
    sentiment TEXT,
    sentiment_score REAL,
    topic TEXT,
    processed_at TEXT
"""

import sqlite3
from typing import Any, Dict, List, Optional

from ..models.schemas import FeedbackRecord


class Database:
    def __init__(self, db_path: str = "data/feedback.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        import os

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                customer_id TEXT,
                feedback_text TEXT NOT NULL,
                source TEXT,
                product TEXT,
                timestamp TEXT,
                rating INTEGER,
                sentiment TEXT,
                sentiment_score REAL,
                topic TEXT,
                processed_at TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_feedback_sentiment ON feedback(sentiment)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_feedback_topic ON feedback(topic)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_feedback_product ON feedback(product)"
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def insert_record(self, record: FeedbackRecord, source: str = "csv") -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO feedback
            (feedback_id, customer_id, feedback_text, source, product, timestamp,
             rating, sentiment, sentiment_score, topic, processed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.customer_id,
                record.feedback_text,
                source,
                record.product,
                record.timestamp,
                record.rating,
                record.sentiment,
                record.sentiment_score,
                record.topic,
                record.processed_at,
            ),
        )
        self._conn.commit()

    def insert_records(self, records: List[FeedbackRecord], source: str = "csv") -> int:
        count = 0
        for record in records:
            self.insert_record(record, source)
            count += 1
        return count

    def delete_record(self, feedback_id: str) -> bool:
        assert self._conn is not None
        cur = self._conn.execute(
            "DELETE FROM feedback WHERE feedback_id = ?", (feedback_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_all(self) -> int:
        assert self._conn is not None
        cur = self._conn.execute("DELETE FROM feedback")
        self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def get_record(self, feedback_id: str) -> Optional[FeedbackRecord]:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT * FROM feedback WHERE feedback_id = ?", (feedback_id,)
        )
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_records(
        self,
        page: int = 1,
        page_size: int = 20,
        sentiment: Optional[str] = None,
        topic: Optional[str] = None,
        channel: Optional[str] = None,
        source: Optional[str] = None,
        product: Optional[str] = None,
    ) -> List[FeedbackRecord]:
        assert self._conn is not None
        query = "SELECT * FROM feedback WHERE 1=1"
        params: List[Any] = []

        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if channel:
            # Channel is not a stored column; kept for API compatibility.
            pass
        if source:
            query += " AND source = ?"
            params.append(source)
        if product:
            query += " AND product = ?"
            params.append(product)

        query += " ORDER BY processed_at DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([page_size, (page - 1) * page_size])

        cur = self._conn.execute(query, params)
        return [self._row_to_record(row) for row in cur.fetchall()]

    def count_records(
        self,
        sentiment: Optional[str] = None,
        topic: Optional[str] = None,
        source: Optional[str] = None,
        product: Optional[str] = None,
    ) -> int:
        assert self._conn is not None
        query = "SELECT COUNT(*) FROM feedback WHERE 1=1"
        params: List[Any] = []

        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if source:
            query += " AND source = ?"
            params.append(source)
        if product:
            query += " AND product = ?"
            params.append(product)

        cur = self._conn.execute(query, params)
        return cur.fetchone()[0]

    # ------------------------------------------------------------------ #
    # Analytics aggregations
    # ------------------------------------------------------------------ #
    def sentiment_distribution(self) -> Dict[str, int]:
        return self._distribution("sentiment")

    def topic_distribution(self) -> Dict[str, int]:
        return self._distribution("topic")

    def source_distribution(self) -> Dict[str, int]:
        return self._distribution("source")

    def product_distribution(self) -> Dict[str, int]:
        return self._distribution("product")

    def rating_distribution(self) -> Dict[str, int]:
        return self._distribution("rating")

    def average_rating(self) -> Optional[float]:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL"
        )
        val = cur.fetchone()[0]
        return round(val, 2) if val is not None else None

    def total_count(self) -> int:
        assert self._conn is not None
        cur = self._conn.execute("SELECT COUNT(*) FROM feedback")
        return cur.fetchone()[0]

    def _distribution(self, column: str) -> Dict[str, int]:
        assert self._conn is not None
        cur = self._conn.execute(
            f"SELECT {column}, COUNT(*) FROM feedback GROUP BY {column}"
        )
        return {str(row[0]) if row[0] is not None else "unknown": row[1] for row in cur.fetchall()}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            id=row["feedback_id"],
            customer_id=row["customer_id"] or "",
            feedback_text=row["feedback_text"],
            source=row["source"],
            product=row["product"],
            timestamp=row["timestamp"],
            rating=row["rating"],
            sentiment=row["sentiment"],
            sentiment_score=row["sentiment_score"],
            topic=row["topic"],
            processed_at=row["processed_at"],
        )
