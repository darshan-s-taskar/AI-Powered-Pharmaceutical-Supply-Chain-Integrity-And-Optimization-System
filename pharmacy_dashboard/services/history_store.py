import json
import sqlite3
from pathlib import Path


class HistoryStore:
    def __init__(self, database_path):
        self.database_path = Path(database_path)

    def initialize(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    source TEXT NOT NULL,
                    label TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    image_score REAL NOT NULL,
                    text_score REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    ocr_text TEXT,
                    model_scores_json TEXT NOT NULL,
                    barcode_value TEXT,
                    barcode_status TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    def create_scan_record(
        self,
        image_name,
        image_path,
        source,
        prediction,
        barcode_value=None,
        barcode_status=None,
    ):
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO scan_history (
                    image_name, image_path, source, label, final_score, confidence,
                    image_score, text_score, threshold_value, ocr_text,
                    model_scores_json, barcode_value, barcode_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_name,
                    image_path,
                    source,
                    prediction["label"],
                    prediction["final_score"],
                    prediction["confidence"],
                    prediction["image_score"],
                    prediction["text_score"],
                    prediction["threshold"],
                    prediction["ocr_text"],
                    json.dumps(prediction["model_scores"]),
                    barcode_value,
                    barcode_status,
                ),
            )
            connection.commit()
            scan_id = cursor.lastrowid

        return self.get_scan_by_id(scan_id)

    def get_scan_by_id(self, scan_id):
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM scan_history WHERE id = ?",
                (scan_id,),
            ).fetchone()

        return self._serialize_row(row) if row else None

    def get_latest_scan(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM scan_history ORDER BY id DESC LIMIT 1"
            ).fetchone()

        return self._serialize_row(row) if row else None

    def get_recent_scans(self, limit=8):
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM scan_history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._serialize_row(row) for row in rows]

    def get_dashboard_stats(self):
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            summary = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_scans,
                    SUM(CASE WHEN label = 'REAL' THEN 1 ELSE 0 END) AS real_count,
                    SUM(CASE WHEN label = 'FAKE' THEN 1 ELSE 0 END) AS fake_count,
                    AVG(confidence) AS average_confidence
                FROM scan_history
                """
            ).fetchone()

        return {
            "total_scans": summary["total_scans"] or 0,
            "real_count": summary["real_count"] or 0,
            "fake_count": summary["fake_count"] or 0,
            "average_confidence": round(float(summary["average_confidence"] or 0), 4),
        }

    def _serialize_row(self, row):
        model_scores = json.loads(row["model_scores_json"])

        return {
            "id": row["id"],
            "image_name": row["image_name"],
            "image_url": f"/uploads/{row['image_path']}",
            "source": row["source"],
            "label": row["label"],
            "final_score": round(float(row["final_score"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "image_score": round(float(row["image_score"]), 4),
            "text_score": round(float(row["text_score"]), 4),
            "threshold": round(float(row["threshold_value"]), 4),
            "ocr_text": row["ocr_text"] or "",
            "model_scores": {
                key: round(float(value), 4)
                for key, value in model_scores.items()
            },
            "barcode_value": row["barcode_value"] or "",
            "barcode_status": row["barcode_status"] or "Integration ready",
            "created_at": row["created_at"],
        }
