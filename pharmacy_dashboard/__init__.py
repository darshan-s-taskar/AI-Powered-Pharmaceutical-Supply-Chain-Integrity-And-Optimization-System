import os
from pathlib import Path

from flask import Flask

from pharmacy_dashboard.routes import dashboard_bp
from pharmacy_dashboard.services.history_store import HistoryStore
from pharmacy_dashboard.services.predictor import PredictorService


def create_app():
    base_dir = Path(__file__).resolve().parent.parent

    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )

    instance_dir = base_dir / "instance"
    uploads_dir = instance_dir / "uploads"
    database_path = instance_dir / "pharmacy_dashboard.db"

    uploads_dir.mkdir(parents=True, exist_ok=True)

    app.config.update(
        SECRET_KEY=os.environ.get("PHARMA_DASHBOARD_SECRET_KEY", "pharmacy-dashboard-dev"),
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        BASE_DIR=base_dir,
        INSTANCE_DIR=instance_dir,
        UPLOAD_DIR=uploads_dir,
        DATABASE_PATH=database_path,
        ALLOWED_EXTENSIONS={"png", "jpg", "jpeg", "webp"},
        BARCODE_STATUS_DEFAULT="Integration ready",
    )

    history_store = HistoryStore(database_path)
    history_store.initialize()

    app.extensions["predictor_service"] = PredictorService()
    app.extensions["history_store"] = history_store

    app.register_blueprint(dashboard_bp)

    return app
