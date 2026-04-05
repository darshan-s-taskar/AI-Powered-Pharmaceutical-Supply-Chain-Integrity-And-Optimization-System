import base64
import binascii
import sqlite3
import uuid
from pathlib import Path

from flask import jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from pharmacy_dashboard import create_app
from utils.inference import analyze_medicine_image, load_models


app = create_app()

UPLOAD_DIR = Path(app.static_folder) / "uploads"
DATABASE_PATH = Path(app.instance_path) / "scan_history.db"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


class ImageRequestError(ValueError):
    pass


class PredictionError(RuntimeError):
    pass


class OCRError(RuntimeError):
    pass


def process_barcode_payload(barcode_value):
    cleaned_value = (barcode_value or "").strip()
    if not cleaned_value:
        raise ImageRequestError("Barcode or QR value is required.")

    return {
        "barcode_value": cleaned_value,
        "status": "captured",
        "blockchain_ready": True,
        "verification_stage": "pending_future_blockchain_integration",
    }


def initialize_database():
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                extracted_text TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


PREDICTION_MODELS, PREDICTION_DEVICE = load_models()


def save_request_image():
    if request.files.get("medicine_image"):
        file = request.files["medicine_image"]
        if not file.filename:
            raise ImageRequestError("No image uploaded. Please choose a medicine image first.")

        filename = secure_filename(file.filename)
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise ImageRequestError("Invalid file type. Please upload a PNG, JPG, JPEG, or WEBP image.")

        saved_name = f"{uuid.uuid4().hex}_{filename}"
        image_path = UPLOAD_DIR / saved_name
        file.save(image_path)
        return image_path, saved_name

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image_data", "")

    if not image_data or "," not in image_data:
        raise ImageRequestError("No captured image was received. Please capture an image and try again.")

    _, encoded_data = image_data.split(",", 1)

    try:
        image_bytes = base64.b64decode(encoded_data)
    except (binascii.Error, ValueError) as exc:
        raise ImageRequestError("The captured image could not be processed. Please capture it again.") from exc

    saved_name = f"capture_{uuid.uuid4().hex}.jpg"
    image_path = UPLOAD_DIR / saved_name
    image_path.write_bytes(image_bytes)
    return image_path, saved_name


def run_prediction(image_path):
    try:
        analysis = analyze_medicine_image(
            str(image_path),
            PREDICTION_MODELS,
            PREDICTION_DEVICE,
        )
    except Exception as exc:
        raise PredictionError(
            "Model prediction failed. Please verify the image quality and model setup, then try again."
        ) from exc
    prediction = analysis["label"]
    score = float(analysis["final_score"])
    confidence = score if prediction == "REAL" else 1 - score

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "score": round(float(score), 4),
        "ocr_text": analysis["ocr_text"],
        "detailed_scores": {
            "resnet_score": round(float(analysis["model_scores"]["ResNet50"]), 4),
            "efficientnet_score": round(float(analysis["model_scores"]["EfficientNet-B4"]), 4),
            "vit_score": round(float(analysis["model_scores"]["ViT Base"]), 4),
            "ocr_score": round(float(analysis["text_score"]), 4),
            "ensemble_score": round(float(analysis["final_score"]), 4),
        },
    }


def save_scan_record(image_path, result):
    relative_image_path = f"uploads/{Path(image_path).name}"

    with sqlite3.connect(DATABASE_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO scan_history (image_path, result, confidence, extracted_text)
            VALUES (?, ?, ?, ?)
            """,
            (
                relative_image_path,
                result["prediction"],
                result["confidence"],
                result["ocr_text"],
            ),
        )
        connection.commit()
        scan_id = cursor.lastrowid

    return get_scan_record(scan_id)


def get_scan_record(scan_id):
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, image_path, result, confidence, extracted_text, timestamp FROM scan_history WHERE id = ?",
            (scan_id,),
        ).fetchone()

    return serialize_scan_row(row) if row else None


def fetch_scan_history(limit=None):
    query = """
        SELECT id, image_path, result, confidence, extracted_text, timestamp
        FROM scan_history
        ORDER BY id DESC
    """
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    return [serialize_scan_row(row) for row in rows]


def serialize_scan_row(row):
    return {
        "id": row["id"],
        "image_path": row["image_path"],
        "image_url": url_for("static", filename=row["image_path"]),
        "result": row["result"],
        "confidence": round(float(row["confidence"]), 4),
        "extracted_text": row["extracted_text"] or "",
        "timestamp": row["timestamp"],
    }


@app.route("/history", methods=["GET"])
def history_page():
    return render_template("history.html", scans=fetch_scan_history())


@app.route("/scan-barcode", methods=["POST"])
def scan_barcode():
    payload = request.get_json(silent=True) or {}

    try:
        barcode_result = process_barcode_payload(payload.get("barcode_value"))
    except ImageRequestError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return jsonify({"ok": True, "barcode": barcode_result})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        image_path, saved_name = save_request_image()
        result = run_prediction(image_path)
        saved_record = save_scan_record(image_path, result)
    except ImageRequestError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except OCRError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    except PredictionError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    except FileNotFoundError as exc:
        return jsonify(
            {
                "ok": False,
                "error": f"Model prediction failed because a required model file is missing: {exc}",
            }
        ), 500
    except Exception:
        return jsonify(
            {
                "ok": False,
                "error": "Something went wrong while processing the scan. Please try again.",
            }
        ), 500

    return jsonify(
        {
            "ok": True,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "ocr_text": result["ocr_text"],
            "score": result["score"],
            "detailed_scores": result["detailed_scores"],
            "image_url": url_for("static", filename=f"uploads/{saved_name}"),
            "history_record": saved_record,
        }
    )


initialize_database()


if __name__ == "__main__":
    app.run(debug=True)
