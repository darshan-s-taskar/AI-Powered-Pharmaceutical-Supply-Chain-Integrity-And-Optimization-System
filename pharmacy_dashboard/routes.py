from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory

from pharmacy_dashboard.services.storage import (
    InvalidImageError,
    save_base64_capture,
    save_uploaded_file,
)


dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/", methods=["GET"])
def index():
    history_store = current_app.extensions["history_store"]

    return render_template(
        "index.html",
        stats=history_store.get_dashboard_stats(),
        history=history_store.get_recent_scans(limit=8),
        latest_result=history_store.get_latest_scan(),
    )


@dashboard_bp.route("/scan", methods=["GET"])
def scan_page():
    history_store = current_app.extensions["history_store"]

    return render_template(
        "scan.html",
        latest_result=history_store.get_latest_scan(),
    )


@dashboard_bp.route("/api/scan", methods=["POST"])
def scan_image():
    predictor = current_app.extensions["predictor_service"]
    history_store = current_app.extensions["history_store"]

    try:
        if request.files.get("medicine_image"):
            image_info = save_uploaded_file(
                file=request.files["medicine_image"],
                upload_dir=current_app.config["UPLOAD_DIR"],
                allowed_extensions=current_app.config["ALLOWED_EXTENSIONS"],
            )
            source = "upload"
            barcode_value = request.form.get("barcode_value")
        else:
            payload = request.get_json(silent=True) or {}
            image_info = save_base64_capture(
                image_data=payload.get("image_data"),
                upload_dir=current_app.config["UPLOAD_DIR"],
            )
            source = "camera"
            barcode_value = payload.get("barcode_value")
    except InvalidImageError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    try:
        result = predictor.predict(str(image_info["path"]))
    except Exception:
        return jsonify(
            {
                "ok": False,
                "error": "Prediction failed. Confirm the trained model weights and OCR runtime are available.",
            }
        ), 500
    scan_record = history_store.create_scan_record(
        image_name=image_info["filename"],
        image_path=image_info["filename"],
        source=source,
        prediction=result,
        barcode_value=barcode_value,
        barcode_status=current_app.config["BARCODE_STATUS_DEFAULT"],
    )

    return jsonify(
        {
            "ok": True,
            "result": scan_record,
            "history": history_store.get_recent_scans(limit=8),
            "stats": history_store.get_dashboard_stats(),
        }
    )


@dashboard_bp.route("/api/history", methods=["GET"])
def scan_history():
    history_store = current_app.extensions["history_store"]
    return jsonify(
        {
            "ok": True,
            "history": history_store.get_recent_scans(limit=20),
            "stats": history_store.get_dashboard_stats(),
        }
    )


@dashboard_bp.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(current_app.config["UPLOAD_DIR"], filename)
