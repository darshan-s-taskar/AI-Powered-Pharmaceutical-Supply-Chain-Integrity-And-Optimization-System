import base64
import binascii
import imghdr
import uuid
from pathlib import Path

from werkzeug.utils import secure_filename


class InvalidImageError(ValueError):
    pass


def save_uploaded_file(file, upload_dir, allowed_extensions):
    if not file or not file.filename:
        raise InvalidImageError("Please choose a medicine image to analyze.")

    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in allowed_extensions:
        raise InvalidImageError("Upload a PNG, JPG, JPEG, or WEBP image.")

    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = Path(upload_dir) / unique_name
    file.save(save_path)

    return {"filename": unique_name, "path": save_path}


def save_base64_capture(image_data, upload_dir):
    if not image_data or "," not in image_data:
        raise InvalidImageError("Camera capture data is missing or invalid.")

    _, encoded = image_data.split(",", 1)

    try:
        binary = base64.b64decode(encoded)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("Could not decode the captured camera image.") from exc

    image_type = imghdr.what(None, binary)
    extension = "jpg" if image_type == "jpeg" else image_type

    if extension not in {"png", "jpg", "webp"}:
        raise InvalidImageError("Captured image format is not supported.")

    filename = f"camera_{uuid.uuid4().hex}.{extension}"
    save_path = Path(upload_dir) / filename
    save_path.write_bytes(binary)

    return {"filename": filename, "path": save_path}
