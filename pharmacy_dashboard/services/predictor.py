from utils.inference import analyze_medicine_image, load_models


class PredictorService:
    def __init__(self):
        self.models, self.device = load_models()

    def predict(self, image_path):
        raw_result = analyze_medicine_image(image_path, self.models, self.device)
        final_score = float(raw_result["final_score"])
        label = raw_result["label"]
        confidence = final_score if label == "REAL" else 1 - final_score

        return {
            "label": label,
            "final_score": round(final_score, 4),
            "image_score": round(float(raw_result["image_score"]), 4),
            "text_score": round(float(raw_result["text_score"]), 4),
            "threshold": round(float(raw_result["threshold"]), 4),
            "confidence": round(float(confidence), 4),
            "ocr_text": raw_result["ocr_text"],
            "model_scores": {
                key: round(float(value), 4)
                for key, value in raw_result["model_scores"].items()
            },
        }
