import torch
import torch.nn.functional as F
import cv2
from utils.ocr import extract_text, validate_text

def ensemble_predict(image, models):
    outputs = []

    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(image)
            prob = F.softmax(out, dim=1)[0][1].item()
            outputs.append(prob)

    return sum(outputs) / len(outputs)

def final_prediction(image_path, models, transform):
    image = transform(cv2.imread(image_path)).unsqueeze(0)

    image_score = ensemble_predict(image, models)

    text = extract_text(image_path)
    text_score = validate_text(text)

    final_score = (0.7 * image_score) + (0.3 * text_score)

    if final_score > 0.6:
        return "REAL", final_score
    else:
        return "FAKE", final_score