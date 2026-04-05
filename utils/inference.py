import os

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

import timm

from utils.ocr import extract_text, validate_text


MODEL_CONFIG = [
    ("ResNet50", "models/resnet50.pth", "resnet"),
    ("EfficientNet-B4", "models/efficientnet.pth", "efficientnet"),
    ("ViT Base", "models/vit.pth", "vit"),
]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_models():
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)

    efficientnet = timm.create_model("efficientnet_b4", pretrained=True)
    efficientnet.classifier = nn.Linear(efficientnet.classifier.in_features, 2)

    vit = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit.head = nn.Linear(vit.head.in_features, 2)

    return {
        "resnet": resnet,
        "efficientnet": efficientnet,
        "vit": vit,
    }


def load_models():
    device = get_device()
    model_map = build_models()

    for display_name, weight_path, key in MODEL_CONFIG:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Missing weights for {display_name}: {weight_path}. "
                "Train the models first so the UI can use them."
            )

        model = model_map[key]
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    return model_map, device


def get_inference_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def predict_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
    return probs[1].item()


def analyze_medicine_image(image_path, model_map, device, threshold=0.6):
    transform = get_inference_transform()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model_scores = {
        "ResNet50": predict_image(model_map["resnet"], image_tensor),
        "EfficientNet-B4": predict_image(model_map["efficientnet"], image_tensor),
        "ViT Base": predict_image(model_map["vit"], image_tensor),
    }

    image_score = sum(model_scores.values()) / len(model_scores)

    extracted_text = extract_text(image_path)
    text_score = validate_text(extracted_text)
    final_score = (0.7 * image_score) + (0.3 * text_score)

    label = "REAL" if final_score >= threshold else "FAKE"

    return {
        "label": label,
        "final_score": final_score,
        "image_score": image_score,
        "text_score": text_score,
        "threshold": threshold,
        "ocr_text": extracted_text.strip(),
        "model_scores": model_scores,
    }
