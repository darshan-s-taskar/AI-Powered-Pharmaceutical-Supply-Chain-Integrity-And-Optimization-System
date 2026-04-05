import os
import torch
import torch.nn as nn
import torchvision.models as models
import timm
import cv2

from utils.data_loader import get_data_loaders
from utils.train import train_model
from utils.evaluate import evaluate

train_loader, test_loader = get_data_loaders()
# OCR utils
from utils.ocr import extract_text, validate_text

train_loader, test_loader = get_data_loaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL DEFINITIONS
# =========================

# ResNet50
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)

# EfficientNet
efficientnet = timm.create_model('efficientnet_b4', pretrained=True)
efficientnet.classifier = nn.Linear(efficientnet.classifier.in_features, 2)

# ViT
vit = timm.create_model('vit_base_patch16_224', pretrained=True)
vit.head = nn.Linear(vit.head.in_features, 2)

# =========================
# LOAD OR TRAIN MODELS
# =========================

def load_or_train(model, path, name):
    if os.path.exists(path):
        print(f"Loading {name} from {path}")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        print(f"Training {name}")
        model = train_model(model, train_loader)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
    return model

resnet = load_or_train(resnet, "models/resnet50.pth", "ResNet50")
efficientnet = load_or_train(efficientnet, "models/efficientnet.pth", "EfficientNet")
vit = load_or_train(vit, "models/vit.pth", "ViT")

# =========================
# BASIC EVALUATION
# =========================

print("\n--- Individual Model Performance ---")
print("\nResNet :")
evaluate(resnet, test_loader)
print("\nEfficeintNet :")
evaluate(efficientnet, test_loader)
print("\nVision Transformer :")
evaluate(vit, test_loader)

# =========================
# ENSEMBLE + OCR EVALUATION
# =========================

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_image_score(model, image):
    model.eval()
    with torch.no_grad():
        output = model(image)
        prob = torch.softmax(output, dim=1)[0][1].item()
    return prob


y_true = []
y_pred = []

print("\n--- Hybrid + Ensemble Evaluation ---")

for images, labels in test_loader:
    
    for i in range(len(images)):
        image = images[i].unsqueeze(0).to(device)
        label = labels[i].item()

        # -------------------------
        # IMAGE MODEL SCORES
        # -------------------------
        r_score = get_image_score(resnet, image)
        e_score = get_image_score(efficientnet, image)
        v_score = get_image_score(vit, image)

        # Ensemble (image only)
        image_score = (r_score + e_score + v_score) / 3

        # -------------------------
        # OCR SCORE
        # -------------------------
        # Convert tensor to image file (temporary)
        img_np = images[i].permute(1, 2, 0).numpy() * 255
        img_np = img_np.astype('uint8')
        cv2.imwrite("temp.jpg", img_np)

        text = extract_text("temp.jpg")
        text_score = validate_text(text)

        # -------------------------
        # FINAL HYBRID SCORE
        # -------------------------
        final_score = (0.7 * image_score) + (0.3 * text_score)

        pred = 1 if final_score > 0.6 else 0

        y_true.append(label)
        y_pred.append(pred)

# =========================
# FINAL METRICS
# =========================

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("\n--- FINAL HYBRID MODEL PERFORMANCE ---")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))