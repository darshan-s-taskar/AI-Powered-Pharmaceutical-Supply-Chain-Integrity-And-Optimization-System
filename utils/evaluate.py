import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def evaluate(model, loader):
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.numpy())
            labels_list.extend(labels.numpy())

    precision = precision_score(labels_list, preds)
    recall = recall_score(labels_list, preds)

    print("Accuracy:", accuracy_score(labels_list, preds))
    print("Confusion Matrix:\n", confusion_matrix(labels_list, preds))
    print("Precision:", precision)
    print("Recall:", recall)