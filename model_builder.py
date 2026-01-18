import torch
import torchvision.models as models

def create_resnet18(num_classes=2):
    model = models.resnet18(weights='DEFAULT') # Modern equivalent of pretrained=True
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model