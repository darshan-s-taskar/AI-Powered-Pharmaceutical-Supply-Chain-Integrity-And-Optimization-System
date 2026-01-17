import torch
import torch.nn as nn
from torchvision import models

class FakeMedicineDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(FakeMedicineDetector, self).__init__()
        
        # 1. Load Pretrained ResNet50
        # include_top=False in Keras is equivalent to taking everything but self.fc in PyTorch
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 2. Freeze the base model (Keras: base_model.trainable = False)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 3. Modify the Head
        # ResNet50's last layer is named 'fc'. Its input features are 2048.
        num_features = self.base_model.fc.in_features
        
        # We replace 'fc' with our custom layers
        # GlobalAveragePooling2D is built-in to ResNet50's forward pass before the 'fc' layer
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3), # Added dropout for better generalization in your app
            nn.Linear(1024, 1),
            nn.Sigmoid()     # Sigmoid for binary classification (0: Fake, 1: Real)
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FakeMedicineDetector().to(device)

# 4. Compile (Loss & Optimizer)
criterion = nn.BCELoss() # binary_crossentropy
optimizer = torch.optim.Adam(model.base_model.fc.parameters(), lr=0.001)