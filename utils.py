import torch
import matplotlib.pyplot as plt
from PIL import Image
from model_builder import create_resnet18
from data_setup import create_dataloaders
from torchvision import transforms
from pathlib import Path

model = create_resnet18(num_classes=2)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load("best_model.pt", map_location=device, weights_only=True)

# 4. Apply the weights to the model
model.load_state_dict(state_dict)



def predict_single_image(model, image_path, device, class_names):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()
    return predicted_class

train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_dir=Path('./dataset/train/'), 
        val_dir=Path('./dataset/train/'), 
        test_dir=Path('./dataset/train/')
    )

image_path = Path(__file__).parent/ 'dataset/test/Fake/Screenshot 2025-09-17 174308.png'
predict_single_image(model=model, image_path=image_path, device = device, class_names = class_names)