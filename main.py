import torch
import os
from pathlib import Path
import data_setup, model_builder, engine, utils

# Configuration
URL = "https://drive.google.com/uc?id=1LQCQDrRtfGgBB-bJxMtbBcJZwH9JiNDd"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10

def main():
    # 1. Setup Data
    # data_setup.download_and_extract(URL)
    train_loader, val_loader, test_loader, class_names = data_setup.create_dataloaders(
        train_dir=Path('dataset/train'), 
        val_dir=Path('dataset/val'), 
        test_dir=Path('/dataset/test')
    )

    # 2. Build Model
    model = model_builder.create_resnet18(num_classes=len(class_names)).to(DEVICE)
    
    # 3. Define Loss & Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 4. Train
    engine.train_step(model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)

    # 5. Evaluate
    engine.evaluate(model, test_loader, DEVICE)

    # 6. Save
    save_path = Path("models")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "best_model.pt")
    print("Model Saved!")

if __name__ == "__main__":
    main()