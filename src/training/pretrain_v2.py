import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from src.models.ae_resnet_v2.model import AEResNetV2

class OCT2017Dataset(Dataset):
    """
    Custom Dataset loader for Kermany et al. (OCT2017) dataset.
    Assumes standard folder structure: root/class_name/image_name.jpeg
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
        self.samples = []

        for class_name, class_idx in self.class_to_idx.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append((os.path.join(class_folder, img_name), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def pretrain_on_oct2017(dataset_path: str, epochs: int = 10, batch_size: int = 32, save_path: str = "models/ae_resnet_v2_backbone_pretrained.pth"):
    """
    Pretrains the AE-ResNet v2 model on the massive OCT2017 dataset (4 classes).
    Saves ONLY the backbone feature extraction weights, discarding the 4-class head.
    """
    print("\n=== Stage 2: Starting OCT2017 Domain Pretraining (AE-ResNet v2) ===")
    
    # Standard normalization and training augmentation for OCT2017
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found. Skipping pretraining.")
        return

    train_dataset = OCT2017Dataset(dataset_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(train_dataset)} images from OCT2017. Training on {device}...")

    # Load AE-ResNet V2 with 4 classification channels
    model = AEResNetV2(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, correct, total = 0.0, 0, 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        elapsed = time.time() - start_time
        print(f"Pretrain Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Time: {elapsed:.1f}s")
        scheduler.step()

    # Extract backbone state dictionary (everything except the classification head)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    backbone_state = {k: v for k, v in model.state_dict().items() if not k.startswith('classifier')}
    torch.save(backbone_state, save_path)
    print(f"\n✅ Pretraining complete! AE-ResNet v2 backbone weights saved to: {save_path}")
    print("============================================================\n")
