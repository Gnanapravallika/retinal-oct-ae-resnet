"""
Standalone Training Script for Model 2 (MSF + CBAM + ResNet-50)
Trains Model 2 with standard Cross-Entropy Loss to reproduce the 96.17% peak feature score.
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from datasets.transforms import TrustOCTTransforms
from datasets.dataset import RetinalDataset
from datasets.utils import patient_level_split, verify_dataset_pipeline
from models.model2 import get_model2
from engine.trainer import Trainer

def main():
    print("=" * 60)
    print("🚀 TRAINING MODEL 2: MSF + CBAM + RESNET-50 (STANDARD SOFTMAX)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 1. Load Dataset Mapping
    csv_path = "kermany_dataset_mapping.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️ Dataset mapping CSV missing at '{csv_path}'. Running in Colab mode.")
        return
        
    df_all = pd.read_csv(csv_path)
    train_df, val_df, test_df = patient_level_split(df_all)
    
    # 2. Setup Data Transforms & DataLoaders
    train_transform = TrustOCTTransforms(image_size=(224, 224), is_training=True)
    val_transform = TrustOCTTransforms(image_size=(224, 224), is_training=False)
    
    train_ds = RetinalDataset(train_df, transform=train_transform, is_training=True)
    val_ds = RetinalDataset(val_df, transform=val_transform, is_training=False)
    test_ds = RetinalDataset(test_df, transform=val_transform, is_training=False)
    
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=num_workers)
    
    # 3. Instantiate Model 2 (MSF + CBAM + ResNet-50)
    model = get_model2(num_classes=4, pretrained=True).to(device)
    
    # 4. Standard Cross-Entropy Loss & AdamW Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # 5. Initialize Trainer (is_evidential=False for standard Softmax)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=4,
        is_evidential=False
    )
    
    # 6. Fit Model 2
    trainer.fit(epochs=30, save_dir="outputs/msf_cbam_resnet50", patience=8)
    print("\n✅ MODEL 2 TRAINING COMPLETE! Saved to outputs/msf_cbam_resnet50/weights_best.pth")

if __name__ == "__main__":
    main()
