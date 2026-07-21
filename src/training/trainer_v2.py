import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from src.models.ae_resnet_v2.model import AEResNetV2
from src.dataset.dataset import RetinalDataset, patient_level_split, auto_detect_columns
from src.preprocessing.standardizer import RetinalPipelineTransform

def print_split_distributions(train_df, val_df, test_df=None):
    class_names = {0: 'AMD', 1: 'DME', 2: 'ERM', 3: 'Normal', 4: 'RAO', 5: 'RVO', 6: 'VID'}
    print("\n--- Split Class Distributions ---")
    print(f"{'Class':<10} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    print("-" * 45)
    for idx, name in class_names.items():
        tr = sum(train_df['label'] == idx)
        vl = sum(val_df['label'] == idx)
        ts = sum(test_df['label'] == idx) if test_df is not None else 0
        print(f"{name:<10} | {tr:<8} | {vl:<8} | {ts:<8}")
    print("-" * 45)

def train_model_v2(model_name: str = "ae-resnet-v2", csv_path: str = None, epochs: int = 40, batch_size: int = 16):
    """
    Main training interface for AE-ResNet v2 supporting dynamic backbone loading.
    """
    if csv_path is None or not os.path.exists(csv_path):
        print(f"Error: Dataset path '{csv_path}' not found.")
        return
        
    raw_df = pd.read_csv(csv_path)
    df = auto_detect_columns(raw_df)
    
    # Correct Windows-format paths dynamically
    drive_base = "/content/drive/MyDrive"
    if len(df) > 0 and not os.path.exists(df.iloc[0]['image_path']):
        def convert_path_to_colab(win_path):
            linux_path = win_path.replace("\\", "/")
            if "OCTDL/" in linux_path:
                relative_path = linux_path[linux_path.find("OCTDL/"):]
            elif "OCTID/" in linux_path:
                relative_path = linux_path[linux_path.find("OCTID/"):]
            else:
                relative_path = "/".join(linux_path.split("/")[-3:])
            return os.path.join(drive_base, relative_path)
        df['image_path'] = df['image_path'].apply(convert_path_to_colab)
        
    train_df, val_df, test_df = patient_level_split(df)
    print_split_distributions(train_df, val_df, test_df)
    
    train_transform = RetinalPipelineTransform(is_training=True)
    val_transform = RetinalPipelineTransform(is_training=False)
    
    train_dataset = RetinalDataset(train_df, transform=train_transform)
    val_dataset = RetinalDataset(val_df, transform=val_transform)
    
    # Calculate class weights robustly for 7 classes to avoid index out of bounds
    class_counts = np.zeros(7)
    for label, count in train_df['label'].value_counts().items():
        class_counts[int(label)] = count
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    class_weights = 1.0 / class_counts
    class_weights = torch.FloatTensor(class_weights)
    
    sample_weights = [class_weights[int(label)] for label in train_df['label'].values]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model dynamically
    model_name_lower = model_name.lower().replace("-", "_")
    if model_name_lower == "resnet50":
        import torchvision.models as models
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 7)
    elif model_name_lower == "densenet121":
        import torchvision.models as models
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 7)
    elif model_name_lower in ["efficientnet_b0", "efficientnet-b0"]:
        import torchvision.models as models
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    elif model_name_lower == "resnet_fixed_fusion":
        model = AEResNetV2(num_classes=7, pretrained=True, use_attention=False, use_adaptive=False)
    elif model_name_lower == "ae_resnet_v1":
        model = AEResNetV2(num_classes=7, pretrained=True, use_attention=True, use_adaptive=False)
    elif model_name_lower == "resnet_amsf":
        model = AEResNetV2(num_classes=7, pretrained=True, use_attention=False, use_adaptive=True)
    elif model_name_lower in ["ae_resnet_v2", "ae_resnet_project"]:
        model = AEResNetV2(num_classes=7, pretrained=True, use_attention=True, use_adaptive=True)
    else:
        raise ValueError(f"Unknown model name configuration: {model_name}")
        
    # If domain-pretraining was completed (Stage 2) and not a baseline, load backbone weights
    baselines = ["resnet50", "densenet121", "efficientnet_b0", "efficientnet-b0"]
    if model_name_lower not in baselines:
        pretrained_backbone_path = "models/ae_resnet_v2_backbone_pretrained.pth"
        if os.path.exists(pretrained_backbone_path):
            print(f"Loading domain-pretrained backbone weights from: {pretrained_backbone_path}")
            model.load_state_dict(torch.load(pretrained_backbone_path), strict=False)
            
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates
    backbone_params = []
    new_layers_params = []
    
    if model_name_lower == "resnet50":
        for name, param in model.named_parameters():
            if 'fc' in name:
                new_layers_params.append(param)
            else:
                backbone_params.append(param)
    elif model_name_lower in ["densenet121", "efficientnet_b0", "efficientnet-b0"]:
        for name, param in model.named_parameters():
            if 'classifier' in name:
                new_layers_params.append(param)
            else:
                backbone_params.append(param)
    else:
        for name, param in model.named_parameters():
            if 'attention' in name or 'classifier' in name or 'fusion' in name or 'projection' in name:
                new_layers_params.append(param)
            else:
                backbone_params.append(param)
                
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': new_layers_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # Cosine Annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Training {model_name} for {epochs} epochs on {device}...")
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    history = []
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        if model_name_lower != "resnet50":
            if epoch <= 3:
                for name, param in model.named_parameters():
                    if not ('attention' in name or 'classifier' in name or 'fusion' in name or 'projection' in name):
                        param.requires_grad = False
                if epoch == 1:
                    print(f"Epoch 1-3 Warm-up: {model_name} Backbone frozen (training new layers only)")
            else:
                for param in model.parameters():
                    param.requires_grad = True
                if epoch == 4:
                    print(f"Epoch 4: {model_name} Backbone unfrozen, full network training")
                
        model.train()
        running_loss, correct, total = 0.0, 0, 0
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
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc,
            'val_f1': epoch_val_f1
        })
        
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {epoch_val_f1:.4f}")
        scheduler.step()
        
        # Early stopping based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), f"models/{model_name_lower}_best.pth")
            print(f"✅ Best model updated! Val Macro F1: {best_val_f1:.4f}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch} due to validation loss plateau.")
            break
            
    # Save training history to CSV
    os.makedirs("results/logs", exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"results/logs/{model_name_lower}_history.csv", index=False)
    print(f"Saved training history to results/logs/{model_name_lower}_history.csv")
    print(f"Training Complete. Best Validation Macro F1 for {model_name}: {best_val_f1:.4f}")
