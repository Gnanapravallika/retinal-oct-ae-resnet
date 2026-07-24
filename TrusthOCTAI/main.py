"""
TrusthOCTAI Main Research Pipeline Entrypoint
Orchestrates Dataset Pipeline, Model Building, Training, Evaluation, and Reporting.
"""
import os
import sys
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import from oct_datasets to prevent HuggingFace 'datasets' collision
from oct_datasets.transforms import TrustOCTTransforms
from oct_datasets.dataset import RetinalDataset
from oct_datasets.utils import patient_level_split, verify_dataset_pipeline

def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("TRUSTOCT — RESEARCH LAB PIPELINE")
    print("=" * 60)
    
    # 1. Load Configs
    dataset_cfg = load_yaml_config('configs/dataset.yaml')
    model_cfg = load_yaml_config('configs/model.yaml')
    train_cfg = load_yaml_config('configs/train.yaml')
    
    print(f"[OK] Loaded Configurations:")
    print(f"   - Dataset: {dataset_cfg['dataset']['name']} ({dataset_cfg['dataset']['num_classes']} classes)")
    print(f"   - Model: {model_cfg['model']['name']} (Backbone: {model_cfg['model']['backbone']})")
    print(f"   - Batch Size: {dataset_cfg['dataloader']['batch_size']}, Image Size: {dataset_cfg['image']['size']}")
    
    # 2. Check Dataset Mapping CSV
    csv_path = 'kermany_dataset_mapping.csv'
    if not os.path.exists(csv_path):
        print(f"[INFO] Dataset mapping CSV not found locally at '{csv_path}'. Running in Colab mode.")
        return
        
    df_all = pd.read_csv(csv_path)
    print(f"\n[DATASET SUMMARY] Total Records Loaded: {len(df_all)}")
    
    # 3. Patient-Level Data Splitting
    train_df, val_df, test_df = patient_level_split(df_all)
    print(f"   - Train Split      : {len(train_df)} images ({len(train_df['patient_id'].unique())} patients)")
    print(f"   - Validation Split : {len(val_df)} images ({len(val_df['patient_id'].unique())} patients)")
    print(f"   - Test Split       : {len(test_df)} images ({len(test_df['patient_id'].unique())} patients)")
    
    # 4. Initialize Data Transforms
    train_transform = TrustOCTTransforms(image_size=tuple(dataset_cfg['image']['size']), is_training=True)
    val_transform = TrustOCTTransforms(image_size=tuple(dataset_cfg['image']['size']), is_training=False)
    
    # 5. Create PyTorch Datasets & DataLoaders
    train_ds = RetinalDataset(train_df, transform=train_transform, is_training=True)
    val_ds = RetinalDataset(val_df, transform=val_transform, is_training=False)
    test_ds = RetinalDataset(test_df, transform=val_transform, is_training=False)
    
    batch_size = dataset_cfg['dataloader']['batch_size']
    num_workers = dataset_cfg['dataloader']['num_workers'] if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 6. Verify Dataset Pipeline
    verify_dataset_pipeline(train_loader, num_batches=1)
    
    print("\n[SUCCESS] DATASET PIPELINE FULLY VERIFIED & READY FOR TRAINING!")

if __name__ == '__main__':
    main()
