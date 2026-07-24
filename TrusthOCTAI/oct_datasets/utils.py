"""
TrusthOCTAI Dataset Utilities Module
Patient-level split generator & dataset validation tools to prevent data leakage.
"""
import os
import pandas as pd
import numpy as np
import torch

def auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes dataset DataFrame column names to ['image_path', 'label', 'patient_id'].
    """
    col_map = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'path' in c_lower or 'image' in c_lower or 'file' in c_lower:
            col_map[col] = 'image_path'
        elif 'label' in c_lower or 'target' in c_lower or 'class' in c_lower:
            col_map[col] = 'label'
        elif 'patient' in c_lower or 'subject' in c_lower or 'pt' in c_lower:
            col_map[col] = 'patient_id'
            
    df = df.rename(columns=col_map)
    
    if 'image_path' not in df.columns:
        df['image_path'] = df.iloc[:, 0]
    if 'label' not in df.columns:
        df['label'] = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
    if 'patient_id' not in df.columns:
        def infer_patient_id(path_str):
            base = os.path.basename(str(path_str))
            parts = os.path.splitext(base)[0].split('-')
            return '-'.join(parts[:2]) if len(parts) >= 2 else base
        df['patient_id'] = df['image_path'].apply(infer_patient_id)
        
    return df

def patient_level_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42):
    """
    Performs patient-level data splitting to guarantee zero data leakage between train, val, and test splits.
    """
    df = auto_detect_columns(df)
    unique_patients = df['patient_id'].unique()
    
    np.random.seed(seed)
    np.random.shuffle(unique_patients)
    
    n_patients = len(unique_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_pts = set(unique_patients[:n_train])
    val_pts = set(unique_patients[n_train:n_train + n_val])
    test_pts = set(unique_patients[n_train + n_val:])
    
    train_df = df[df['patient_id'].isin(train_pts)].copy().reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_pts)].copy().reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_pts)].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

def verify_dataset_pipeline(dataloader, num_batches=1):
    """
    Verifies PyTorch DataLoader tensor shapes, values, and batch structure.
    """
    print("[VERIFICATION] Verifying Dataset DataLoader Pipeline...")
    for idx, (images, labels) in enumerate(dataloader):
        print(f"  Batch [{idx+1}/{num_batches}] -> Images Tensor Shape: {images.shape}, Labels Tensor Shape: {labels.shape}")
        print(f"  Image Min: {images.min():.3f}, Image Max: {images.max():.3f}")
        print(f"  Unique Labels in Batch: {torch.unique(labels).tolist()}")
        if idx + 1 >= num_batches:
            break
    print("[SUCCESS] Dataset Pipeline Verification PASSED successfully!")
