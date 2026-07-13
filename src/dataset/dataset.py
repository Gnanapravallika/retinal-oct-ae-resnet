import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.preprocessing.standardizer import load_and_preprocess_scan

CLASSES = ["AMD", "DME", "ERM", "NO", "RAO", "RVO", "VID"]
CLASS_TO_IDX = {cls.lower(): idx for idx, cls in enumerate(CLASSES)}

class RetinalDataset(Dataset):
    """
    Custom PyTorch Dataset for loading retinal OCT scans.
    """
    def __init__(self, df: pd.DataFrame, transform=None, apply_bilateral: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.apply_bilateral = apply_bilateral

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['image_path']
        label = int(row['label'])
        
        try:
            img = load_and_preprocess_scan(file_path, apply_bilateral=self.apply_bilateral)
        except Exception:
            # Fallback placeholder image to prevent training crashes
            img = Image.new('RGB', (224, 224), color=128)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def patient_level_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15) -> tuple:
    """
    Enforces strict patient-level split (70:15:15) to prevent training data leakage.
    """
    unique_patients = df['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patients = set(unique_patients[:n_train])
    val_patients = set(unique_patients[n_train:n_train + n_val])
    test_patients = set(unique_patients[n_train + n_val:])
    
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    return train_df, val_df, test_df

def auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically maps dataset CSV columns to the standard format.
    """
    cols = df.columns
    rename_dict = {}
    
    path_col = next((c for c in cols if any(k in c.lower() for k in ["path", "file", "image", "filename"])), None)
    if path_col:
        rename_dict[path_col] = 'image_path'
        
    label_col = next((c for c in cols if any(k in c.lower() for k in ["label", "class", "disease", "category", "target"])), None)
    if label_col:
        rename_dict[label_col] = 'label'
        
    patient_col = next((c for c in cols if any(k in c.lower() for k in ["patient", "subject", "id", "user"])), None)
    if patient_col:
        rename_dict[patient_col] = 'patient_id'
        
    df = df.rename(columns=rename_dict)
    
    for req in ['image_path', 'label', 'patient_id']:
        if req not in df.columns:
            if req == 'patient_id':
                df['patient_id'] = df.index.astype(str)
            else:
                raise ValueError(f"Required column '{req}' could not be detected.")
                
    if df['label'].dtype == object or df['label'].dtype == str:
        df['label'] = df['label'].apply(lambda x: CLASS_TO_IDX.get(str(x).strip().lower(), -1))
        df = df[df['label'] != -1]
        
    return df
