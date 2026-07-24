"""
TrusthOCTAI PyTorch Dataset Module
Modular dataset class supporting Kermany OCT2017 and external OCTID benchmarks.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class RetinalDataset(Dataset):
    """
    PyTorch Dataset for Retinal OCT B-scans.
    Supports patient-level split dataframes and dynamic path resolution across Colab & Local environments.
    """
    def __init__(self, df: pd.DataFrame, transform=None, is_training=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_training = is_training
        
        self.path_col = 'image_path' if 'image_path' in self.df.columns else self.df.columns[0]
        
        possible_label_cols = ['label', 'target', 'class', 'category', 'disease']
        self.label_col = None
        for col in possible_label_cols:
            if col in self.df.columns:
                self.label_col = col
                break
        if self.label_col is None:
            self.label_col = self.df.columns[1] if len(self.df.columns) > 1 else self.df.columns[0]

        self.class_map = {
            'cnv': 0, 'dme': 1, 'drusen': 2, 'normal': 3,
            'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3,
            '0': 0, '1': 1, '2': 2, '3': 3, 0: 0, 1: 1, 2: 2, 3: 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row[self.path_col])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        if self.transform is not None:
            image = self.transform(image)
            
        raw_label = row[self.label_col]
        label = self.class_map.get(str(raw_label).lower().strip(), self.class_map.get(raw_label, 0))
        
        return image, torch.tensor(label, dtype=torch.long)
