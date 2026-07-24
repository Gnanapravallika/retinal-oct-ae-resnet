"""
TrusthOCTAI Data Transforms & Augmentation Module
Publication-grade ImageNet-standard preprocessing & anatomical-safe augmentation.
"""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

class TrustOCTTransforms:
    """
    Standardized transform pipeline for Retinal OCT B-scans.
    - Training: Resize (224x224), RGB conversion, Random Horizontal Flip (p=0.5),
                Random Rotation (+/-10 deg), ColorJitter, ImageNet Normalization.
    - Validation/Test: Resize (224x224), RGB conversion, ImageNet Normalization.
    """
    def __init__(self, image_size=(224, 224), is_training=False):
        self.image_size = image_size
        self.is_training = is_training
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        if self.is_training:
            self.transform = T.Compose([
                T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=(-10, 10)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img)
