"""
TrusthOCTAI Training Engine
Handles model training, mixed precision autocast, early stopping, and checkpoint management.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.edl_head import get_evidence_metrics

class Trainer:
    """
    Modular Trainer for TrustOCT network architectures.
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 criterion: nn.Module, optimizer: optim.Optimizer, scheduler, device: torch.device,
                 num_classes: int = 4, is_evidential: bool = True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.is_evidential = is_evidential

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_kl = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            if self.is_evidential:
                try:
                    loss, mse_v, kl_v = self.criterion(outputs, targets, epoch, return_components=True)
                    running_mse += mse_v.item() * inputs.size(0)
                    running_kl += kl_v.item() * inputs.size(0)
                except Exception:
                    loss = self.criterion(outputs, targets, epoch)
            else:
                loss = self.criterion(outputs, targets)
                
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            if self.is_evidential:
                probs, _ = get_evidence_metrics(outputs)
                preds = torch.argmax(probs, dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)
                
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_mse = running_mse / total if self.is_evidential else epoch_loss
        epoch_kl = running_kl / total if self.is_evidential else 0.0
        return epoch_loss, epoch_acc, epoch_mse, epoch_kl

    @torch.no_grad()
    def validate_epoch(self, epoch: int):
        self.model.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_kl = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            
            if self.is_evidential:
                try:
                    loss, mse_v, kl_v = self.criterion(outputs, targets, epoch, return_components=True)
                    running_mse += mse_v.item() * inputs.size(0)
                    running_kl += kl_v.item() * inputs.size(0)
                except Exception:
                    loss = self.criterion(outputs, targets, epoch)
            else:
                loss = self.criterion(outputs, targets)
                
            running_loss += loss.item() * inputs.size(0)
            
            if self.is_evidential:
                probs, _ = get_evidence_metrics(outputs)
                preds = torch.argmax(probs, dim=1)
            else:
                preds = torch.argmax(outputs, dim=1)
                
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        epoch_mse = running_mse / total if self.is_evidential else epoch_loss
        epoch_kl = running_kl / total if self.is_evidential else 0.0
        return epoch_loss, epoch_acc, epoch_mse, epoch_kl

    def fit(self, epochs: int = 30, save_dir: str = "outputs/trustoct", patience: int = 8):
        os.makedirs(save_dir, exist_ok=True)
        best_metric = float("inf")
        patience_counter = 0
        
        print(f"🚀 Training TrustOCT for {epochs} epochs on device: {self.device}")
        for epoch in range(epochs):
            tr_loss, tr_acc, tr_mse, tr_kl = self.train_epoch(epoch)
            val_loss, val_acc, val_mse, val_kl = self.validate_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train Acc: {tr_acc:.4f} (Loss: {tr_loss:.4f}) | Val Acc: {val_acc:.4f} (MSE: {val_mse:.4f})")
            
            # Monitor val_mse for evidential models to avoid KL annealing metric distortion
            monitor_metric = val_mse if self.is_evidential else val_loss
            if monitor_metric < best_metric:
                best_metric = monitor_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(save_dir, "weights_best.pth"))
                print(f"  [BEST] Saved checkpoint to {save_dir}/weights_best.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [EARLY STOP] Triggered at epoch {epoch+1}")
                    break
