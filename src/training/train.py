import sys
import os
import time
import csv
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__)) 
src_dir = os.path.dirname(current_dir)                   
if src_dir not in sys.path:
    sys.path.append(src_dir)

from model.model import ResUnet3D
from preprocessing.dataset import BratsDataset
from configs import config  

class RegionWiseDiceLoss(nn.Module):
    """
    Calculates Dice Loss for specific clinical tumor regions:
    1. Whole Tumor (WT): Labels 1 + 2 + 3
    2. Tumor Core (TC): Labels 1 + 3
    3. Enhancing Tumor (ET): Label 3
    
    Supports Deep Supervision by accepting a list of inputs.
    """
    def __init__(self):
        super(RegionWiseDiceLoss, self).__init__()
        self.smooth = 1e-5

    def _dice_loss(self, inputs, targets):
        """Helper to compute Dice loss for a single binary region."""
        # Sum over spatial dimensions (Depth, Height, Width)
        intersection = (inputs * targets).sum(dim=(1, 2, 3))
        union = inputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward_single(self, inputs, targets):
        """Calculates combined loss for one resolution level."""
        # 1. Get Softmax Probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # 2. Convert targets to One-Hot encoding
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=4).permute(0, 4, 1, 2, 3).float()
        
        # --- DEFINE REGIONS ---
        # WT: Class 1 (NCR) + 2 (ED) + 3 (ET)
        pred_wt = probs[:, 1, ...] + probs[:, 2, ...] + probs[:, 3, ...]
        target_wt = targets_one_hot[:, 1, ...] + targets_one_hot[:, 2, ...] + targets_one_hot[:, 3, ...]
        
        # TC: Class 1 (NCR) + 3 (ET)
        pred_tc = probs[:, 1, ...] + probs[:, 3, ...]
        target_tc = targets_one_hot[:, 1, ...] + targets_one_hot[:, 3, ...]
        
        # ET: Class 3 (ET)
        pred_et = probs[:, 3, ...]
        target_et = targets_one_hot[:, 3, ...]
        
        # --- CALCULATE LOSSES ---
        loss_wt = self._dice_loss(pred_wt, target_wt)
        loss_tc = self._dice_loss(pred_tc, target_tc)
        loss_et = self._dice_loss(pred_et, target_et)
        
        return loss_wt + loss_tc + loss_et

    def forward(self, inputs_list, targets):
        """
        Main forward pass.
        If inputs_list has multiple tensors, applies deep supervision weighting (1.0, 0.5, 0.25).
        """
        # Loss for final output (Full Resolution)
        loss = self.forward_single(inputs_list[0], targets)
        
        # Loss for Deep Supervision 1 (1/2 Resolution)
        if len(inputs_list) > 1:
            target_ds1 = torch.nn.functional.interpolate(targets.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
            loss += 0.5 * self.forward_single(inputs_list[1], target_ds1)
            
        # Loss for Deep Supervision 2 (1/4 Resolution)
        if len(inputs_list) > 2:
            target_ds2 = torch.nn.functional.interpolate(targets.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
            loss += 0.25 * self.forward_single(inputs_list[2], target_ds2)
            
        return loss

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, accumulation_steps):
    """Executes one epoch of training."""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    loop = tqdm(loader, desc="Training", leave=False, smoothing=0)
    for i, batch in enumerate(loop):
        data = batch['image'].to(device, non_blocking=True)
        targets = batch['mask'].to(device, non_blocking=True).long()
        
        # Mixed Precision Training
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps # Normalize loss for accumulation
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accumulation_steps
        loop.set_postfix(loss=running_loss / (i + 1))
        
    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, device):
    """Executes validation loop."""
    model.eval()
    running_loss = 0.0
    loop = tqdm(loader, desc="Validation", leave=False, smoothing=0)
    
    with torch.no_grad():
        for batch in loop:
            data = batch['image'].to(device, non_blocking=True)
            targets = batch['mask'].to(device, non_blocking=True).long()
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                predictions = model(data)
                # Validation only uses the final output (no deep supervision)
                loss = criterion.forward_single(predictions, targets)
            running_loss += loss.item()
            
    return running_loss / len(loader)

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    project_root = os.path.dirname(src_dir) 
    log_dir = os.path.join(project_root, "logs", f"run_{timestamp}_RegionLoss")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"Initializing Training (Region Loss) on {DEVICE}...")
    
    # Data Loaders
    train_dataset = BratsDataset(phase="train")
    val_dataset = BratsDataset(phase="val")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              num_workers=config.NUM_WORKERS, pin_memory=True, prefetch_factor=config.PREFETCH_FACTOR)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                            num_workers=config.NUM_WORKERS, pin_memory=True, prefetch_factor=config.PREFETCH_FACTOR)
    
    # Model & Optimization
    model = ResUnet3D(in_channels=4, num_classes=4).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = RegionWiseDiceLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    
    # Training Loop
    for epoch in range(100):
        print(f"\nEpoch [{epoch+1}/100]")
        start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, DEVICE, config.ACCUM_STEPS)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - start:.1f}s")
        writer.add_scalars("Loss", {"Train": train_loss, "Val": val_loss}, epoch)
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(">>> Best Model Saved!")

    writer.close()

if __name__ == "__main__":
    main()