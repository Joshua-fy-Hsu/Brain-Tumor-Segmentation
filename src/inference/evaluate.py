import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from model.model import ResUnet3D
from preprocessing.dataset import BratsDataset
from configs import config

# --- CONFIGURATION ---
WINDOW_SIZE = (96, 96, 96) # Patch size for sliding window
STRIDE = (48, 48, 48)      # Overlap stride
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_dice(pred, target):
    """Computes Dice coefficient between two binary masks."""
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)

def get_clinical_dice(pred_mask, gt_mask):
    """
    Computes Dice scores for the three clinical regions:
    WT: Whole Tumor (Labels 1, 2, 3)
    TC: Tumor Core (Labels 1, 3)
    ET: Enhancing Tumor (Label 3)
    """
    # 1. Whole Tumor (WT)
    pred_wt = (pred_mask > 0).astype(float)
    gt_wt   = (gt_mask > 0).astype(float)
    dice_wt = calculate_dice(pred_wt, gt_wt)

    # 2. Tumor Core (TC)
    pred_tc = ((pred_mask == 1) | (pred_mask == 3)).astype(float)
    gt_tc   = ((gt_mask == 1) | (gt_mask == 3)).astype(float)
    dice_tc = calculate_dice(pred_tc, gt_tc)

    # 3. Enhancing Tumor (ET)
    pred_et = (pred_mask == 3).astype(float)
    gt_et   = (gt_mask == 3).astype(float)
    dice_et = calculate_dice(pred_et, gt_et)
    
    return dice_wt, dice_tc, dice_et

def predict_with_tta(model, patch):
    """
    Test Time Augmentation (TTA).
    Predicts the patch 4 times:
      1. Original
      2. Flip X
      3. Flip Y
      4. Flip Z
    Returns the average probability map.
    """
    # 1. Original
    logits = model(patch)
    probs = F.softmax(logits, dim=1)
    
    # 2. Flip X (dim 2)
    patch_x = torch.flip(patch, dims=[2])
    logits_x = model(patch_x)
    probs_x = torch.flip(F.softmax(logits_x, dim=1), dims=[2])
    probs += probs_x
    
    # 3. Flip Y (dim 3)
    patch_y = torch.flip(patch, dims=[3])
    logits_y = model(patch_y)
    probs_y = torch.flip(F.softmax(logits_y, dim=1), dims=[3])
    probs += probs_y

    # 4. Flip Z (dim 4)
    patch_z = torch.flip(patch, dims=[4])
    logits_z = model(patch_z)
    probs_z = torch.flip(F.softmax(logits_z, dim=1), dims=[4])
    probs += probs_z
    
    return probs / 4.0 # Average

def predict_sliding_window(model, image, use_tta=True):
    """
    Performs inference on a large volume using sliding window approach.
    Stitches patches together and handles padding.
    """
    model.eval()
    image = image.to(DEVICE)
    _, h, w, d = image.shape
    
    # Pad image to fit window size
    ph = max(WINDOW_SIZE[0] - h, 0)
    pw = max(WINDOW_SIZE[1] - w, 0)
    pd = max(WINDOW_SIZE[2] - d, 0)
    image = F.pad(image, (0, pd, 0, pw, 0, ph))
    
    _, h_pad, w_pad, d_pad = image.shape
    output_map = torch.zeros((4, h_pad, w_pad, d_pad), device=DEVICE)
    count_map = torch.zeros((4, h_pad, w_pad, d_pad), device=DEVICE)
    
    # Calculate number of steps
    steps_h = int(np.ceil((h_pad - WINDOW_SIZE[0]) / STRIDE[0])) + 1
    steps_w = int(np.ceil((w_pad - WINDOW_SIZE[1]) / STRIDE[1])) + 1
    steps_d = int(np.ceil((d_pad - WINDOW_SIZE[2]) / STRIDE[2])) + 1
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            for i in range(steps_h):
                for j in range(steps_w):
                    for k in range(steps_d):
                        sh = min(i * STRIDE[0], h_pad - WINDOW_SIZE[0])
                        sw = min(j * STRIDE[1], w_pad - WINDOW_SIZE[1])
                        sd = min(k * STRIDE[2], d_pad - WINDOW_SIZE[2])
                        
                        patch = image[:, sh:sh+WINDOW_SIZE[0], sw:sw+WINDOW_SIZE[1], sd:sd+WINDOW_SIZE[2]].unsqueeze(0)
                        
                        if use_tta:
                            prob = predict_with_tta(model, patch).squeeze(0)
                        else:
                            pred = model(patch)
                            prob = F.softmax(pred, dim=1).squeeze(0)
                        
                        output_map[:, sh:sh+WINDOW_SIZE[0], sw:sw+WINDOW_SIZE[1], sd:sd+WINDOW_SIZE[2]] += prob
                        count_map[:, sh:sh+WINDOW_SIZE[0], sw:sw+WINDOW_SIZE[1], sd:sd+WINDOW_SIZE[2]] += 1

    output_map /= count_map
    prediction = torch.argmax(output_map, dim=0)
    
    # Crop back to original size
    prediction = prediction[:h, :w, :d]
    return prediction.cpu().numpy().astype(np.uint8)

def main():
    print("--- STARTING TTA EVALUATION ---")
    
    # 1. Locate the best model from logs
    logs_root = os.path.join(os.path.dirname(src_dir), "logs")
    run_folders = sorted([os.path.join(logs_root, d) for d in os.listdir(logs_root) if os.path.isdir(os.path.join(logs_root, d))], reverse=True)
    
    model_path = None
    for folder in run_folders:
        potential = os.path.join(folder, "best_model.pth")
        if os.path.exists(potential):
            model_path = potential
            break
            
    if not model_path:
        print("Error: No 'best_model.pth' found.")
        return

    print(f"Loading Model: {model_path}")
    model = ResUnet3D(in_channels=4, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    
    # 2. Load Validation Dataset
    val_dataset = BratsDataset(phase="val") 
    results = []
    
    print(f"Evaluating on {len(val_dataset)} patients...")
    for i in tqdm(range(len(val_dataset))):
        patient_id = val_dataset.patient_list[i]
        pat_path = os.path.join(val_dataset.data_root, patient_id)
        
        # Load Stats
        import json
        with open(os.path.join(pat_path, "stats.json"), "r") as f:
            stats = json.load(f)
            
        # Load Images & Normalize
        import nibabel as nib
        channels = []
        for mod in config.MODALITIES:
            p = os.path.join(pat_path, f"{patient_id}_{mod}.nii")
            d = nib.load(p).get_fdata().astype(np.float32)
            d = (d - stats[mod]['mean']) / (stats[mod]['std'] + 1e-8)
            channels.append(d)
        
        image = torch.from_numpy(np.stack(channels))
        
        # Load GT Mask
        p_mask = os.path.join(pat_path, f"{patient_id}_seg.nii")
        gt_mask = nib.load(p_mask).get_fdata().astype(np.uint8)
        
        # Predict
        pred_mask = predict_sliding_window(model, image, use_tta=True)
        
        # Calculate Scores
        dice_wt, dice_tc, dice_et = get_clinical_dice(pred_mask, gt_mask)
        
        results.append({
            "PatientID": patient_id,
            "WT": dice_wt,
            "TC": dice_tc,
            "ET": dice_et
        })

    # 3. Save & Report
    df = pd.DataFrame(results)
    
    avg_wt = df["WT"].mean()
    avg_tc = df["TC"].mean()
    avg_et = df["ET"].mean()
    
    print("\n" + "="*30)
    print("FINAL RESULTS (With TTA)")
    print("="*30)
    print(f"Whole Tumor (WT):     {avg_wt:.4f}")
    print(f"Tumor Core (TC):      {avg_tc:.4f}")
    print(f"Enhancing Tumor (ET): {avg_et:.4f}")
    print("="*30)
    
    save_path = os.path.join(os.path.dirname(model_path), "final_scores_tta.csv")
    df.to_csv(save_path, index=False)
    print(f"Report saved to: {save_path}")

if __name__ == "__main__":
    main()