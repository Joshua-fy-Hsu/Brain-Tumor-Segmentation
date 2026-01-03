import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
# Input: Path to raw BraTS 2021 Data
RAW_PATH = r"D:/University/Junior/DL/Brain_Tumor_Segmentation/data/BraTS2021_TrainingData"

# Output: Path for the optimized dataset
OPT_PATH = r"D:/University/Junior/DL/Brain_Tumor_Segmentation/data/BraTS2021_Optimized"

MODALITIES = ["t1", "t1ce", "t2", "flair"]

def optimize_dataset():
    """
    Reads raw BraTS data, standardizes filenames, calculates statistics (mean/std),
    fixes label mappings, and saves optimized files for faster loading.
    """
    print(f"--> Source: {RAW_PATH}")
    print(f"--> Destination: {OPT_PATH}")
    
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: Source path not found: {RAW_PATH}")
        return

    os.makedirs(OPT_PATH, exist_ok=True)
    
    # Identify patient folders
    patients = sorted([
        f for f in os.listdir(RAW_PATH) 
        if os.path.isdir(os.path.join(RAW_PATH, f))
    ])
    
    print(f"Found {len(patients)} patients. Starting optimization...")
    
    for pat_id in tqdm(patients, desc="Optimizing"):
        src_pat_path = os.path.join(RAW_PATH, pat_id)
        dst_pat_path = os.path.join(OPT_PATH, pat_id)
        os.makedirs(dst_pat_path, exist_ok=True)
        
        stats = {}
        
        # 1. Process Modalities
        for mod in MODALITIES:
            # Handle potential file extensions (.nii vs .nii.gz)
            src_file = os.path.join(src_pat_path, f"{pat_id}_{mod}.nii.gz")
            if not os.path.exists(src_file):
                src_file = os.path.join(src_pat_path, f"{pat_id}_{mod}.nii")
                
            if not os.path.exists(src_file):
                continue
            
            # Load Data
            img = nib.load(src_file)
            data = img.get_fdata().astype(np.float32)
            
            # Calculate Mean/Std only on brain tissue (non-zero region)
            mask = data > 0
            if mask.any():
                mean = float(data[mask].mean())
                std = float(data[mask].std())
            else:
                mean, std = 0.0, 1.0
            
            stats[mod] = {"mean": mean, "std": std}
            
            # Save as uncompressed NIfTI for faster I/O during training
            new_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(new_img, os.path.join(dst_pat_path, f"{pat_id}_{mod}.nii"))
            
        # 2. Process Segmentation Mask
        src_mask = os.path.join(src_pat_path, f"{pat_id}_seg.nii.gz")
        if not os.path.exists(src_mask):
             src_mask = os.path.join(src_pat_path, f"{pat_id}_seg.nii")
             
        if os.path.exists(src_mask):
            mask_img = nib.load(src_mask)
            mask_data = mask_img.get_fdata().astype(np.uint8)
            
            # Fix Label 4 -> 3 (BraTS 2021 label 4 is 'Enhancing Tumor', standard is 3)
            mask_data[mask_data == 4] = 3
            
            new_mask = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
            nib.save(new_mask, os.path.join(dst_pat_path, f"{pat_id}_seg.nii"))
        
        # 3. Save Stats JSON
        with open(os.path.join(dst_pat_path, "stats.json"), "w") as f:
            json.dump(stats, f)

if __name__ == "__main__":
    optimize_dataset()