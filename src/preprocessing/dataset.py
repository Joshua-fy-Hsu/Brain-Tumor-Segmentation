import sys
import os
import json
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

# Ensure the source directory is in the system path for module imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from configs import config

class BratsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading BraTS 2021 3D MRI data.
    
    Handles loading NIfTI files, applying Z-score normalization using pre-computed stats,
    and extracting 3D patches (either random or tumor-centered) for training.
    """

    def __init__(self, phase="train"):
        """
        Initialize the dataset.

        Args:
            phase (str): 'train' for training set, 'val' for validation set.
        """
        self.phase = phase
        self.data_root = config.TRAIN_DATA_PATH
        self.patch_size = config.PATCH_SIZE
        
        if not os.path.exists(self.data_root):
             raise FileNotFoundError(f"Optimized path not found: {self.data_root}")

        # 1. Identify valid patient directories (must contain segmentation and stats).
        all_folders = sorted([
            f for f in os.listdir(self.data_root) 
            if os.path.isdir(os.path.join(self.data_root, f))
        ])
        
        valid_patients = []
        for pat_id in all_folders:
            pat_path = os.path.join(self.data_root, pat_id)
            if os.path.exists(os.path.join(pat_path, f"{pat_id}_seg.nii")) and \
               os.path.exists(os.path.join(pat_path, "stats.json")):
                valid_patients.append(pat_id)

        # 2. Split dataset based on configured count.
        split_idx = config.TRAIN_COUNT
        
        # Fallback to 80% split if the dataset is smaller than TRAIN_COUNT.
        if split_idx > len(valid_patients):
            split_idx = int(len(valid_patients) * 0.8)

        if self.phase == "train":
            self.patient_list = valid_patients[:split_idx]
        elif self.phase == "val":
            self.patient_list = valid_patients[split_idx:]
        else:
            self.patient_list = valid_patients

        print(f"[{self.phase.upper()}] Dataset Loaded: {len(self.patient_list)} patients.")

    def __len__(self):
        """Returns the total number of patients in this split."""
        return len(self.patient_list)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (image patch and mask patch) by index.

        Args:
            idx (int): Index of the patient.

        Returns:
            dict: Contains 'image' (tensor, 4xDxHxW) and 'mask' (tensor, DxHxW).
        """
        patient_id = self.patient_list[idx]
        patient_path = os.path.join(self.data_root, patient_id)
        
        # Load the segmentation mask to determine crop coordinates.
        mask_path = os.path.join(patient_path, f"{patient_id}_seg.nii")
        # Use dataobj to avoid loading the full array into memory immediately.
        mask_proxy = nib.load(mask_path).dataobj 
        shape = (240, 240, 155) # Standard BraTS volume shape
        
        # Determine crop strategy:
        # During training, 50% chance to force a crop centered on the tumor.
        if self.phase == "train" and np.random.rand() < 0.5:
            mask = np.asarray(mask_proxy)
            x, y, z = self.get_tumor_centered_coords(mask, shape)
            mask_patch = mask[x:x+96, y:y+96, z:z+96]
        else:
            # Otherwise (and during validation), take a random crop.
            x, y, z = self.get_crop_coords(shape)
            mask_patch = np.asarray(mask_proxy[x:x+96, y:y+96, z:z+96])

        # Load pre-computed statistics for normalization.
        with open(os.path.join(patient_path, "stats.json"), "r") as f:
            stats = json.load(f)

        # Load and normalize MRI modalities.
        img_patch_channels = []
        for mod in config.MODALITIES:
            mod_path = os.path.join(patient_path, f"{patient_id}_{mod}.nii")
            proxy = nib.load(mod_path).dataobj
            patch_data = np.asarray(proxy[x:x+96, y:y+96, z:z+96]).astype(np.float32)
            
            # Apply Z-score normalization: (x - mean) / std
            mean = stats[mod]["mean"]
            std = stats[mod]["std"]
            patch_data = (patch_data - mean) / (std + 1e-8)
            img_patch_channels.append(patch_data)
            
        img_patch = np.stack(img_patch_channels, axis=0)

        return {
            "image": torch.from_numpy(img_patch),
            "mask": torch.from_numpy(mask_patch).long()
        }

    def get_crop_coords(self, shape):
        """Generates random top-left coordinates for a crop."""
        h, w, d = shape
        ph, pw, pd = self.patch_size
        x = np.random.randint(0, max(1, h - ph))
        y = np.random.randint(0, max(1, w - pw))
        z = np.random.randint(0, max(1, d - pd))
        return x, y, z

    def get_tumor_centered_coords(self, mask, shape):
        """
        Generates crop coordinates centered around a random voxel of the tumor.
        If no tumor is present, falls back to a random crop.
        """
        tumor_indices = np.argwhere(mask > 0)
        if len(tumor_indices) > 0:
            center = tumor_indices[np.random.randint(len(tumor_indices))]
            cx, cy, cz = center
            ph, pw, pd = self.patch_size
            # Ensure the crop stays within volume bounds
            x = max(0, min(cx - ph // 2, shape[0] - ph))
            y = max(0, min(cy - pw // 2, shape[1] - pw))
            z = max(0, min(cz - pd // 2, shape[2] - pd))
            return x, y, z
        else:
            return self.get_crop_coords(shape)