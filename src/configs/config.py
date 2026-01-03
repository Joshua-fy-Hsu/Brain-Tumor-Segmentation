import torch
import os

# --- PATH CONFIGURATION ---
# Path to the optimized BraTS 2021 dataset (processed NIfTI files).
TRAIN_DATA_PATH = r"D:/University/Junior/DL/Brain_Tumor_Segmentation/data/BraTS2021_TrainingData"

# List of MRI modalities used as input channels.
MODALITIES = ["t1", "t1ce", "t2", "flair"]

# Number of output classes (0: Background, 1: Necrotic, 2: Edema, 3: Enhancing).
NUM_CLASSES = 4

# Device selection: Use GPU (CUDA) if available, otherwise CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Input patch size for the 3D model (Depth, Height, Width).
PATCH_SIZE = (96, 96, 96)

# --- DATA SPLIT CONFIGURATION ---
# Number of patients to use for the training set (the remainder are used for validation).
TRAIN_COUNT = 1000

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4        # Number of samples per batch.
ACCUM_STEPS = 8       # Gradient accumulation steps to simulate larger batch sizes.
NUM_WORKERS = 8       # Number of subprocesses for data loading.
PREFETCH_FACTOR = 4   # Number of batches loaded in advance by each worker.
PIN_MEMORY = True     # Pin memory for faster data transfer to CUDA devices.
SEED = 67             # Random seed for reproducibility.