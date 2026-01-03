import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
import nibabel as nib

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from model.model import ResUnet3D
from configs import config

# --- CONFIG ---
WINDOW_SIZE = (96, 96, 96)
STRIDE = (48, 48, 48)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_patient_data(patient_id):
    """
    Loads MRI data and Ground Truth for a specific patient.
    Normalizes data using the patient's stats.json.
    """
    data_root = config.TRAIN_DATA_PATH
    pat_path = os.path.join(data_root, patient_id)
    
    # 1. Load Stats
    import json
    stats_path = os.path.join(pat_path, "stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Could not find stats.json for {patient_id}")
    
    with open(stats_path, "r") as f:
        stats = json.load(f)

    # 2. Load Scans & Normalize
    channels = []
    for mod in config.MODALITIES:
        p = os.path.join(pat_path, f"{patient_id}_{mod}.nii")
        if not os.path.exists(p): p += ".gz"
        
        d = nib.load(p).get_fdata().astype(np.float32)
        mean, std = stats[mod]['mean'], stats[mod]['std']
        d = (d - mean) / (std + 1e-8)
        channels.append(d)
        
    image = np.stack(channels) # (4, D, H, W)
    
    # 3. Load Ground Truth
    seg_path = os.path.join(pat_path, f"{patient_id}_seg.nii")
    if not os.path.exists(seg_path): seg_path += ".gz"
    gt_mask = nib.load(seg_path).get_fdata().astype(np.uint8)
    
    return torch.from_numpy(image), gt_mask

def predict_sliding_window(model, image):
    """Performs inference using sliding window (without TTA for speed)."""
    model.eval()
    image = image.to(DEVICE)
    _, d, h, w = image.shape
    
    # Padding
    pd = max(WINDOW_SIZE[0] - d, 0)
    ph = max(WINDOW_SIZE[1] - h, 0)
    pw = max(WINDOW_SIZE[2] - w, 0)
    image = F.pad(image, (0, pw, 0, ph, 0, pd))
    
    _, d_pad, h_pad, w_pad = image.shape
    
    output_map = torch.zeros((4, d_pad, h_pad, w_pad), device=DEVICE)
    count_map = torch.zeros((4, d_pad, h_pad, w_pad), device=DEVICE)
    
    steps_d = int(np.ceil((d_pad - WINDOW_SIZE[0]) / STRIDE[0])) + 1
    steps_h = int(np.ceil((h_pad - WINDOW_SIZE[1]) / STRIDE[1])) + 1
    steps_w = int(np.ceil((w_pad - WINDOW_SIZE[2]) / STRIDE[2])) + 1
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            for i in range(steps_d):
                for j in range(steps_h):
                    for k in range(steps_w):
                        sd = min(i * STRIDE[0], d_pad - WINDOW_SIZE[0])
                        sh = min(j * STRIDE[1], h_pad - WINDOW_SIZE[1])
                        sw = min(k * STRIDE[2], w_pad - WINDOW_SIZE[2])
                        
                        patch = image[:, sd:sd+WINDOW_SIZE[0], sh:sh+WINDOW_SIZE[1], sw:sw+WINDOW_SIZE[2]].unsqueeze(0)
                        
                        out = model(patch)
                        if isinstance(out, tuple): out = out[0]
                            
                        prob = F.softmax(out, dim=1).squeeze(0)
                        output_map[:, sd:sd+WINDOW_SIZE[0], sh:sh+WINDOW_SIZE[1], sw:sw+WINDOW_SIZE[2]] += prob
                        count_map[:, sd:sd+WINDOW_SIZE[0], sh:sh+WINDOW_SIZE[1], sw:sw+WINDOW_SIZE[2]] += 1

    output_map /= count_map
    prediction = torch.argmax(output_map, dim=0)
    prediction = prediction[:d, :h, :w]
    return prediction.cpu().numpy().astype(np.uint8)

def plot_verification(patient_id, t1ce, gt, pred, dice_scores, volumes, save_path=None):
    """
    Generates a 2x2 visualization grid:
    1. T1ce Scan
    2. Ground Truth Overlay
    3. Model Prediction Overlay
    4. Error Analysis (Correct, Missed, False Alarm)
    """
    
    # Find slice with the most tumor area for visualization
    tumor_counts = np.sum(gt > 0, axis=(1, 2))
    z_slice = np.argmax(tumor_counts)
    
    img_t1ce = np.rot90(t1ce[z_slice, :, :])
    img_gt = np.rot90(gt[z_slice, :, :])
    img_pred = np.rot90(pred[z_slice, :, :])
    
    # Custom Colormap: 0=BG, 1=NCR(Red), 2=ED(Green), 3=ET(Yellow)
    cmap = ListedColormap(['black', 'red', 'green', 'yellow'])
    
    # --- LAYOUT ---
    fig, ax = plt.subplots(2, 2, figsize=(14, 16))
    
    # 1. Top-Left: Raw Scan
    ax[0, 0].imshow(img_t1ce, cmap='gray')
    ax[0, 0].set_title(f"T1ce Scan (Slice {z_slice})\nPatient {patient_id}", fontsize=14)
    ax[0, 0].axis('off')
    
    # 2. Top-Right: Ground Truth
    ax[0, 1].imshow(img_t1ce, cmap='gray', alpha=0.6)
    ax[0, 1].imshow(img_gt, cmap=cmap, alpha=0.8, vmin=0, vmax=3)
    ax[0, 1].set_title("Ground Truth (Target)", fontsize=14)
    ax[0, 1].axis('off')

    # 3. Bottom-Left: Prediction
    ax[1, 0].imshow(img_t1ce, cmap='gray', alpha=0.6)
    ax[1, 0].imshow(img_pred, cmap=cmap, alpha=0.8, vmin=0, vmax=3)
    ax[1, 0].set_title("Model Prediction", fontsize=14)
    ax[1, 0].axis('off')

    # 4. Bottom-Right: Error Analysis
    err_rgb = np.zeros((*img_gt.shape, 3))
    bg = (img_t1ce - img_t1ce.min()) / (img_t1ce.max() - img_t1ce.min())
    err_rgb[..., 0] = bg * 0.4
    err_rgb[..., 1] = bg * 0.4
    err_rgb[..., 2] = bg * 0.4
    
    tp = (img_gt > 0) & (img_pred > 0)     # True Positive
    fn = (img_gt > 0) & (img_pred == 0)    # False Negative (Missed)
    fp = (img_gt == 0) & (img_pred > 0)    # False Positive (False Alarm)
    
    # Colors: Teal (Correct), Orange (Missed), Blue (False Alarm)
    err_rgb[tp, 1] = 0.7; err_rgb[tp, 2] = 0.7 
    err_rgb[fn, 0] = 1.0; err_rgb[fn, 1] = 0.5  
    err_rgb[fp, 2] = 1.0; err_rgb[fp, 0] = 0.0; err_rgb[fp, 1] = 0.0

    ax[1, 1].imshow(err_rgb)
    ax[1, 1].set_title("Error Analysis", fontsize=14)
    ax[1, 1].axis('off')
    
    # --- INFO BOX ---
    score_text = (
        f"Metrics (Dice | Volume):\n"
        f"Whole Tumor (WT): {dice_scores[0]:.4f}  |  {volumes[0]:.1f} mL\n"
        f"Tumor Core (TC):  {dice_scores[1]:.4f}  |  {volumes[1]:.1f} mL\n"
        f"Enhancing (ET):   {dice_scores[2]:.4f}  |  {volumes[2]:.1f} mL"
    )
    plt.figtext(0.5, 0.92, score_text, fontsize=16, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
    
    # --- LEGEND ---
    labels = [
        mpatches.Patch(color='red', label='Necrotic (NCR)'),
        mpatches.Patch(color='green', label='Edema (ED)'),
        mpatches.Patch(color='yellow', label='Enhancing (ET)'),
        mpatches.Patch(color='black', label=' | '),
        mpatches.Patch(color='teal', label='Correct Match'),
        mpatches.Patch(color='orange', label='Missed Tumor'),
        mpatches.Patch(color='blue', label='False Alarm')
    ]
    fig.legend(handles=labels, loc='lower center', ncol=7, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.08) 
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", type=str, help="Specific Patient ID to verify")
    args = parser.parse_args()

    print("--- MODEL VERIFICATION START ---")
    
    # Find Best Model
    logs_root = os.path.join(os.path.dirname(src_dir), "logs")
    run_folders = sorted([os.path.join(logs_root, d) for d in os.listdir(logs_root) if os.path.isdir(os.path.join(logs_root, d))], reverse=True)
    
    model_path = None
    for folder in run_folders:
        p = os.path.join(folder, "best_model.pth")
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        print("Error: No 'best_model.pth' found.")
        return
        
    print(f"Loading Model: {model_path}")
    model = ResUnet3D(in_channels=4, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    
    # Select Patient
    if args.patient_id:
        patient_id = args.patient_id
    else:
        # Default to a patient from the validation set
        all_patients = sorted(os.listdir(config.TRAIN_DATA_PATH))
        val_start_idx = getattr(config, 'TRAIN_COUNT', 1000)
        if val_start_idx + 5 < len(all_patients):
            patient_id = all_patients[val_start_idx + 5]
        else:
            patient_id = all_patients[-1]
        
    print(f"Verifying on Patient: {patient_id}")
    
    image_tensor, gt_mask = load_patient_data(patient_id)
    t1ce_vol = image_tensor[1].numpy()
    
    start = time.time()
    pred_mask = predict_sliding_window(model, image_tensor)
    print(f"Prediction finished in {time.time() - start:.2f}s")
    
    # 1. Calculate Dice Scores
    def get_dice(p, g, label):
        inter = ((p == label) & (g == label)).sum()
        union = (p == label).sum() + (g == label).sum()
        return (2*inter + 1e-5)/(union + 1e-5)
    
    dice_et = get_dice(pred_mask, gt_mask, 3)
    dice_tc = get_dice((pred_mask==1)|(pred_mask==3), (gt_mask==1)|(gt_mask==3), 1)
    dice_wt = get_dice(pred_mask>0, gt_mask>0, 1)
    
    scores = (dice_wt, dice_tc, dice_et)
    
    # 2. Calculate Volumes (mL) - 1 voxel = 1 mm^3 in BraTS
    vol_wt = np.sum(pred_mask > 0) / 1000.0
    vol_tc = np.sum((pred_mask == 1) | (pred_mask == 3)) / 1000.0
    vol_et = np.sum(pred_mask == 3) / 1000.0
    volumes = (vol_wt, vol_tc, vol_et)
    
    print(f"\n--- Results for {patient_id} ---")
    print(f"Whole Tumor (WT): {dice_wt:.4f} | {vol_wt:.2f} mL")
    print(f"Tumor Core (TC):  {dice_tc:.4f} | {vol_tc:.2f} mL")
    print(f"Enhancing (ET):   {dice_et:.4f} | {vol_et:.2f} mL")
    
    save_file = os.path.join(os.path.dirname(model_path), f"VERIFY_{patient_id}.png")
    plot_verification(patient_id, t1ce_vol, gt_mask, pred_mask, scores, volumes, save_path=save_file)

if __name__ == "__main__":
    main()