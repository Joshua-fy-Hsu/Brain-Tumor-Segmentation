# Optimized 3D Deep Learning for Brain Tumor Segmentation


## I. Executive Summary

This project addresses the computational barriers of training volumetric Deep Learning models for medical imaging. While 3D models are critical for capturing z-axis spatial context in Brain Tumor Segmentation (BraTS), they traditionally require enterprise-grade GPUs (e.g., NVIDIA A100) to handle the massive memory footprint of voxel data.

We successfully engineered an optimized training pipeline capable of running on consumer hardware (NVIDIA RTX 4060, 8GB VRAM) without compromising segmentation accuracy. By employing a custom 3D Res-UNet, Region-Wise Dice Loss, and hardware-aware optimizations like Mixed Precision and Gradient Accumulation, we achieved a Whole Tumor (WT) Dice Score of ~0.90 on the BraTS 2021 dataset.

---

## II. Dataset and Preprocessing

### 1. Data Source
The project utilizes the BraTS 2021 Dataset, which is significantly larger than previous iterations, containing over 1,000 patient scans.

* **Volume Dimensions:** $240 \times 240 \times 155$ voxels.
* **Modalities (4 Channels):**
    * **T1:** Anatomical
    * **T1ce:** Active Tumor
    * **T2:** Edema
    * **FLAIR:** Suppressed Cerebrospinal Fluid
* **Split:** The first 1,000 patients were allocated for training, with the remaining ~251 used for validation.
<img width="598" height="224" alt="image" src="https://github.com/user-attachments/assets/cda0f515-636d-4d7f-a985-f4b40ee90c8d" />

### 2. Preprocessing & Memory Optimization
To standardize inputs and maximize training efficiency within the 8GB VRAM envelope:

1.  **Z-Score Normalization:** We implemented a custom normalization function (`normalize_modality`) that calculates statistics solely on non-zero brain regions. This prevents the large volume of background air from skewing the intensity distribution and destabilizing gradients.
2.  **Balanced Sampling:** A `get_tumor_centered_coords` logic was implemented to ensure 50% of training patches are centered on tumor regions, counteracting the inherent class imbalance where healthy tissue vastly outweighs tumor voxels.

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/79130272-0a53-48f8-9d1e-4b2fe7014618" />


---

## III. Challenges and Strategic Solutions

Developing a high-performance 3D segmentation model on consumer hardware presented three primary engineering challenges. We addressed each through targeted architectural and algorithmic interventions.

### 1. Challenge: Extreme Class Imbalance
**Problem:** The Tumor Core (TC) represents a tiny fraction (often <5%) of the total brain volume compared to the Whole Tumor (WT) and healthy background. Standard Cross-Entropy loss functions calculate error per voxel, meaning the massive number of "easy" background voxels overwhelms the gradients from the "hard" tumor voxels. This typically results in models that predict boundaries well (high WT score) but fail to identify the internal core (low TC score).

**Solution: Region-Wise Dice Loss**
We replaced voxel-wise classification with a Region-Wise Dice Loss. Unlike standard Dice loss which optimizes for independent classes (0, 1, 2, 3), our implementation optimizes for clinically nested sub-regions directly:

* **Mechanism:** We aggregate softmax probabilities to form soft masks for the Whole Tumor (sum of labels 1, 2, 3), Tumor Core (sum of labels 1, 3), and Enhancing Tumor (label 3).
* **Impact:** This formulation decouples the optimization of the small Tumor Core from the larger Whole Tumor. Even if the Core is only 1% of the volume, the Dice coefficient ($2|A\cap B|/(|A|+|B|)$) normalizes against the size of the region itself, preventing the background from dominating the loss.

### 2. Challenge: Vanishing Gradients in 3D Networks
**Problem:** In 3D networks, the signal path from the input to the bottleneck and back to the output is exceptionally long. Gradients propagating backward from the final layer decay exponentially as they pass through multiple convolutions and downsampling layers. By the time they reach the early encoder layers, the signal is often too weak to effectively update the weights, leading to poor feature extraction.

**Solution: Deep Supervision**
We structurally modified the U-Net decoder to enforce learning at multiple resolutions:

* **Auxiliary Heads:** We attached $1 \times 1 \times 1$ convolution heads to Decoder Level 2 ($1/2$ resolution) and Decoder Level 3 ($1/4$ resolution).
* **Gradient Injection:** During backpropagation, these heads provide direct error signals to the middle layers of the network, bypassing the deepest layers. This "short-circuiting" of the gradient flow ensures that the encoder learns robust features early in training.
* **Weighted Loss:** The final loss is a weighted sum: $L_{total} = 1.0 \times L_{final} + 0.5 \times L_{ds1} + 0.25 \times L_{ds2}$, gradually reducing the influence of lower-resolution outputs.

### 3. Challenge: Hardware Constraints (8GB VRAM)
**Problem:** A single 3D float32 tensor of size $128 \times 128 \times 128$ with 32 feature maps consumes significant memory. When including gradients and optimizer states, a standard batch size of 1 exceeds the 8GB limit of the RTX 4060, causing immediate OOM errors.

**Solution: The "Optimization Trinity"**
1.  **Optimized Patching (58% Reduction):** We reduced the input patch size from $128^3$ to $96^3$. While $128^3$ contains ~2.1 million voxels, $96^3$ contains only ~0.88 million. This 57.8% reduction in input volume was the single most critical factor in fitting the model into VRAM.
2.  **Mixed Precision (50% Storage Reduction):** We utilized `torch.amp.autocast` to perform forward pass operations in Float16 instead of Float32. This halves the memory required for storing activation maps and allows for larger batch sizes.
3.  **Gradient Accumulation (Virtual Batching):** Medical imaging requires stable batch statistics, which a Batch Size of 4 cannot provide. We implemented Gradient Accumulation, where gradients are calculated for 4 samples, but weights are only updated every 8 steps. This mathematically simulates a Batch Size of 32 ($4 \times 8$), stabilizing the convergence trajectory without the memory cost.

---

## IV. Methodology: Architecture and Optimization

### 1. Network Architecture: 3D Res-Unet
We implemented a 3D Residual U-Net designed for volumetric efficiency. The network follows a symmetric encoder-decoder structure with a channel expansion factor of 2 at each depth level:

* **Encoder Levels:** 32, 64, 128, 256 filters.
* **Bottleneck:** 512 filters.
* **Residual Blocks:** Standard convolutions were replaced with Residual Blocks (two $3 \times 3 \times 3$ convolutions + skip connection) to facilitate gradient flow.
* **Instance Normalization:** We specifically chose Instance Normalization over Batch Normalization. Batch Norm relies on batch statistics (mean/variance) which are highly noisy when the batch size is small (e.g., 4). Instance Norm calculates statistics per sample, making it independent of batch size and ideal for our constrained hardware setup.

<img width="687" height="409" alt="image" src="https://github.com/user-attachments/assets/9d6b0315-c7bf-404c-a150-28fd39727b92" />


### 2. Training Hyperparameters
* **Optimizer:** AdamW ($LR=1e^{-4}$, Weight Decay $=1e^{-5}$)
* **Scheduler:** Cosine Annealing LR ($T\_max=100$)
* **Epochs:** 100 (Best model saved approx. Epoch 90)

---

## V. Experimental Results

### 1. Training Dynamics
The training loss demonstrated a sharp and consistent decline from >4.0 to ~1.3, indicating successful volumetric feature learning. Validation loss remained stable without divergence, confirming the efficacy of the regularization strategies.

<img width="916" height="508" alt="image" src="https://github.com/user-attachments/assets/79f75ca4-9392-4022-affd-b33c3a8a5a19" />


### 2. Quantitative Evaluation
The model was evaluated on the validation set of ~251 patients. Analysis of the `final_scores.csv` data reveals:

* **Whole Tumor (WT):** Achieved a Median Dice Score of 0.952 (Mean: $0.923 \pm 0.08$), indicating excellent boundary detection for surgical planning.
* **Tumor Core (TC):** Achieved a Median Dice Score of 0.941 (Mean: $0.895 \pm 0.12$) showing high precision in identifying the solid tumor mass.
* **Enhancing Tumor (ET):** Achieved a Median Dice Score of 0.897 (Mean: $0.842 \pm 0.15$). This remains the most challenging region due to variable contrast uptake, yet the score confirms the effectiveness of Deep Supervision.

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/3c200da2-ef5d-43c9-9cfe-771687046299" />


### 3. Case Analysis
Specific patient results highlight the model's capabilities and limitations:

* **High Performance (BraTS2021_01652):** WT: 0.976 | TC: 0.977 | ET: 0.956. This case demonstrates near-human accuracy on standard glioblastoma presentations.
* **Standard Performance (BraTS2021_01421):** WT: 0.9537 | TC: 0.9480 | ET: 0.8859. The model effectively distinguishes the necrotic core from the surrounding edema.
* **Edge Case (BraTS2021_01416):** WT: 0.785 | TC: 0.809 | ET: 0.688. Performance drops in cases with complex necrotic structures or motion artifacts, highlighting areas for future refinement via advanced augmentation.

---

## VI. Conclusion

This project demonstrates that high-fidelity 3D medical image segmentation is achievable on consumer-grade hardware. By integrating Deep Supervision, Region-Wise Dice Loss, and Gradient Accumulation, we effectively mitigated the limitations of 8GB VRAM. The final pipeline delivers accurate segmentation results on the BraTS 2021 dataset with a Whole Tumor Dice score of ~0.90, providing a scalable and cost-effective solution for automated brain tumor analysis.
