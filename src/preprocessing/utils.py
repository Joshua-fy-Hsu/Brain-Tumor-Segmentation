import numpy as np

def normalize_modality(vol_data):
    """
    Applies Z-Score normalization to the brain region only.
    
    Args:
        vol_data (np.ndarray): 3D array (D, H, W) of a single MRI modality.
        
    Returns:
        np.ndarray: Normalized volume where background remains 0.
    """
    # Create a mask for non-zero regions (the actual brain)
    mask = vol_data > 0
    
    if mask.any():
        # Compute mean and std ONLY on the brain region
        mean = vol_data[mask].mean()
        std = vol_data[mask].std()
        
        # Z-Score Normalization: (x - mean) / std
        vol_data = (vol_data - mean) / (std + 1e-8) # 1e-8 prevents divide by zero
        
        # Critical: Re-mask background to 0 to prevent "intensity skewing"
        # Without this, the background (originally 0) would become negative (0 - mean)/std
        vol_data[~mask] = 0
        
    return vol_data

def preprocess_volume(volume):
    """
    Normalizes a 4-channel MRI volume.
    
    Args:
        volume (np.ndarray): 4D array (4, Depth, Height, Width)
                             Channels: T1, T1ce, T2, FLAIR
                             
    Returns:
        np.ndarray: The normalized volume.
    """
    # Loop through each of the 4 channels and normalize them independently
    for c in range(volume.shape[0]):
        volume[c] = normalize_modality(volume[c])
        
    return volume