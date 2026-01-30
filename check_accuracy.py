# ==========================================================================================================
# check_accuracy_ULTRACLEAN_FINAL.py
# Computes PSNR, SSIM, MAE; cleans AI-generated FLAIR; shows MRI-style sharp visualizations
# ==========================================================================================================

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.filters import unsharp_mask
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# ==========================================================================================================
# PATHS — change these to your own
# ==========================================================================================================
CASE_DIR   = r"D:\DLH\brats2020_250cases\Case_004"
SYNTH_PATH = r"D:\DLH\OUTPUTS\SYNTHETIC_FLAIR_FINAL.nii.gz"

# ==========================================================================================================
# HELPER FUNCTIONS
# ==========================================================================================================
def find_file(case_dir, key):
    for f in Path(case_dir).glob("*.nii*"):
        if key.lower() in f.name.lower():
            return str(f)
    raise FileNotFoundError(f"{key} not found in {case_dir}")

def normalize_volume(vol, clip_percentiles=(1, 99), mask_thresh=0.01):
    out = np.zeros_like(vol)
    for z in range(vol.shape[2]):
        sl = vol[:, :, z]
        mask = sl > mask_thresh
        if mask.any():
            lo, hi = np.percentile(sl[mask], clip_percentiles)
            sl = (sl - lo) / (hi - lo + 1e-6)
            sl = np.clip(sl, 0, 1)
        sl[~mask] = 0
        out[:, :, z] = sl
    return out

def enhance_synth_slice(sl, mask_thresh=0.03):
    x = sl.copy()
    mask = x > mask_thresh
    x[~mask] = 0.0
    if mask.any():
        lo, hi = np.percentile(x[mask], [2, 98])
        x = (x - lo) / (hi - lo + 1e-6)
        x = np.clip(x, 0, 1)
    x = equalize_adapthist(x, clip_limit=0.02)
    x = unsharp_mask(x, radius=1.5, amount=1.2)
    x[~mask] = 0.0
    return np.clip(x, 0, 1)

# ==========================================================================================================
# LOAD VOLUMES
# ==========================================================================================================
real_path  = find_file(CASE_DIR, "flair")
real_nii   = nib.load(real_path)
synth_nii  = nib.load(SYNTH_PATH)

real  = real_nii.get_fdata(dtype=np.float32)
synth = synth_nii.get_fdata(dtype=np.float32)

# ==========================================================================================================
# NORMALIZE REAL & CLEAN SYNTHETIC
# ==========================================================================================================
real_norm = normalize_volume(real)

synth_clean = np.zeros_like(synth)
for z in range(synth.shape[2]):
    synth_clean[:, :, z] = enhance_synth_slice(synth[:, :, z])

# ==========================================================================================================
# METRICS (on brain only)
# ==========================================================================================================
brain_mask = real_norm > 0.05
PSNR = psnr(real_norm[brain_mask], synth_clean[brain_mask], data_range=1.0)
SSIM = ssim(real_norm, synth_clean, data_range=1.0, gaussian_weights=True, sigma=1.0)
MAE  = np.mean(np.abs(real_norm[brain_mask] - synth_clean[brain_mask]))

print("\n===================== METRICS =====================")
print(f"PSNR : {PSNR:.2f} dB")
print(f"SSIM : {SSIM:.4f}")
print(f"MAE  : {MAE:.6f}")
print("====================================================\n")

# ==========================================================================================================
# VISUALIZATION
# ==========================================================================================================
mid = real_norm.shape[2] // 2

plt.figure(figsize=(22, 10))
plt.suptitle("ULTRA-CLEAN SYNTHETIC FLAIR (MRI-like, Sharp, Haze-free)", fontsize=26, fontweight="bold")

# Ground Truth
plt.subplot(1, 3, 1)
plt.title("Ground Truth FLAIR", fontsize=18)
plt.imshow(real_norm[:, :, mid], cmap="gray", vmin=0, vmax=1)
plt.axis("off")

# Cleaned Synthetic
plt.subplot(1, 3, 2)
plt.title("Enhanced AI Synthetic FLAIR", fontsize=18, color="lime")
plt.imshow(synth_clean[:, :, mid], cmap="gray", vmin=0, vmax=1)
plt.axis("off")

# Error Map
err = np.abs(real_norm[:, :, mid] - synth_clean[:, :, mid])
plt.subplot(1, 3, 3)
plt.title("Absolute Error Map", fontsize=18)
im = plt.imshow(err, cmap="hot", vmin=0, vmax=0.25)
plt.colorbar(im)
plt.axis("off")

plt.tight_layout()
plt.show()
