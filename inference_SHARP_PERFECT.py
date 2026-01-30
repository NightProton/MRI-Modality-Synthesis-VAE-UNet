import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================
CASE_DIR       = r"D:\DLH\brats2020_250cases\Case_001"
MODEL_PATH     = r"D:\DLH\vae_unet_checkpt.pt.epoch10.pt"
OUTPUT_DIR     = Path(r"D:\DLH\OUTPUTS")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_NII     = OUTPUT_DIR / "SYNTHETIC_FLAIR_FINAL.nii.gz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ======================================================
# MODEL ARCHITECTURE — MUST MATCH TRAINING EXACTLY
# ======================================================

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = ConvBlock(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)
        return c, p

class Up(nn.Module):
    def __init__(self, dec_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(dec_channels, dec_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(dec_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class VAEUNet(nn.Module):
    def __init__(self, in_channels=3, base_c=32, latent_dim=128):
        super().__init__()
        c = base_c

        self.down1 = Down(in_channels, c)
        self.down2 = Down(c, c*2)
        self.down3 = Down(c*2, c*4)
        self.down4 = Down(c*4, c*8)

        # <-- EXACT NAME AND CHANNELS MATCH YOUR CHECKPOINT
        self.bottleneck_conv = ConvBlock(c*8, 256)

        # <-- EXACT LATENT LAYERS (shape must match)
        self.fc_mu = nn.Linear(256*10*10, latent_dim)
        self.fc_logvar = nn.Linear(256*10*10, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*10*10)

        # <-- EXACT UPSAMPLING BLOCKS MATCH TRAINING
        self.up4 = Up(256, c*8, c*4)
        self.up3 = Up(c*4, c*4, c*2)
        self.up2 = Up(c*2, c*2, c)
        self.up1 = Up(c, c, c)

        self.final = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x, deterministic=True):
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)

        b = self.bottleneck_conv(p4)
        b = F.adaptive_avg_pool2d(b, (10, 10))

        flat = b.flatten(1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)

        z = mu if deterministic else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        dec = self.fc_dec(z).view(-1, 256, 10, 10)
        dec = F.interpolate(dec, p4.shape[2:], mode="bilinear", align_corners=False)

        x = self.up4(dec, c4)
        x = self.up3(x,  c3)
        x = self.up2(x,  c2)
        x = self.up1(x,  c1)

        return torch.sigmoid(self.final(x))

# ======================================================
# LOAD MODEL
# ======================================================
model = VAEUNet(in_channels=3, base_c=32, latent_dim=128).to(DEVICE)

print("Loading checkpoint:", MODEL_PATH)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()
print("Checkpoint loaded ✔")

# ======================================================
# PREPROCESSING
# ======================================================

def normalize_slice(slice2d):
    mask = slice2d > 0
    if not mask.any(): return np.zeros_like(slice2d)
    arr = slice2d[mask]
    lo, hi = np.percentile(arr, [1, 99])
    slice2d = np.clip(slice2d, lo, hi)
    slice2d = (slice2d - lo) / (hi - lo + 1e-8)
    slice2d[~mask] = 0
    return slice2d.astype(np.float32)

def center_crop_160(img):
    h, w = img.shape
    sh = (h - 160) // 2
    sw = (w - 160) // 2
    return img[sh:sh+160, sw:sw+160]

def pad_240(img):
    out = np.zeros((240, 240), dtype=np.float32)
    o = (240 - 160) // 2
    out[o:o+160, o:o+160] = img
    return out

def find_modality(case_dir, key):
    key = key.lower()
    for f in Path(case_dir).iterdir():
        if key in f.name.lower() and (f.suffix == ".nii" or f.suffix.endswith("gz")):
            return str(f)
    raise FileNotFoundError(key)

# ======================================================
# LOAD VOLUMES
# ======================================================
inputs = {}
for key in ["t1", "t1ce", "t2"]:
    path = find_modality(CASE_DIR, key)
    inputs[key] = nib.load(path).get_fdata(dtype=np.float32)

flair_path = find_modality(CASE_DIR, "flair")
ref = nib.load(flair_path)
affine, header = ref.affine, ref.header
depth = ref.shape[2]

# ======================================================
# INFERENCE
# ======================================================
synthetic = np.zeros((240, 240, depth), dtype=np.float32)

print("Synthesizing FLAIR...")
with torch.no_grad():
    for z in tqdm(range(depth)):
        slices = []
        for key in ["t1", "t1ce", "t2"]:
            sl = inputs[key][:, :, z]
            sl = normalize_slice(sl)
            sl = center_crop_160(sl)
            slices.append(sl)

        x = torch.tensor(np.stack(slices)[None], dtype=torch.float32).to(DEVICE)
        pred = model(x)[0, 0].cpu().numpy()
        synthetic[:, :, z] = pad_240(pred)

# ======================================================
# SAVE OUTPUT
# ======================================================
nib.save(nib.Nifti1Image(synthetic, affine, header), str(OUTPUT_NII))
print("DONE ✔ Saved:", OUTPUT_NII)
