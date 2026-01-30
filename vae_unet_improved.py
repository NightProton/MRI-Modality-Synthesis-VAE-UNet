#!/usr/bin/env python3
"""
vae_unet_improved.py
Improved VAE-UNet training + inference script based on your original.
- Default epochs: 10
- Checkpoint every 5 epochs
- KL annealing, L1 + MSE losses, optional perceptual loss
- ConvTranspose upsampling (sharper)
- Safer normalization & robust metrics
"""

import os
import argparse
import random
import math
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# optional imports
try:
    import nibabel as nib
except Exception:
    nib = None

from PIL import Image
import matplotlib.pyplot as plt

# optional metrics
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except Exception:
    ssim = None
    compare_psnr = None

# optional perceptual
try:
    import torchvision
    from torchvision import models, transforms
    TORCHVISION_AVAILABLE = True
except Exception:
    TORCHVISION_AVAILABLE = False

# ----------------------------
# DEFAULT CONFIG (modified)
# ----------------------------
DEFAULTS = {
    "ROOT_DIR": r"D:\DLH\brats2020_250cases",
    "MODEL_SAVE_PATH": r"./vae_unet_checkpt.pt",
    "IMG_SIZE": 160,
    "SLICE_STRIDE": 2,
    "BATCH_SIZE": 8,
    "LATENT_DIM": 128,
    "EPOCHS": 10,
    "LR": 1e-4,
    "KL_WEIGHT": 1e-4,
    "RANDOM_SEED": 42,
    "MODALITY_KEYS": ["flair", "t1", "t1ce", "t2"],
    "NUM_WORKERS": 4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SAMPLE_DIR": "./samples",
    "CHECKPOINT_EVERY": 5
}

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def psnr_torch(a: torch.Tensor, b: torch.Tensor, data_range=1.0):
    mse = F.mse_loss(a, b, reduction='mean').item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def numpy_ssim(a: np.ndarray, b: np.ndarray):
    if ssim is None:
        return None
    return ssim(a, b, data_range=1.0)

def save_image_grid(inputs: List[np.ndarray], filename: str, titles: List[str] = None):
    n = len(inputs)
    plt.figure(figsize=(3 * n, 3))
    for i, img in enumerate(inputs):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        if titles and i < len(titles):
            plt.title(titles[i])
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ----------------------------
# Data helpers
# ----------------------------
def find_modality_file(case_dir: str, key: str):
    key = key.lower()
    for f in os.listdir(case_dir):
        if key in f.lower() and f.lower().endswith((".nii", ".nii.gz")):
            return os.path.join(case_dir, f)
    for f in os.listdir(case_dir):
        if key in f.lower() and f.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(case_dir, f)
    raise FileNotFoundError(f"Modality '{key}' not found in {case_dir}")

def load_slice_from_nifti(path: str, z: int) -> np.ndarray:
    if nib is None:
        raise RuntimeError("nibabel is required to load nifti files")
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if z < 0 or z >= data.shape[2]:
        raise IndexError("z index out of range")
    return data[:, :, z]

def normalize_slice(slice2d: np.ndarray) -> np.ndarray:
    mask = slice2d > 0
    if not mask.any():
        return np.zeros_like(slice2d, dtype=np.float32)
    arr = slice2d[mask]
    lo, hi = np.percentile(arr, [1.0, 99.0])
    slice2d = np.clip(slice2d, lo, hi)
    slice2d = (slice2d - lo) / (hi - lo + 1e-8)
    slice2d = np.nan_to_num(slice2d).astype(np.float32)
    slice2d[~mask] = 0.0
    return slice2d

def center_crop_2d(img: np.ndarray, size: int):
    h, w = img.shape
    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode='constant')
        h, w = img.shape
    sh, sw = (h - size) // 2, (w - size) // 2
    return img[sh:sh+size, sw:sw+size]

# ----------------------------
# Dataset class (BraTS-style)
# ----------------------------
class BraTSSliceDataset(Dataset):
    def __init__(self, root_dir: str, modality_keys: List[str], crop_size: int = 160, stride: int = 2):
        self.root_dir = Path(root_dir)
        self.cases = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.modality_keys = [k.lower() for k in modality_keys]
        self.target_key = self.modality_keys[0]  # first is target (e.g. flair)
        self.input_keys = self.modality_keys[1:]
        self.crop_size = crop_size
        self.stride = stride
        self.slices = self._build_slice_index()

    def _build_slice_index(self):
        slices = []
        for case in self.cases:
            target_path = find_modality_file(str(case), self.target_key)
            target_nii = nib.load(target_path)
            depth = target_nii.shape[2]
            for z in range(0, depth, self.stride):
                slices.append((case, z))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        case_dir, z = self.slices[idx]
        target_path = find_modality_file(str(case_dir), self.target_key)
        target = load_slice_from_nifti(target_path, z)
        target = normalize_slice(target)
        target = center_crop_2d(target, self.crop_size)

        inputs = []
        for key in self.input_keys:
            path = find_modality_file(str(case_dir), key)
            sl = load_slice_from_nifti(path, z)
            sl = normalize_slice(sl)
            sl = center_crop_2d(sl, self.crop_size)
            inputs.append(sl)

        return {
            "input": torch.from_numpy(np.stack(inputs)).float(),
            "target": torch.from_numpy(target[None]).float()
        }

# ----------------------------
# Model blocks
# ----------------------------
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

# FIXED UP BLOCK — THIS IS THE ONLY CHANGE
class Up(nn.Module):
    def __init__(self, dec_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(dec_channels, dec_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(dec_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
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

        self.bottleneck_conv = ConvBlock(c*8, 256)

        self.fc_mu = nn.Linear(256*10*10, latent_dim)
        self.fc_logvar = nn.Linear(256*10*10, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*10*10)

        # FIXED UP LAYERS — ONLY THESE 4 LINES CHANGED
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

        if deterministic:
            z = mu
        else:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        dec = self.fc_dec(z).view(-1, 256, 10, 10)
        dec = F.interpolate(dec, size=p4.shape[2:], mode="bilinear", align_corners=False)

        x = self.up4(dec, c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)

        out = self.final(x)
        return torch.sigmoid(out), mu, logvar

# ----------------------------
# Loss & training helpers
# ----------------------------
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def train_epoch(model, loader, optimizer, device, epoch, total_epochs, base_kl_weight):
    model.train()
    total_loss = 0
    total_mse = total_l1 = total_kl = 0
    kl_weight = base_kl_weight * min(epoch / (total_epochs * 0.5), 1.0)

    for batch in tqdm(loader, desc=f"Train {epoch}"):
        inp = batch["input"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(inp, deterministic=False)
        mse = F.mse_loss(recon, target)
        l1 = F.l1_loss(recon, target)
        kl = kl_divergence(mu, logvar)
        loss = mse + l1 + kl_weight * kl

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_l1 += l1.item()
        total_kl += kl.item()

    return (total_loss / len(loader), total_mse / len(loader),
            total_l1 / len(loader), total_kl / len(loader))

def eval_epoch(model, loader, device, base_kl_weight, epoch, total_epochs):
    model.eval()
    total_loss = total_mse = total_l1 = total_kl = 0
    total_psnr = total_ssim = 0
    kl_weight = base_kl_weight * min(epoch / (total_epochs * 0.5), 1.0)

    with torch.no_grad():
        for batch in loader:
            inp = batch["input"].to(device)
            target = batch["target"].to(device)
            recon, mu, logvar = model(inp)
            mse = F.mse_loss(recon, target)
            l1 = F.l1_loss(recon, target)
            kl = kl_divergence(mu, logvar)
            loss = mse + l1 + kl_weight * kl

            total_loss += loss.item()
            total_mse += mse.item()
            total_l1 += l1.item()
            total_kl += kl.item()
            total_psnr += psnr_torch(recon, target)
            if ssim is not None:
                total_ssim += numpy_ssim(recon[0,0].cpu().numpy(), target[0,0].cpu().numpy())

    n = len(loader)
    return (total_loss / n, total_mse / n, total_l1 / n, total_kl / n,
            total_psnr / n, total_ssim / n if ssim else None)

def save_checkpoint(ckpt, path):
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

def run_inference(model, loader, device, save_dir, max_samples=16):
    model.eval()
    ensure_dir(save_dir)
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= max_samples:
                break
            inp = batch["input"].to(device)
            recon, _, _ = model(inp)
            for i in range(inp.shape[0]):
                if count >= max_samples:
                    break
                imgs = [inp[i,j].cpu().numpy() for j in range(inp.shape[1])] + [recon[i,0].cpu().numpy()]
                titles = [f"Input {j}" for j in range(inp.shape[1])] + ["Recon"]
                save_image_grid(imgs, os.path.join(save_dir, f"sample_{count}.png"), titles)
                count += 1
    return count

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "infer"], default="train")
    parser.add_argument("--root", type=str, default=DEFAULTS["ROOT_DIR"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["EPOCHS"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["BATCH_SIZE"])
    parser.add_argument("--img_size", type=int, default=DEFAULTS["IMG_SIZE"])
    parser.add_argument("--stride", type=int, default=DEFAULTS["SLICE_STRIDE"])
    parser.add_argument("--latent", type=int, default=DEFAULTS["LATENT_DIM"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["LR"])
    parser.add_argument("--kl", type=float, default=DEFAULTS["KL_WEIGHT"])
    parser.add_argument("--save", type=str, default=DEFAULTS["MODEL_SAVE_PATH"])
    parser.add_argument("--samples", type=str, default=DEFAULTS["SAMPLE_DIR"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["RANDOM_SEED"])
    parser.add_argument("--workers", type=int, default=DEFAULTS["NUM_WORKERS"])
    parser.add_argument("--modalities", type=str, default=",".join(DEFAULTS["MODALITY_KEYS"]))
    parser.add_argument("--max_infer", type=int, default=16)
    parser.add_argument("--checkpoint_every", type=int, default=DEFAULTS["CHECKPOINT_EVERY"])
    args = parser.parse_args()

    cfg = {
        **DEFAULTS,
        "ROOT_DIR": args.root,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch,
        "IMG_SIZE": args.img_size,
        "SLICE_STRIDE": args.stride,
        "LATENT_DIM": args.latent,
        "LR": args.lr,
        "KL_WEIGHT": args.kl,
        "MODEL_SAVE_PATH": args.save,
        "SAMPLE_DIR": args.samples,
        "RANDOM_SEED": args.seed,
        "NUM_WORKERS": args.workers,
        "MODALITY_KEYS": [m.strip() for m in args.modalities.split(",")],
        "CHECKPOINT_EVERY": args.checkpoint_every
    }

    print("[Config]")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    set_seed(cfg["RANDOM_SEED"])
    device = cfg["DEVICE"]
    print(f"[Device] Using {device}")

    dataset = BraTSSliceDataset(cfg["ROOT_DIR"], cfg["MODALITY_KEYS"], crop_size=cfg["IMG_SIZE"], stride=cfg["SLICE_STRIDE"])
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(0.9 * n)
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=cfg["BATCH_SIZE"], shuffle=True,
                              num_workers=cfg["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=cfg["BATCH_SIZE"], shuffle=False,
                            num_workers=cfg["NUM_WORKERS"], pin_memory=True)

    model = VAEUNet(in_channels=len(cfg["MODALITY_KEYS"]) - 1, base_c=32, latent_dim=cfg["LATENT_DIM"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    if args.mode == "train":
        best_val_loss = float("inf")
        ensure_dir(os.path.dirname(cfg["MODEL_SAVE_PATH"]) or ".")
        for epoch in range(1, cfg["EPOCHS"] + 1):
            print(f"\n=== Epoch {epoch}/{cfg['EPOCHS']} ===")
            train_loss, train_mse, train_l1, train_kl = train_epoch(model, train_loader, optimizer, device, epoch, cfg["EPOCHS"], cfg["KL_WEIGHT"])
            val_loss, val_mse, val_l1, val_kl, val_psnr, val_ssim = eval_epoch(model, val_loader, device, cfg["KL_WEIGHT"], epoch, cfg["EPOCHS"])
            print(f"[Epoch {epoch}] Train loss: {train_loss:.6f} (mse {train_mse:.6f}, l1 {train_l1:.6f}, kl {train_kl:.6f})")
            print(f"[Epoch {epoch}] Val   loss: {val_loss:.6f} (mse {val_mse:.6f}, l1 {val_l1:.6f}, kl {val_kl:.6f}) PSNR: {val_psnr:.2f} SSIM: {val_ssim:.4f}")

            scheduler.step(val_loss)

            if epoch % cfg["CHECKPOINT_EVERY"] == 0 or epoch == cfg["EPOCHS"]:
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "cfg": cfg
                }
                ckpt_path = f"{cfg['MODEL_SAVE_PATH']}.epoch{epoch}.pt"
                save_checkpoint(ckpt, ckpt_path)
                print(f"[Checkpoint] saved {ckpt_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({"epoch": epoch, "model_state": model.state_dict(), "optim_state": optimizer.state_dict(), "cfg": cfg},
                                cfg["MODEL_SAVE_PATH"])
                print(f"[Saved Best] {cfg['MODEL_SAVE_PATH']}")

        print("Training finished.")

    elif args.mode == "eval":
        if os.path.exists(cfg["MODEL_SAVE_PATH"]):
            print(f"Loading checkpoint {cfg['MODEL_SAVE_PATH']}")
            load_checkpoint(cfg["MODEL_SAVE_PATH"], model, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {cfg['MODEL_SAVE_PATH']}")

        val_loss, val_mse, val_l1, val_kl, val_psnr, val_ssim = eval_epoch(model, val_loader, device, cfg["KL_WEIGHT"], cfg["EPOCHS"], cfg["EPOCHS"])
        print(f"[Eval] PSNR: {val_psnr:.2f} SSIM: {val_ssim:.4f}")

    elif args.mode == "infer":
        if os.path.exists(cfg["MODEL_SAVE_PATH"]):
            print(f"Loading checkpoint {cfg['MODEL_SAVE_PATH']}")
            load_checkpoint(cfg["MODEL_SAVE_PATH"], model, map_location=device)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {cfg['MODEL_SAVE_PATH']}")

        ensure_dir(cfg["SAMPLE_DIR"])
        n_saved = run_inference(model, val_loader, device, cfg["SAMPLE_DIR"], max_samples=args.max_infer)
        print(f"Inference finished. Saved {n_saved} samples to {cfg['SAMPLE_DIR']}")

if __name__ == "__main__":
    main()