# -*- coding: utf-8 -*-
"""
train.py (cGAN: 3D conditional GAN for Voxel-wise Regression on A-SIM 128^3)

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-07-30
Last-Modified: 2025-11-12

Description
----------
End-to-end training script for a 3D conditional GAN adapted from Pix2PixCC (Generator + Multi-Scale PatchGAN Discriminator).
This version mirrors the experimental conditions of previous UNet/ViT runs to enable fair comparison:
  - Same A-SIM HDF5 dataloaders (input: 2 channels [ngal, vpec]; target: 1 channel)
  - Same input-channel selection policy (both/ch1/ch2) with optional zero-padding to keep 2 channels
  - Cyclical learning rate (CLR) policy applied to the Generator only (Discriminator uses a fixed LR)
  - Validation uses Generator-only MSE (identical to UNet/ViT validation metric)
  - Logging includes adversarial losses (loss_D, loss_G) and a reconstruction proxy (MSE on fake vs target)

Outputs
-------
- Checkpoints:
    <ckpt_dir>/{case_tag}_cgan_tgt-<target>_bs<B>_clr[<min>-<max>]_s<seed>_smp<%>_G_best.pt
    <ckpt_dir>/{...}_D_best.pt
    <ckpt_dir>/{...}_G_final.pt
    <ckpt_dir>/{...}_D_final.pt
- CSV log:
    <ckpt_dir>/{...}_log.csv  (epoch, train_loss_G, train_loss_D, train_recon_mse, val_mse, lr_G)

Notes
-----
- Uses InstanceNorm3d + reflection padding in Generator (3D-safe). To avoid trilinear upsample FLOPs warnings,
  set trans_conv=True to use ConvTranspose3d upsampling.
"""

from __future__ import annotations
import os, sys, argparse, random
from types import SimpleNamespace
from typing import Optional, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# ---------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------
# Make project root (two levels up) importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.data_loader import get_dataloader
from src.logger import get_logger
# cGAN components
from src.model import GeneratorPix2PixCC3D, MultiScaleDiscriminator3D, Pix2PixCCLoss3D


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best = float("inf")
        self.early_stop = False

    def __call__(self, val_metric: float):
        if val_metric < self.best - self.delta:
            self.best = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_clr_scheduler(optimizer, min_lr: float, max_lr: float, cycle_length: int = 8):
    """Epoch-wise triangular cyclical LR (applied to Generator only)."""
    assert max_lr >= min_lr > 0
    assert cycle_length >= 2

    def triangular_clr(epoch: int):
        mid = cycle_length // 2
        ep = epoch % cycle_length
        scale = ep / max(1, mid) if ep <= mid else (cycle_length - ep) / max(1, mid)
        return (min_lr / max_lr) + (1.0 - (min_lr / max_lr)) * scale

    for pg in optimizer.param_groups:
        pg["lr"] = max_lr
    return LambdaLR(optimizer, lr_lambda=triangular_clr)


def str2bool(v):
    """Parse boolean-like CLI strings."""
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [B,2,D,H,W], channels = [ngal, vpec]
    case: "both" | "ch1" | "ch2"
    keep_two=True  -> always return 2 channels (missing channel is zero)
    keep_two=False -> return single-channel tensor for ch1/ch2
    """
    assert x.ndim == 5 and x.size(1) == 2, f"Expected [B,2,D,H,W], got {tuple(x.shape)}"
    if case == "both":
        return x
    if case == "ch1":
        if keep_two:
            ch1 = x[:, 0:1]
            z = torch.zeros_like(ch1)
            return torch.cat([ch1, z], dim=1)
        else:
            return x[:, 0:1]
    if case == "ch2":
        if keep_two:
            ch2 = x[:, 1:2]
            z = torch.zeros_like(ch2)
            return torch.cat([z, ch2], dim=1)
        else:
            return x[:, 1:2]
    raise ValueError(f"Unknown input case: {case}")


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def train(args):
    logger = get_logger("train_cgan3d")
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info("🚀 Starting cGAN (Pix2PixCC-3D) training for 3D voxel-wise regression")
    logger.info(f"Args: {vars(args)}")

    # ---------------- Data ----------------
    train_loader = get_dataloader(
        yaml_path=args.yaml_path, split="train",
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        target_field=args.target_field, train_val_split=args.train_val_split,
        sample_fraction=args.sample_fraction, dtype=torch.float32, seed=args.seed,
        validate_keys=args.validate_keys, strict=False,
        exclude_list_path=args.exclude_list, include_list_path=args.include_list,
    )
    val_loader = get_dataloader(
        yaml_path=args.yaml_path, split="val",
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        target_field=args.target_field, train_val_split=args.train_val_split,
        sample_fraction=1.0, dtype=torch.float32, seed=args.seed,
        validate_keys=args.validate_keys, strict=False,
        exclude_list_path=args.exclude_list, include_list_path=args.include_list,
    )
    logger.info(f"📊 Train samples (files): {len(train_loader.dataset)}")
    logger.info(f"📊 Validation samples (files): {len(val_loader.dataset)}")

    # ------------- Model (input channels) -------------
    # Honor the same input-case policy as UNet/ViT experiments
    if args.input_case == "both":
        in_ch = 2
    else:
        in_ch = 2 if args.keep_two_channels else 1

    # cGAN options aligned for 3D and profiling stability
    gan_opt = SimpleNamespace(
        input_ch=in_ch, target_ch=1,
        n_gf=32, n_df=32,
        n_downsample=3, n_residual=6,
        norm_type="InstanceNorm3d",   # get_norm_layer expects this exact 3D class name
        padding_type="reflection",    # 3D-safe key ("reflection" | "replication" | "zero")
        trans_conv=True,              # Use ConvTranspose3d (avoid trilinear upsample warnings)
        n_D=3, ch_balance=0.0,
        lambda_LSGAN=1.0, lambda_FM=10.0, lambda_CC=5.0,
        n_CC=2, ccc=True, eps=1e-8,
        gpu_ids=0, data_type=32,
    )

    G = GeneratorPix2PixCC3D(gan_opt).to(args.device)
    D = MultiScaleDiscriminator3D(gan_opt).to(args.device)
    criterion = Pix2PixCCLoss3D(gan_opt)
    logger.info(f"🧱 cGAN created: G(in_ch={in_ch}, out_ch=1), D(n_D={gan_opt.n_D})")

    # ------------- Optimizers / Scheduler / AMP -------------
    use_amp = args.amp and str(args.device).startswith("cuda")

    optimizer_G = Adam(G.parameters(), lr=args.max_lr, betas=(0.5, 0.999))
    optimizer_D = Adam(D.parameters(), lr=args.max_lr * 0.5, betas=(0.5, 0.999))
    scheduler_G = get_clr_scheduler(optimizer_G, args.min_lr, args.max_lr, args.cycle_length)
    early_stopper = EarlyStopping(patience=args.patience, delta=args.es_delta)

    # AMP compatibility wrapper
    try:
        import torch.amp as amp
        scaler = amp.GradScaler("cuda") if use_amp else amp.GradScaler(enabled=False)

        def amp_autocast():
            if not use_amp:
                from contextlib import nullcontext
                return nullcontext()
            return amp.autocast("cuda", dtype=torch.float16)
    except Exception:
        from torch.cuda.amp import GradScaler as OldScaler, autocast as old_autocast
        scaler = OldScaler(enabled=use_amp)

        def amp_autocast():
            return old_autocast(enabled=use_amp)

    # ------------- Paths -------------
    os.makedirs(args.ckpt_dir, exist_ok=True)
    sample_percent = int(args.sample_fraction * 100)
    case_tag = f"icase-{args.input_case}{'-keep2' if args.keep_two_channels else ''}"
    model_prefix = (
        f"{case_tag}_cgan_tgt-{args.target_field}_"
        f"bs{args.batch_size}_clr[{args.min_lr:.0e}-{args.max_lr:.0e}]_"
        f"s{args.seed}_smp{sample_percent}"
    )
    best_G_path = os.path.join(args.ckpt_dir, f"{model_prefix}_G_best.pt")
    best_D_path = os.path.join(args.ckpt_dir, f"{model_prefix}_D_best.pt")
    final_G_path = os.path.join(args.ckpt_dir, f"{model_prefix}_G_final.pt")
    final_D_path = os.path.join(args.ckpt_dir, f"{model_prefix}_D_final.pt")
    log_path = os.path.join(args.ckpt_dir, f"{model_prefix}_log.csv")

    # ------------- Loop -------------
    log_records = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"🔁 Epoch {epoch+1}/{args.epochs} started.")
        G.train(); D.train()
        epoch_loss_G, epoch_loss_D, epoch_recon = 0.0, 0.0, 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        for step, (x, y) in enumerate(loop):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            x = select_inputs(x, args.input_case, args.keep_two_channels)

            optimizer_G.zero_grad(set_to_none=True)
            optimizer_D.zero_grad(set_to_none=True)

            # Forward + losses
            with amp_autocast():
                # Returns: (loss_D, loss_G, target_tensor, fake_tensor)
                loss_D, loss_G, y_real, y_fake = criterion(D, G, x, y)
                # Reconstruction proxy (for logging/monitoring)
                recon_mse = F.mse_loss(y_fake, y)

            # --- Update D ---
            if use_amp:
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
            else:
                loss_D.backward()
                optimizer_D.step()

            # --- Update G ---
            optimizer_G.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
            else:
                loss_G.backward()
                optimizer_G.step()

            # Accumulate
            bsz = x.size(0)
            epoch_loss_G += float(loss_G.detach()) * bsz
            epoch_loss_D += float(loss_D.detach()) * bsz
            epoch_recon  += float(recon_mse.detach()) * bsz

            if step % max(1, args.log_interval) == 0:
                loop.set_postfix(
                    loss_G=f"{float(loss_G):.5f}",
                    loss_D=f"{float(loss_D):.5f}",
                    recon=f"{float(recon_mse):.5f}"
                )

        # Epoch aggregates
        n_train = len(train_loader.dataset)
        avg_train_loss_G = epoch_loss_G / n_train
        avg_train_loss_D = epoch_loss_D / n_train
        avg_train_recon  = epoch_recon  / n_train
        scheduler_G.step()

        logger.info(f"📊 Train | G: {avg_train_loss_G:.6f} | D: {avg_train_loss_D:.6f} | Recon(MSE): {avg_train_recon:.6f}")

        # ------------- Validation (Generator-only MSE) -------------
        G.eval(); val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(args.device, non_blocking=True)
                y_val = y_val.to(args.device, non_blocking=True)
                x_val = select_inputs(x_val, args.input_case, args.keep_two_channels)

                with amp_autocast():
                    pred_val = G(x_val)
                    loss_val = F.mse_loss(pred_val, y_val)
                val_loss += float(loss_val) * x_val.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        current_lr_G = scheduler_G.get_last_lr()[0]
        logger.info(f"📉 Epoch {epoch+1:03d} | Val MSE: {avg_val_loss:.6f} | LR_G: {current_lr_G:.2e}")

        log_records.append({
            "epoch": epoch + 1,
            "train_loss_G": avg_train_loss_G,
            "train_loss_D": avg_train_loss_D,
            "train_recon_mse": avg_train_recon,
            "val_mse": avg_val_loss,
            "lr_G": current_lr_G,
        })

        # Checkpointing on best validation (MSE)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(G.state_dict(), best_G_path)
            torch.save(D.state_dict(), best_D_path)
            logger.info(f"✅ New best G/D saved (epoch {epoch+1})")

        # Early stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            logger.warning(f"🛑 Early stopping at epoch {epoch+1}")
            break

    # ------------- Save final -------------
    torch.save(G.state_dict(), final_G_path)
    torch.save(D.state_dict(), final_D_path)
    pd.DataFrame(log_records).to_csv(log_path, index=False)
    logger.info(f"📦 Final G saved: {final_G_path}")
    logger.info(f"📦 Final D saved: {final_D_path}")
    logger.info(f"📝 Training log saved: {log_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cGAN (Pix2PixCC-3D) on A-SIM 128^3.")
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--target_field", type=str, choices=["rho", "tscphi"], default="rho")
    parser.add_argument("--train_val_split", type=float, default=0.8)
    parser.add_argument("--sample_fraction", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", type=str2bool, default=True)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--cycle_length", type=int, default=8)

    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--es_delta", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt_dir", type=str, default="results/cgan/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # Input channel policy (same as UNet/ViT)
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both",
                        help="Select which input channels are provided to the model.")
    parser.add_argument("--keep_two_channels", action="store_true",
                        help="If set, keep in_channels=2 and zero-pad the missing channel for single-channel cases.")

    # Validation & file filtering
    parser.add_argument("--validate_keys", type=str2bool, default=True,
                        help="Pre-scan HDF5 to check required keys. Set False to skip (faster).")
    parser.add_argument("--exclude_list", type=str, default=None,
                        help="Path to a text file of bad HDF5 file paths to exclude.")
    parser.add_argument("--include_list", type=str, default=None,
                        help="Path to a text file of HDF5 file paths to include only.")

    args = parser.parse_args()

    try:
        train(args)
    except Exception:
        import traceback
        print("🔥 Training failed due to exception:")
        traceback.print_exc()
        sys.exit(1)
