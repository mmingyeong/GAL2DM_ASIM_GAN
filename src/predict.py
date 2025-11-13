# models/gan/predict.py
# -*- coding: utf-8 -*-
"""
Predict with cGAN Generator (Pix2PixCC3D) for 3D voxel-wise regression and save to HDF5.

- Uses the trained Generator only (no Discriminator needed at inference).
- Channel-ablation inference supported:
    --input_case {both,ch1,ch2}
    --keep_two_channels  (keep in_channels=2 and zero-pad missing channel)
- Matches the evaluation I/O convention used by UNet/ViT experiments.

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-11-12
"""

from __future__ import annotations
import os, sys, argparse, yaml, h5py, torch, numpy as np
from typing import List
from glob import glob
from contextlib import nullcontext
from types import SimpleNamespace
from tqdm import tqdm

# project root import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.logger import get_logger
from src.model import GeneratorPix2PixCC3D  # Generator only

logger = get_logger("predict_cgan3d")


# ----------------------------
# Helpers
# ----------------------------
def _natkey(path: str):
    import re
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def _load_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_test_files(yaml_cfg: dict) -> List[str]:
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    test_rel = yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"]  # e.g., test/*.hdf5
    pattern = os.path.join(base, test_rel)
    files = sorted(glob(pattern), key=_natkey)
    if not files:
        raise FileNotFoundError(f"No test HDF5 files matched: {pattern}")
    return files


def _ensure_input_shape(x: np.ndarray) -> np.ndarray:
    """
    Normalize to (N,C,D,H,W) with C in {1,2}.
    Accepts: (C,D,H,W), (N,C,D,H,W).
    """
    a = np.asarray(x)
    while a.ndim > 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 4 and a.shape[0] in (1, 2):      # (C,D,H,W)
        a = a[None, ...]                           # (1,C,D,H,W)
    elif a.ndim == 5 and a.shape[1] in (1, 2):     # (N,C,D,H,W)
        pass
    else:
        raise ValueError(f"Unsupported 'input' shape: {a.shape}")
    return a


def _find_input_dataset(h5: h5py.File) -> str | None:
    candidate_keys = [
        "input", "inputs", "X",
        "data/input", "dataset/input", "features/input",
    ]
    for k in candidate_keys:
        if k in h5:
            return k
    for k in candidate_keys:
        if "/" in k:
            grp, dset = k.split("/", 1)
            if grp in h5 and dset in h5[grp]:
                return k
    lowered = {kk.lower(): kk for kk in h5.keys()}
    for name in ("input", "inputs", "x"):
        if name in lowered:
            return lowered[name]
    return None


def _load_checkpoint(model_path: str, device: torch.device):
    """Safe checkpoint load (supports plain state_dict or wrapped)."""
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        state = torch.load(model_path, map_location=device)
    except Exception as e:
        logger.warning(f"weights_only load failed with {e}; falling back to standard torch.load")
        state = torch.load(model_path, map_location=device)

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    return state


def select_inputs(x: torch.Tensor, case: str, keep_two: bool) -> torch.Tensor:
    """
    x: [N,C,D,H,W] with C in {1,2}
    case: "both" | "ch1" | "ch2"
    keep_two=True  -> always return 2-ch with a zero-padded missing channel
    keep_two=False -> return a single channel tensor for ch1/ch2
    """
    assert x.ndim == 5 and x.size(1) in (1, 2), f"Expected [N,1or2,D,H,W], got {tuple(x.shape)}"

    if case == "both":
        if x.size(1) != 2:
            raise ValueError(f"--input_case both requires 2 channels, but got {x.size(1)}")
        return x

    if case == "ch1":
        if x.size(1) == 2:
            if keep_two:
                ch1 = x[:, 0:1]
                z   = torch.zeros_like(ch1)
                return torch.cat([ch1, z], dim=1)
            else:
                return x[:, 0:1]
        else:
            return x if not keep_two else torch.cat([x, torch.zeros_like(x)], dim=1)

    if case == "ch2":
        if x.size(1) == 2:
            if keep_two:
                ch2 = x[:, 1:2]
                z   = torch.zeros_like(ch2)
                return torch.cat([z, ch2], dim=1)
            else:
                return x[:, 1:2]
        else:
            return torch.cat([torch.zeros_like(x), x], dim=1) if keep_two else x

    raise ValueError(case)


# ----------------------------
# Inference
# ----------------------------
def run_prediction(
    yaml_path: str,
    output_dir: str,
    model_path: str,
    device: str = "cuda",
    batch_size: int = 1,
    amp: bool = False,
    sample_fraction: float = 1.0,
    sample_seed: int = 42,
    input_case: str = "both",
    keep_two_channels: bool = False,
    on_missing_input: str = "skip",  # "skip" | "stop"
    # cGAN generator init options (defaults match train.py)
    norm_type: str = "InstanceNorm3d",
    padding_type: str = "reflection",  # one of ['reflection','replication','zero']
    trans_conv: bool = True,           # True -> ConvTranspose3d upsampling
    n_gf: int = 32,
    n_residual: int = 6,
    n_downsample: int = 3,
):
    """
    Run inference on HDF5 test files using the cGAN Generator (Pix2PixCC3D).
    """
    if not (0 < sample_fraction <= 1.0):
        raise ValueError(f"--sample_fraction must be in (0,1], got {sample_fraction}")

    # case-specific subdir to avoid mixing outputs
    case_suffix = f"icase-{input_case}{'-keep2' if keep_two_channels else ''}"
    output_dir = os.path.join(output_dir, case_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # RNG for reproducible file subsampling
    rng = np.random.default_rng(sample_seed)

    # 1) Resolve test set
    cfg = _load_yaml(yaml_path)
    test_files = _resolve_test_files(cfg)

    if sample_fraction < 1.0:
        n_total = len(test_files)
        n_keep = max(1, int(np.ceil(sample_fraction * n_total)))
        keep_idx = np.sort(rng.choice(n_total, size=n_keep, replace=False))
        test_files = [test_files[i] for i in keep_idx]
        logger.info(f"🧪 Test files subsampled: {n_keep}/{n_total} (fraction={sample_fraction:.3f})")
    else:
        logger.info(f"🧪 Test files: {len(test_files)} found from YAML (no subsampling).")

    # 2) Build Generator & load weights
    dev = torch.device(device)
    if input_case == "both":
        in_ch = 2
    else:
        in_ch = 2 if keep_two_channels else 1

    gan_opt = SimpleNamespace(
        input_ch=in_ch, target_ch=1,
        n_gf=n_gf, n_df=32,
        n_downsample=n_downsample, n_residual=n_residual,
        norm_type=norm_type,          # 3D class name expected by get_norm_layer
        padding_type=padding_type,    # 3D-safe padding key
        trans_conv=trans_conv,        # avoid trilinear warnings when True
        n_D=3, ch_balance=0.0,
        lambda_LSGAN=1.0, lambda_FM=10.0, lambda_CC=5.0,
        n_CC=2, ccc=True, eps=1e-8,
        gpu_ids=0, data_type=32,
    )
    model = GeneratorPix2PixCC3D(gan_opt).to(dev)
    logger.info(f"🧱 Model: cGAN-Generator(in_ch={in_ch}, out_ch=1, norm={norm_type}, pad={padding_type}, trans_conv={trans_conv})")

    logger.info(f"📥 Loading checkpoint: {model_path}")
    state = _load_checkpoint(model_path, dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys while loading: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading: {unexpected}")
    model.eval()

    # AMP context
    try:
        _ = torch.amp
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16) if (amp and dev.type == "cuda")
            else torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16) if (amp and dev.type == "cpu")
            else nullcontext()
        )
    except Exception:
        from torch.cuda.amp import autocast as legacy_autocast
        autocast_ctx = legacy_autocast(enabled=amp)

    # 3) Predict per file
    saved_files: list[str] = []
    skipped_files: list[str] = []

    torch.set_grad_enabled(False)
    with torch.no_grad():
        for input_path in tqdm(test_files, desc="🚀 Running cGAN test predictions"):
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_dir, filename)

            # Load inputs
            with h5py.File(input_path, "r") as f:
                key = _find_input_dataset(f)
                if key is None:
                    msg = f"No input-like dataset found in {input_path}"
                    if on_missing_input == "stop":
                        raise KeyError(msg)
                    logger.warning(f"⚠️ {msg} — SKIP")
                    skipped_files.append(input_path)
                    continue

                try:
                    x_np = f[key][:]
                except Exception as e:
                    msg = f"Failed to read dataset '{key}' in {input_path}: {e}"
                    if on_missing_input == "stop":
                        raise RuntimeError(msg)
                    logger.warning(f"⚠️ {msg} — SKIP")
                    skipped_files.append(input_path)
                    continue

            # Normalize to (N,C,D,H,W)
            try:
                x_np = _ensure_input_shape(x_np)
            except Exception as e:
                msg = f"Invalid input shape in {input_path}: {e}"
                if on_missing_input == "stop":
                    raise
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            # To tensor & select channels
            x_tensor = torch.from_numpy(np.ascontiguousarray(x_np)).float().to(dev)
            try:
                x_tensor = select_inputs(x_tensor, input_case, keep_two_channels)  # [N,in_ch,D,H,W]
            except Exception as e:
                msg = f"Channel selection failed for {input_path}: {e}"
                if on_missing_input == "stop":
                    raise
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            if x_tensor.size(1) != in_ch:
                msg = f"Post-selection channels {x_tensor.size(1)} != model.in_channels {in_ch} in {input_path}"
                if on_missing_input == "stop":
                    raise RuntimeError(msg)
                logger.warning(f"⚠️ {msg} — SKIP")
                skipped_files.append(input_path)
                continue

            # Batched inference
            preds = []
            for i in range(0, x_tensor.shape[0], batch_size):
                x_batch = x_tensor[i: i + batch_size]
                with autocast_ctx:
                    y_batch = model(x_batch)  # [B,1,D,H,W]
                preds.append(y_batch.float().cpu().numpy())

            y_pred = np.concatenate(preds, axis=0)  # (N,1,D,H,W)
            y_pred = np.squeeze(y_pred, axis=1)     # (N,D,H,W) or (D,H,W) if N==1

            # Save prediction
            with h5py.File(output_path, "w") as f_out:
                f_out.create_dataset("prediction", data=y_pred, compression="gzip")
                # Meta-info
                f_out.attrs["source_file"] = input_path
                f_out.attrs["model_path"] = model_path
                f_out.attrs["model_class"] = model.__class__.__name__
                f_out.attrs["amp"] = bool(amp)
                f_out.attrs["input_case"] = str(input_case)
                f_out.attrs["keep_two_channels"] = bool(keep_two_channels)
                f_out.attrs["norm_type"] = str(norm_type)
                f_out.attrs["padding_type"] = str(padding_type)
                f_out.attrs["trans_conv"] = bool(trans_conv)

            logger.info(f"✅ Saved: {output_path}")
            saved_files.append(output_path)

    # Summary
    logger.info("====== Inference Summary ======")
    logger.info(f"Saved files : {len(saved_files)}")
    logger.info(f"Skipped     : {len(skipped_files)}")
    if skipped_files:
        logger.info("Skipped list (first 20): " + ", ".join(os.path.basename(p) for p in skipped_files[:20]))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cGAN Generator (Pix2PixCC3D) inference on A-SIM test files.")

    # Data / Paths
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to asim_paths.yaml")
    parser.add_argument("--output_dir", type=str, required=True, help="Root dir to save predictions (per-case subdir created)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained GENERATOR .pt file (e.g., *_G_best.pt)")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision inference")

    # Subsampling
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction (0,1] of TEST FILES to run")
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for file-level subsampling")

    # Channel ablation flags
    parser.add_argument("--input_case", type=str, choices=["both", "ch1", "ch2"], default="both",
                        help="Which input channels are provided to the model")
    parser.add_argument("--keep_two_channels", action="store_true",
                        help="Keep in_channels=2 and zero-pad the missing channel for single-channel cases")

    # Missing-input handling policy
    parser.add_argument("--on_missing_input", type=str, choices=["skip", "stop"], default="skip",
                        help="When an HDF5 file lacks the input dataset: skip or stop with error")

    # cGAN generator init options (kept flexible; defaults mirror train.py)
    parser.add_argument("--norm_type", type=str, default="InstanceNorm3d",
                        help="Normalization type expected by get_norm_layer (3D class name)")
    parser.add_argument("--padding_type", type=str, default="reflection",
                        help="3D padding: 'reflection'|'replication'|'zero'")
    parser.add_argument("--trans_conv", action="store_true", default=True,
                        help="Use ConvTranspose3d upsampling (avoids trilinear warnings)")
    parser.add_argument("--n_gf", type=int, default=32, help="Generator base channels")
    parser.add_argument("--n_residual", type=int, default=6, help="Number of ResBlocks at bottleneck")
    parser.add_argument("--n_downsample", type=int, default=3, help="Number of downsample stages")

    args = parser.parse_args()

    run_prediction(
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        amp=args.amp,
        sample_fraction=args.sample_fraction,
        sample_seed=args.sample_seed,
        input_case=args.input_case,
        keep_two_channels=args.keep_two_channels,
        on_missing_input=args.on_missing_input,
        norm_type=args.norm_type,
        padding_type=args.padding_type,
        trans_conv=args.trans_conv,
        n_gf=args.n_gf,
        n_residual=args.n_residual,
        n_downsample=args.n_downsample,
    )
