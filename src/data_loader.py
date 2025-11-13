"""
src/data_loader.py

Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
Created: 2025-10-13
Modified: 2025-11-03 (Added exclude/include list support)

Description:
    Data loader for A-SIM HDF5 training/validation/test datasets.
    - Inputs: ngal (galaxy number density), vpec (peculiar velocity) → 2 channels
    - Targets: output_rho (default) or output_tscphi (optional; NOT output_phi)
      * If target == 'tscphi', multiply by (0.72**-2) and subtract its mean.

New Features:
    - exclude_list_path: text file listing files to skip
    - include_list_path: text file listing files to keep strictly
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import List, Tuple, Literal, Sequence, Optional

import yaml
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from src.logger import get_logger

logger = get_logger("data_loader", log_dir="logs")


# ----------------------------
# Utilities
# ----------------------------
def _natkey(path: str):
    tokens = re.split(r"(\d+)", os.path.basename(path))
    return tuple(int(t) if t.isdigit() else t for t in tokens)


def _load_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_split_files(
    yaml_cfg: dict,
    split: Literal["train", "val", "test"],
    train_val_split: float = 0.8,
) -> List[str]:
    base = yaml_cfg["asim_datasets_hdf5"]["base_path"]
    train_pattern = os.path.join(base, yaml_cfg["asim_datasets_hdf5"]["training_set"]["path"])
    test_pattern = os.path.join(base, yaml_cfg["asim_datasets_hdf5"]["validation_set"]["path"])

    train_files = sorted(glob(train_pattern), key=_natkey)
    test_files = sorted(glob(test_pattern), key=_natkey)
    if not train_files:
        raise FileNotFoundError(f"No HDF5 training files found in {train_pattern}")

    n_train_total = len(train_files)
    n_train_split = int(n_train_total * train_val_split)

    if split == "train":
        selected = train_files[:n_train_split]
    elif split == "val":
        selected = train_files[n_train_split:]
    elif split == "test":
        selected = test_files
    else:
        raise ValueError(f"Invalid split '{split}'. Use ['train','val','test'].")

    logger.info(
        f"📂 Split '{split}': {len(selected)} files "
        f"({n_train_split}/{n_train_total} train-val split, {len(test_files)} test files)"
    )
    return selected


# ----------------------------
# Shape normalization helpers
# ----------------------------
def _squeeze_leading_ones_to_nd(arr: np.ndarray, nd: int) -> np.ndarray:
    out = arr
    while out.ndim > nd and out.shape[0] == 1:
        out = out[0]
    return out


def _ensure_input_channels(arr: np.ndarray) -> np.ndarray:
    out = _squeeze_leading_ones_to_nd(arr, nd=4)
    if out.ndim != 4 or out.shape[0] != 2:
        raise ValueError(f"'input' must be (2,D,H,W); got {arr.shape} -> {out.shape}")
    return out


def _ensure_target_3d(arr: np.ndarray) -> np.ndarray:
    out = _squeeze_leading_ones_to_nd(arr, nd=3)
    if out.ndim != 3:
        raise ValueError(f"target must be 3D; got {arr.shape} -> {out.shape}")
    return out


# ----------------------------
# Dataset
# ----------------------------
class ASIMHDF5Dataset(Dataset):
    def __init__(
        self,
        file_paths: Sequence[str],
        target_field: Literal["rho", "tscphi"] = "rho",
        dtype: torch.dtype = torch.float32,
    ):
        self.file_paths = list(file_paths)
        self.target_field = target_field
        self.dtype = dtype
        assert target_field in ("rho", "tscphi")

        logger.info(f"🔍 ASIMHDF5Dataset initialized: {len(self.file_paths)} samples, target={target_field}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fpath = self.file_paths[idx]
        with h5py.File(fpath, "r") as f:
            if "input" not in f:
                raise KeyError(f"'input' dataset not found in {fpath}")
            x_np = _ensure_input_channels(f["input"][:])

            if self.target_field == "rho":
                if "output_rho" not in f:
                    raise KeyError(f"'output_rho' not found in {fpath}")
                y_np = _ensure_target_3d(f["output_rho"][:])
            else:
                if "output_tscphi" not in f:
                    raise KeyError(f"'output_tscphi' not found in {fpath}")
                y_np = _ensure_target_3d(f["output_tscphi"][:])
                y_np = y_np * (0.72 ** -2)
                y_np -= y_np.mean(dtype=np.float64)

        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(self.dtype)
        y = torch.from_numpy(np.ascontiguousarray(y_np)).to(self.dtype).unsqueeze(0)
        return x, y


# ----------------------------
# Validation & filtering
# ----------------------------
def _filter_files_by_keys(
    file_paths: Sequence[str],
    target_field: Literal["rho", "tscphi"],
    strict: bool = False,
) -> List[str]:
    req_target = "output_rho" if target_field == "rho" else "output_tscphi"
    kept, dropped = [], []
    for p in file_paths:
        try:
            with h5py.File(p, "r") as f:
                ok = ("input" in f) and (req_target in f)
            if ok:
                kept.append(p)
            else:
                dropped.append(p)
                if strict:
                    raise KeyError(f"Missing key(s) in {p}")
        except Exception as e:
            dropped.append(p)
            if strict:
                raise
            logger.warning(f"⚠️ Skip invalid file: {p} | {e}")

    if dropped:
        logger.warning(f"⚠️ Filtered out {len(dropped)} invalid file(s).")
    logger.info(f"✅ Valid files kept: {len(kept)} / {len(file_paths)}")
    return kept


# ----------------------------
# Public API: get_dataloader
# ----------------------------
# ----------------------------
# Public API: get_dataloader
# ----------------------------
def get_dataloader(
    yaml_path: str,
    split: Literal["train", "val", "test"],
    batch_size: int,
    shuffle: bool = True,
    sample_fraction: float = 1.0,
    num_workers: int = 0,
    pin_memory: bool = True,
    target_field: Literal["rho", "tscphi"] = "rho",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = 42,
    train_val_split: float = 0.8,
    validate_keys: bool = True,
    strict: bool = False,
    exclude_list_path: Optional[str] = None,
    include_list_path: Optional[str] = None,
) -> DataLoader:
    """
    Build DataLoader with optional file include/exclude lists.
    Automatically excludes known invalid HDF5 files.
    """
    cfg = _load_yaml(yaml_path)
    files = _resolve_split_files(cfg, split, train_val_split=train_val_split)

    # ----------------------------
    # (0) Hardcoded auto-exclude list (known broken files)
    # ----------------------------
    AUTO_EXCLUDE = {
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/1264.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/1265.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/1266.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10248.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10249.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10250.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10251.hdf5",
        "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/training/10252.hdf5",
    }

    before_auto = len(files)
    files = [f for f in files if f not in AUTO_EXCLUDE]
    removed_auto = before_auto - len(files)
    if removed_auto > 0:
        logger.warning(f"🚫 Auto-excluded {removed_auto} known broken files (A-SIM patch list).")

    # (1) include list (if provided)
    if include_list_path and os.path.exists(include_list_path):
        with open(include_list_path, "r", encoding="utf-8") as f:
            includes = {line.strip() for line in f if line.strip()}
        before = len(files)
        files = [f for f in files if f in includes]
        logger.info(f"✅ include_list applied ({len(files)}/{before}) from {include_list_path}")

    # (2) exclude list (if provided)
    if exclude_list_path and os.path.exists(exclude_list_path):
        with open(exclude_list_path, "r", encoding="utf-8") as f:
            excludes = {line.strip() for line in f if line.strip()}
        before = len(files)
        files = [f for f in files if f not in excludes]
        logger.info(f"🚫 exclude_list applied (removed {before - len(files)}) from {exclude_list_path}")

    # (3) validate HDF5 keys if requested
    if validate_keys:
        files = _filter_files_by_keys(files, target_field=target_field, strict=strict)
        if not files:
            raise RuntimeError(f"No valid HDF5 files remain for split='{split}' after validation.")

    # (4) Dataset
    dataset: Dataset = ASIMHDF5Dataset(files, target_field=target_field, dtype=dtype)

    # (5) Sample fraction
    if 0.0 < sample_fraction < 1.0:
        total_len = len(dataset)
        sample_size = max(1, int(round(sample_fraction * total_len)))
        split_offset = {"train": 0, "val": 1, "test": 2}[split]
        rng = np.random.default_rng((seed or 0) + split_offset)
        indices = np.sort(rng.choice(total_len, size=sample_size, replace=False))
        dataset = Subset(dataset, indices)
        logger.info(f"🔎 Sub-sampled {sample_size}/{total_len} ({sample_fraction*100:.1f}%)")

    logger.info(f"📦 Split='{split}' | files={len(files)} | batch={batch_size} | target='{target_field}'")

    # --- worker별 시드 고정 (재현성) ---
    def _worker_init_fn(worker_id: int):
        if seed is not None:
            base = int(seed) + worker_id
            np.random.seed(base)

    # --- split별 셔플 정책 및 성능 옵션 ---
    effective_shuffle = (split == "train") and shuffle
    kwargs = dict(
        batch_size=batch_size,
        shuffle=effective_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    if pin_memory and torch.cuda.is_available():
        kwargs["pin_memory_device"] = "cuda"

    return DataLoader(dataset, **kwargs)



# ----------------------------
# Sanity check utility
# ----------------------------
def sanity_check_sample(
    yaml_path: str,
    split: str = "train",
    idx: int = 0,
    target_field: Literal["rho", "tscphi"] = "rho",
):
    cfg = _load_yaml(yaml_path)
    files = _resolve_split_files(cfg, split)
    files = _filter_files_by_keys(files, target_field=target_field, strict=False)

    if not (0 <= idx < len(files)):
        raise IndexError(f"idx out of range for split '{split}': 0..{len(files)-1}")

    path = files[idx]
    with h5py.File(path, "r") as f:
        x = _ensure_input_channels(f["input"][:])
        if target_field == "rho":
            y = _ensure_target_3d(f["output_rho"][:])
        else:
            y = _ensure_target_3d(f["output_tscphi"][:])
            y = y * (0.72 ** -2)
            y = y - y.mean(dtype=np.float64)

    logger.info(
        f"[SanityCheck] {split}[{idx}] = {os.path.basename(path)} | "
        f"x.shape={x.shape}, y.shape={y.shape} | "
        f"x stats: min={np.min(x):.4g}, max={np.max(x):.4g}, mean={np.mean(x):.4g} | "
        f"y stats: min={np.min(y):.4g}, max={np.max(y):.4g}, mean={np.mean(y):.4g}"
    )
