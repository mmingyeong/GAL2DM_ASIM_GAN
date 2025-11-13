#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Evaluation Script (Standalone)

Computes three tables (FULL evaluation, not FAST):
  1) Prediction Accuracy Summary
  2) Distribution & Bias Diagnostics
  3) Structural Consistency (P(k), T(k), ξ(r))

Outputs:
  - prediction_accuracy_full.csv
  - distribution_bias_full.csv
  - structural_consistency_full.csv
  - structural_consistency_curves_full.npz

Notes:
  - Paths & basic config are defined near the top.
  - No TensorFlow/PyTorch models are constructed here; we only READ prediction files.
  - P(k), cross-P(k), ξ(r) are computed via FFTs with spherical binning.
"""

import os, re, glob, time, itertools
import numpy as np
import pandas as pd
import h5py
from typing import List, Tuple, Optional, Dict

from scipy.stats import pearsonr, ks_2samp, wasserstein_distance
from scipy.ndimage import uniform_filter

# Progress bar
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

# =========================
# Configuration (EDIT ME)
# =========================
BASE_PRED_DIR = "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions"   # Base model .npy
UNET_PRED_DIR = "/home/mingyeong/GAL2DM_ASIM_VNET/results/unet_predictions/28845/icase-both-keep2"
VIT_PRED_DIR  = "/home/mingyeong/GAL2DM_ASIM_ViT/results/vit_predictions/28846/icase-both"
TRUE_PATH_TPL = "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/{idx}.hdf5"

# Base .npy are stored as log10(rho)/2.5
BASE_IS_LOG10_OVER_2P5 = True

# Box size for spectra/correlation (h^-1 Mpc)
BOX_SIZE = 160.0

# Radial bins (ξ) and k-bins (P(k))
N_R_BINS = 25
N_K_BINS = 40

# =========================
# Lightweight Logger
# =========================
_RUN_START = time.time()
def _hhmmss(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def log(msg: str) -> None:
    dt = time.time() - _RUN_START
    print(f"[{_hhmmss(dt)}] {msg}", flush=True)

def log_kv(title: str, kv: Dict) -> None:
    dt = time.time() - _RUN_START
    items = " | ".join(f"{k}={v}" for k,v in kv.items())
    print(f"[{_hhmmss(dt)}] {title} :: {items}", flush=True)

# =========================
# Array shape utilities
# =========================
def to_DHW(a: np.ndarray) -> np.ndarray:
    """(N,C,D,H,W)/(C,D,H,W)/(N,D,H,W)/(H,W,D)/(D,H,W) -> (D,H,W)"""
    a = np.asarray(a)
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    if   a.ndim == 5: a = a[0,0]
    elif a.ndim == 4: a = a[0]
    elif a.ndim != 3: raise ValueError(f"Unsupported shape: {a.shape}")
    # (H,W,D) -> (D,H,W)
    if a.shape[2] == max(a.shape) and a.shape[0] != max(a.shape):
        a = a.transpose(2,0,1)
    return a

def squeeze_to_3d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    if a.ndim == 5: a = a[0,0]
    elif a.ndim == 4: a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expect 3D, got {a.shape}")
    return a

def permute_to_shape(a: np.ndarray, ref_shape: Tuple[int,int,int]) -> Optional[np.ndarray]:
    if a.shape == ref_shape:
        return a
    if sorted(a.shape) == sorted(ref_shape):
        for perm in itertools.permutations(range(3)):
            if tuple(np.array(a.shape)[list(perm)]) == tuple(ref_shape):
                return np.transpose(a, axes=perm)
    return None

def center_crop_to(a: np.ndarray, ref_shape: Tuple[int,int,int], tol: int = 2) -> Optional[np.ndarray]:
    diffs = np.array(a.shape) - np.array(ref_shape)
    if np.all(np.abs(diffs) <= tol):
        zs, ys, xs = a.shape
        rz, ry, rx = ref_shape
        z0 = (zs - rz)//2; y0 = (ys - ry)//2; x0 = (xs - rx)//2
        return a[z0:z0+rz, y0:y0+ry, x0:x0+rx]
    return None

def harmonize_to_ref(a: np.ndarray, ref_shape: Tuple[int,int,int], tol: int = 2) -> Optional[np.ndarray]:
    try:
        a = squeeze_to_3d(a)
    except Exception:
        return None
    if a.shape == ref_shape:
        return a
    b = permute_to_shape(a, ref_shape)
    if b is not None:
        return b
    c = center_crop_to(a, ref_shape, tol)
    if c is not None:
        return c
    return None

# =========================
# Index discovery
# =========================
def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _first_int(text: str, default=None):
    m = re.findall(r"\d+", text)
    return int(m[0]) if m else default

def list_true_indices() -> List[str]:
    tdir = os.path.dirname(TRUE_PATH_TPL.format(idx="0"))
    files = glob.glob(os.path.join(tdir, "*.hdf5"))
    return [_stem(p) for p in files]

def list_unet_indices() -> List[str]:
    files = glob.glob(os.path.join(UNET_PRED_DIR, "*.hdf5"))
    return [_stem(p) for p in files]

def list_vit_indices() -> List[str]:
    files = glob.glob(os.path.join(VIT_PRED_DIR, "*.hdf5"))
    return [_stem(p) for p in files]

def list_base_indices() -> List[str]:
    files = glob.glob(os.path.join(BASE_PRED_DIR, "test_*.npy"))
    idxs = []
    for p in files:
        m = re.search(r"test_(\d+)_", os.path.basename(p))
        if m: idxs.append(m.group(1))
    return idxs

def common_indices() -> List[str]:
    T = set(list_true_indices())
    U = set(list_unet_indices())
    V = set(list_vit_indices())
    B = set(list_base_indices())
    return sorted(T & U & V & B, key=_first_int)

# =========================
# Loaders
# =========================
def load_true(idx: str) -> np.ndarray:
    with h5py.File(TRUE_PATH_TPL.format(idx=idx), "r") as f:
        a = to_DHW(f["output_rho"][:])
    return np.clip(a, 0, None)

def load_unet(idx: str) -> np.ndarray:
    with h5py.File(os.path.join(UNET_PRED_DIR, f"{idx}.hdf5"), "r") as f:
        a = to_DHW(f["prediction"][:])
    return np.clip(a, 0, None)

def load_vit(idx: str) -> np.ndarray:
    with h5py.File(os.path.join(VIT_PRED_DIR, f"{idx}.hdf5"), "r") as f:
        a = to_DHW(f["prediction"][:])
    return np.clip(a, 0, None)

def load_base(idx: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(BASE_PRED_DIR, f"test_{idx}_*.npy")))
    if not files: raise FileNotFoundError(f"Base not found for idx={idx}")
    arr = np.load(files[0])
    rho = 10**(2.5*arr) if BASE_IS_LOG10_OVER_2P5 else arr
    return np.clip(to_DHW(rho), 0, None)

LOADERS = {
    "Base Model": load_base,
    "V-NET (UNet3D)": load_unet,
    "ViT (3D Transformer)": load_vit,
}

# =========================
# Metrics
# =========================
def RMSE(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth)**2)))

def MAE(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))

def Pearson(pred: np.ndarray, truth: np.ndarray) -> float:
    r = pearsonr(pred.ravel(), truth.ravel())[0]
    return float(0.0 if np.isnan(r) else r)

def R2(pred: np.ndarray, truth: np.ndarray) -> float:
    ss_res = np.sum((truth - pred)**2)
    ss_tot = np.sum((truth - np.mean(truth))**2)
    val = 1.0 - ss_res/ss_tot if ss_tot != 0 else np.nan
    return float(val)

# Lightweight SSIM (2D) using uniform_filter; averaged over slices
def _ssim2d(x: np.ndarray, y: np.ndarray, win: int = 7) -> float:
    x = x.astype(np.float64); y = y.astype(np.float64)
    # dynamic range per slice
    Lx = max(1e-8, x.max() - x.min()); Ly = max(1e-8, y.max() - y.min())
    L = 0.5*(Lx + Ly)
    C1 = (0.01*L)**2
    C2 = (0.03*L)**2
    ux  = uniform_filter(x, size=win)
    uy  = uniform_filter(y, size=win)
    uxx = uniform_filter(x*x, size=win)
    uyy = uniform_filter(y*y, size=win)
    uxy = uniform_filter(x*y, size=win)
    vx  = uxx - ux*ux
    vy  = uyy - uy*uy
    vxy = uxy - ux*uy
    num = (2*ux*uy + C1)*(2*vxy + C2)
    den = (ux*ux + uy*uy + C1)*(vx + vy + C2)
    ssim_map = num / (den + 1e-12)
    return float(np.nanmean(ssim_map))

def SSIM3D(pred: np.ndarray, truth: np.ndarray, slices: int = None) -> float:
    """Average 2D SSIM over all slices along z (or a subset)."""
    zdim = pred.shape[0]
    if slices is None or slices >= zdim:
        idxs = range(zdim)
    else:
        step = max(1, zdim // slices)
        idxs = range(0, zdim, step)
    vals = []
    for i in idxs:
        vals.append(_ssim2d(pred[i], truth[i]))
    return float(np.mean(vals)) if len(vals) else float("nan")

# =========================
# Spectra / Correlation utils
# =========================
def _delta_from_rho(rho: np.ndarray) -> np.ndarray:
    m = np.mean(rho)
    if m <= 0:
        return np.zeros_like(rho, dtype=np.float64)
    return (rho / m) - 1.0

def _kgrid(nz, ny, nx, box_size):
    # build k-grid magnitudes
    kx = np.fft.fftfreq(nx, d=box_size/nx) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=box_size/ny) * 2*np.pi
    kz = np.fft.fftfreq(nz, d=box_size/nz) * 2*np.pi
    kkx, kky, kkz = np.meshgrid(kx, ky, kz, indexing="xy")  # shape (ny,nx,nz) but we need (nz,ny,nx)
    kkx = np.transpose(kkx, (2,0,1))
    kky = np.transpose(kky, (2,0,1))
    kkz = np.transpose(kkz, (2,0,1))
    kk = np.sqrt(kkx*kkx + kky*kky + kkz*kkz)
    return kk

def _bin_means(values: np.ndarray, radii: np.ndarray, edges: np.ndarray):
    """Return (bin_means, counts, bin_centers) with robust masking.

    - Drops entries with NaNs/Infs in either array.
    - Ignores samples falling outside the provided bin edges (e.g., k=0 if edges start > 0).
    """
    values = np.asarray(values, dtype=np.float64)
    radii  = np.asarray(radii,  dtype=np.float64)

    # finite mask
    m = np.isfinite(values) & np.isfinite(radii)
    if not np.any(m):
        nb = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.full(nb, np.nan), np.zeros(nb, dtype=np.int64), centers

    # digitize only the valid part
    inds_all = np.digitize(radii[m], edges) - 1  # 0..nbins-1 expected
    nb = len(edges) - 1

    # keep only indices within [0, nb-1]
    keep = (inds_all >= 0) & (inds_all < nb)
    if not np.any(keep):
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.full(nb, np.nan), np.zeros(nb, dtype=np.int64), centers

    inds = inds_all[keep]
    w    = values[m][keep]

    sums   = np.bincount(inds, weights=w, minlength=nb).astype(np.float64)
    counts = np.bincount(inds,           minlength=nb).astype(np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return means, counts, centers

def compute_pk_cross(truth_rho: np.ndarray, pred_rho: np.ndarray, nbins_k: int = N_K_BINS):
    t = _delta_from_rho(truth_rho)
    p = _delta_from_rho(pred_rho)
    nz, ny, nx = t.shape
    vol = BOX_SIZE**3

    Ft = np.fft.fftn(t)
    Fp = np.fft.fftn(p)
    Ptt3d = (Ft * np.conj(Ft)).real / vol
    Ppp3d = (Fp * np.conj(Fp)).real / vol
    Ptp3d = (Ft * np.conj(Fp)).real / vol

    kk = _kgrid(nz, ny, nx, BOX_SIZE)

    # define edges; include k=0 explicitly as the first edge
    kpos = kk[kk > 0]
    if kpos.size == 0:
        raise ValueError("All k are zero? Check BOX_SIZE and grid.")
    kmin = float(np.min(kpos))
    kmax = float(np.max(kk))
    k_edges_geom = np.geomspace(kmin, kmax, nbins_k)     # length = nbins_k
    k_edges = np.concatenate(([0.0], k_edges_geom))      # length = nbins_k+1; bins = nbins_k

    Ptt_1d, cnt, k_centers = _bin_means(Ptt3d, kk, k_edges)
    Ppp_1d, _,   _        = _bin_means(Ppp3d, kk, k_edges)
    Ptp_1d, _,   _        = _bin_means(Ptp3d, kk, k_edges)

    return k_centers, Ptt_1d, Ppp_1d, Ptp_1d


def compute_xi(truth_rho: np.ndarray, pred_rho: np.ndarray, nbins_r: int = N_R_BINS):
    """Return r_centers, xi_true(r), xi_pred(r) from FFT-based autocorrelation with spherical binning."""
    t = _delta_from_rho(truth_rho)
    p = _delta_from_rho(pred_rho)
    nz, ny, nx = t.shape
    vol = BOX_SIZE**3

    # auto-correlation via Wiener-Khinchin: xi = IFFT( |F|^2 ) / V
    Ft = np.fft.fftn(t);  xi_t = np.fft.ifftn(Ft*np.conj(Ft)).real / (nz*ny*nx)
    Fp = np.fft.fftn(p);  xi_p = np.fft.ifftn(Fp*np.conj(Fp)).real / (nz*ny*nx)

    # shift to center
    xi_t = np.fft.fftshift(xi_t)
    xi_p = np.fft.fftshift(xi_p)

    # build r-grid in physical units
    dz = dy = dx = BOX_SIZE / nz
    z = (np.arange(nz) - nz//2) * dz
    y = (np.arange(ny) - ny//2) * dy
    x = (np.arange(nx) - nx//2) * dx
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")
    rr = np.sqrt(xx*xx + yy*yy + zz*zz)

    r_edges = np.linspace(0, 0.5*BOX_SIZE, nbins_r+1)
    xi_t_1d, cnt, r_centers = _bin_means(xi_t, rr, r_edges)
    xi_p_1d, _,   _         = _bin_means(xi_p, rr, r_edges)

    return r_centers, xi_t_1d, xi_p_1d

# =========================
# 1) FULL: Prediction Accuracy Summary
# =========================
def evaluate_prediction_accuracy_full(models: List[str]) -> pd.DataFrame:
    idx_list = common_indices()
    if len(idx_list) == 0:
        raise RuntimeError("No common indices. Check paths/filenames.")

    ref_truth = load_true(idx_list[0])
    ref_shape = tuple(ref_truth.shape)
    log_kv("PRED_FULL_CONFIG", {"num_common_indices": len(idx_list), "ref_shape": ref_shape})

    results: List[Dict] = []

    for model_name in models:
        loader = LOADERS[model_name]
        sums = {"RMSE":0.0, "MAE":0.0, "Pearson r":0.0, "R²":0.0, "SSIM (3D)":0.0}
        count_ok = 0

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as p:
            task = p.add_task(f"{model_name} [FULL]", total=len(idx_list))

            for idx in idx_list:
                t = harmonize_to_ref(load_true(idx),  ref_shape)
                y = harmonize_to_ref(loader(idx),     ref_shape)
                if t is None or y is None:
                    p.advance(task); continue
                if not np.isfinite(t).all() or not np.isfinite(y).all():
                    p.advance(task); continue

                sums["RMSE"]      += RMSE(y, t)
                sums["MAE"]       += MAE(y, t)
                sums["Pearson r"] += Pearson(y, t)
                sums["R²"]        += R2(y, t)
                sums["SSIM (3D)"] += SSIM3D(y, t)  # all slices
                count_ok += 1
                p.advance(task)

        if count_ok == 0:
            row = {"Model": model_name, "RMSE":np.nan,"MAE":np.nan,"Pearson r":np.nan,"R²":np.nan,"SSIM (3D)":np.nan}
        else:
            row = {"Model": model_name}
            row.update({k: v / count_ok for k,v in sums.items()})
        results.append(row)

    df = pd.DataFrame(results, columns=["Model","RMSE","MAE","Pearson r","R²","SSIM (3D)"])
    df.to_csv("prediction_accuracy_full.csv", index=False)
    log_kv("SAVED", {"prediction_accuracy_full.csv": "OK"})
    return df

# =========================
# 2) FULL: Distribution & Bias Diagnostics
# =========================
def evaluate_distribution_bias_full(models: List[str], bins: int = 200) -> pd.DataFrame:
    idx_list = common_indices()
    if len(idx_list) == 0:
        raise RuntimeError("No common indices. Check paths/filenames.")

    ref_truth = load_true(idx_list[0])
    ref_shape = tuple(ref_truth.shape)
    log_kv("DIST_FULL_CONFIG", {"num_common_indices": len(idx_list), "ref_shape": ref_shape})

    results: List[Dict] = []

    for model_name in models:
        loader = LOADERS[model_name]
        logs = []
        true_pool = []
        pred_pool = []

        with Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as p:
            task = p.add_task(f"{model_name} [DIST FULL]", total=len(idx_list))

            for idx in idx_list:
                t = harmonize_to_ref(load_true(idx),  ref_shape)
                y = harmonize_to_ref(loader(idx),     ref_shape)
                if t is None or y is None:
                    p.advance(task); continue

                # avoid log of zero
                t = np.clip(t, 1e-12, None)
                y = np.clip(y, 1e-12, None)

                logs.append(np.log10(y / t))
                true_pool.append(t.ravel())
                pred_pool.append(y.ravel())

                p.advance(task)

        logs = np.concatenate(logs) if len(logs) else np.array([np.nan])
        true_pool = np.concatenate(true_pool) if len(true_pool) else np.array([np.nan])
        pred_pool = np.concatenate(pred_pool) if len(pred_pool) else np.array([np.nan])

        # Histogram intersection (mass-wise overlap)
        # Normalize by sum(true) to get [0,1]-like scale
        denom = np.sum(true_pool)
        hist_intersection = float(np.sum(np.minimum(true_pool, pred_pool)) / denom) if denom > 0 else np.nan

        row = {
            "Model": model_name,
            "log_bias": float(np.nanmean(logs)),
            "log_std":  float(np.nanstd(logs)),
            "KS":       float(ks_2samp(pred_pool, true_pool).statistic) if np.all(np.isfinite([pred_pool.mean(), true_pool.mean()])) else np.nan,
            "Hist_Intersection": hist_intersection,
            "Wasserstein": float(wasserstein_distance(true_pool, pred_pool)) if np.all(np.isfinite([pred_pool.mean(), true_pool.mean()])) else np.nan,
        }
        results.append(row)

    df = pd.DataFrame(results, columns=["Model","log_bias","log_std","KS","Hist_Intersection","Wasserstein"])
    df.to_csv("distribution_bias_full.csv", index=False)
    log_kv("SAVED", {"distribution_bias_full.csv": "OK"})
    return df

# =========================
# 3) FULL: Structural Consistency
# =========================
def evaluate_structural_consistency_full(models: List[str], nbins_k: int = N_K_BINS, nbins_r: int = N_R_BINS,
                                         max_indices: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
    idx_list = common_indices()
    if len(idx_list) == 0:
        raise RuntimeError("No common indices. Check paths/filenames.")

    # FULL 평가이지만, 지나치게 큰 셋이면 상한을 두는 옵션 제공
    if max_indices is not None:
        idx_list = idx_list[:max_indices]

    # reference shapes just for logs
    ref_truth = load_true(idx_list[0])
    ref_shape = tuple(ref_truth.shape)
    log_kv("STRUCT_FULL_CONFIG", {"num_indices": len(idx_list), "ref_shape": ref_shape, "nbins_k": nbins_k, "nbins_r": nbins_r})

    results: List[Dict] = []
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in models:
        loader = LOADERS[model_name]
        T_list = []   # correlation coefficient in k (T(k) = P_tp / sqrt(P_tt P_pp))
        xi_list = []  # ξ_pred/ξ_true ratio curves (as a function of r)
        k_ref, r_ref = None, None

        with Progress(
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True
        ) as p:
            task = p.add_task(f"{model_name} [STRUCT FULL]", total=len(idx_list))

            for idx in idx_list:
                t = load_true(idx)
                y = loader(idx)

                # Spectra (k)
                k_centers, Ptt, Pyy, Pty = compute_pk_cross(t, y, nbins_k)
                T = Pty / (np.sqrt(Ptt * Pyy) + 1e-12)

                # Correlation (r)
                r_centers, xi_t, xi_y = compute_xi(t, y, nbins_r)
                xi_ratio = xi_y / (xi_t + 1e-12)

                # Collect
                T_list.append(T)
                xi_list.append(xi_ratio)
                if k_ref is None: k_ref = k_centers
                if r_ref is None: r_ref = r_centers

                p.advance(task)

        # Aggregate curves
        T_mean  = np.nanmedian(np.vstack(T_list), axis=0) if len(T_list) else np.full(nbins_k, np.nan)
        xi_mean = np.nanmedian(np.vstack(xi_list), axis=0) if len(xi_list) else np.full(nbins_r, np.nan)

        row = {
            "Model": model_name,
            "r_mean":   float(np.nanmean(T_mean)),
            "T_median": float(np.nanmedian(T_mean)),
            "T_MAE":    float(np.nanmean(np.abs(T_mean - 1.0))),
            "xi_rel_L2":float(np.nanmean((xi_mean - 1.0)**2)),
        }
        # optional: also report correlation between xi_pred and xi_true shapes across r
        # (not requested in table, but available)
        results.append(row)

        curves[model_name] = {"k": k_ref, "T": T_mean, "r": r_ref, "xi": xi_mean}

    df = pd.DataFrame(results, columns=["Model","r_mean","T_median","T_MAE","xi_rel_L2"])
    df.to_csv("structural_consistency_full.csv", index=False)
    np.savez("structural_consistency_curves_full.npz", **curves)
    log_kv("SAVED", {
        "structural_consistency_full.csv": "OK",
        "structural_consistency_curves_full.npz": "OK"
    })
    return df, curves

# =========================
# Main
# =========================
def main():
    models = ["Base Model", "V-NET (UNet3D)", "ViT (3D Transformer)"]

    #log("=== FULL EVALUATION START ===")
    #df1 = evaluate_prediction_accuracy_full(models)
    #log("\n=== Prediction Accuracy Summary (FULL) ===")
    #print(df1.to_string(index=False))

    #df2 = evaluate_distribution_bias_full(models)
    #log("\n=== Distribution & Bias Diagnostics (FULL) ===")
    #print(df2.to_string(index=False))

    # Tip: max_indices=None for truly full set; or set an upper bound if needed
    df3, curves = evaluate_structural_consistency_full(models, nbins_k=N_K_BINS, nbins_r=N_R_BINS, max_indices=None)
    log("\n=== Structural Consistency (FULL; aggregated curves) ===")
    print(df3.to_string(index=False))

    log("=== FULL EVALUATION FINISHED ===")

if __name__ == "__main__":
    main()
