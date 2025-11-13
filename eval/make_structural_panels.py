#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_structural_panels.py  (Python 3.8/3.9 compatible)

Computes and saves:
  (L) Joint PDF (truth vs prediction) with 1σ/2σ/3σ highest-density contours
  (C) Density PDF (ρ/ρ0): median line + 1σ band (16–84%)
  (R) Two-point correlation ξ(r): median line + 1σ band

Outputs:
  - CSV: bins_pdf.csv, r_bins.csv
  - CSV: truth_*_percentiles.csv, {base,unet,vit}_*_percentiles.csv
  - NPZ: joint2d_histograms.npz (2D histograms + contour thresholds)
  - PNG: panels_{base,unet,vit}.png

Run (example, matches your SLURM script):
  python make_structural_panels.py \
    --base_dir "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions" \
    --unet_dir "/home/mingyeong/GAL2DM_ASIM_VNET/results/unet_predictions/28845/icase-both-keep2" \
    --vit_dir  "/home/mingyeong/GAL2DM_ASIM_ViT/results/vit_predictions/28846/icase-both" \
    --truth_tpl "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/{idx}.hdf5" \
    --outdir "./struct_outputs"
"""

import os, re, glob, itertools, time, argparse
from typing import Optional, List, Tuple, Dict
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

# =========================
# Default Paths (override via CLI)
# =========================
DEFAULT_BASE_PRED_DIR = "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/predictions"
DEFAULT_UNET_PRED_DIR = "/home/mingyeong/GAL2DM_ASIM_VNET/results/unet_predictions/28845/icase-both-keep2"
DEFAULT_VIT_PRED_DIR  = "/home/mingyeong/GAL2DM_ASIM_ViT/results/vit_predictions/28846/icase-both"
DEFAULT_TRUE_PATH_TPL = "/scratch/adupuy/cosmicweb_asim/ASIM_TSC/samples/test/{idx}.hdf5"

# =========================
# Global Config
# =========================
BOX_SIZE = 160.0              # [h^-1 Mpc], assume periodic (N=128 → Δ=1.25, r_max~80)
N_R_BINS = 25
REF_TOL = 2                   # center crop tolerance (±voxels)
RANDOM_SEED = 42
LOG_EVERY = 5

BASE_IS_LOG10_OVER_2P5 = True
MAX_VOX_SAMPLES = 800_000     # total sampled voxels per index for joint/PDF
JOINT_BINS = np.geomspace(1e-5, 1e4, 256)   # for 2D joint histogram (truth/pred in ρ/ρ0)
PDF_BINS  = np.geomspace(1e-5, 1e4, 200)    # for 1D density PDF(ρ/ρ0)

USE_STRICT_LOG_FOR_XI = True  # if band ≤0 dominates, code auto-switches to symlog

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
    items = " | ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{_hhmmss(dt)}] {title} :: {items}", flush=True)

# =========================
# Shape / IO Utilities (3.8/3.9-safe typing)
# =========================
def to_DHW(a: np.ndarray) -> np.ndarray:
    """(N,C,D,H,W)/(C,D,H,W)/(N,D,H,W)/(H,W,D)/(D,H,W) -> (D,H,W)."""
    a = np.asarray(a)
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    if   a.ndim == 5: a = a[0, 0]
    elif a.ndim == 4: a = a[0]
    elif a.ndim != 3: raise ValueError(f"Unsupported shape: {a.shape}")
    if a.shape[2] == max(a.shape) and a.shape[0] != max(a.shape):
        a = a.transpose(2, 0, 1)  # (H,W,D)->(D,H,W)
    return a

def squeeze_to_3d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    while a.ndim > 3 and a.shape[0] == 1: a = a[0]
    if a.ndim == 5: a = a[0, 0]
    elif a.ndim == 4: a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expect 3D, got {a.shape}")
    return a

def permute_to_shape(a: np.ndarray, ref_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if a.shape == ref_shape:
        return a
    if sorted(a.shape) == sorted(ref_shape):
        for perm in itertools.permutations(range(3)):
            if tuple(np.array(a.shape)[list(perm)]) == tuple(ref_shape):
                return np.transpose(a, axes=perm)
    return None

def center_crop_to(a: np.ndarray, ref_shape: Tuple[int, int, int], tol: int = REF_TOL) -> Optional[np.ndarray]:
    diffs = np.array(a.shape) - np.array(ref_shape)
    if np.all(np.abs(diffs) <= tol):
        zs, ys, xs = a.shape
        rz, ry, rx = ref_shape
        z0 = (zs - rz)//2; y0 = (ys - ry)//2; x0 = (xs - rx)//2
        return a[z0:z0+rz, y0:y0+ry, x0:x0+rx]
    return None

def harmonize_to_ref(a: np.ndarray, ref_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    try:
        a = squeeze_to_3d(a)
    except Exception:
        return None
    if a.shape == ref_shape:
        return a
    b = permute_to_shape(a, ref_shape)
    if b is not None:
        return b
    c = center_crop_to(a, ref_shape, REF_TOL)
    if c is not None:
        return c
    return None

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _first_int(text: str, default=None):
    m = re.findall(r"\d+", text)
    return int(m[0]) if m else default

def list_true_indices(tpl: str) -> List[str]:
    tdir = os.path.dirname(tpl.format(idx="0"))
    files = glob.glob(os.path.join(tdir, "*.hdf5"))
    return [_stem(p) for p in files]

def list_unet_indices(dir_: str) -> List[str]:
    return [_stem(p) for p in glob.glob(os.path.join(dir_, "*.hdf5"))]

def list_vit_indices(dir_: str) -> List[str]:
    return [_stem(p) for p in glob.glob(os.path.join(dir_, "*.hdf5"))]

def list_base_indices(dir_: str) -> List[str]:
    files = glob.glob(os.path.join(dir_, "test_*.npy"))
    idxs: List[str] = []
    for p in files:
        m = re.search(r"test_(\d+)_", os.path.basename(p))
        if m: idxs.append(m.group(1))
    return idxs

def common_indices(base_dir: str, unet_dir: str, vit_dir: str, truth_tpl: str) -> List[str]:
    T = set(list_true_indices(truth_tpl))
    U = set(list_unet_indices(unet_dir))
    V = set(list_vit_indices(vit_dir))
    B = set(list_base_indices(base_dir))
    return sorted(T & U & V & B, key=_first_int)

# ---------------- Loaders (to_DHW + clip) ----------------
def load_true(tpl: str, idx: str) -> np.ndarray:
    with h5py.File(tpl.format(idx=idx), "r") as f:
        a = to_DHW(f["output_rho"][:])
    return np.clip(a, 0, None)

def load_unet(dir_: str, idx: str) -> np.ndarray:
    with h5py.File(os.path.join(dir_, f"{idx}.hdf5"), "r") as f:
        a = to_DHW(f["prediction"][:])
    return np.clip(a, 0, None)

def load_vit(dir_: str, idx: str) -> np.ndarray:
    with h5py.File(os.path.join(dir_, f"{idx}.hdf5"), "r") as f:
        a = to_DHW(f["prediction"][:])
    return np.clip(a, 0, None)

def load_base(dir_: str, idx: str) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(dir_, f"test_{idx}_*.npy")))
    if not files:
        raise FileNotFoundError(f"Base not found for idx={idx}")
    arr = np.load(files[0])
    rho = 10**(2.5*arr) if BASE_IS_LOG10_OVER_2P5 else arr
    return np.clip(to_DHW(rho), 0, None)

# =========================
# Stats helpers
# =========================
def percentile_band(samples: np.ndarray, q=(16, 50, 84), axis=0):
    if samples.size == 0:
        return None, None, None
    return np.percentile(samples, q, axis=axis)

def xi_from_delta_fft(delta: np.ndarray, box_size: float, nbins: int):
    """
    Isotropic ξ(r) via FFT autocorrelation (periodic BC).
    Steps:
      δ -> FFT -> |F|^2 -> IFFT -> radial bin average.
    """
    delta = delta.astype(np.float64, copy=False)
    delta = delta - delta.mean()
    fk = np.fft.fftn(delta)
    corr = np.fft.ifftn(np.abs(fk)**2).real / delta.size

    nz, ny, nx = delta.shape
    dx = box_size / nx
    dy = box_size / ny
    dz = box_size / nz

    # distance grid via FFT frequencies (min-image)
    zz = np.fft.fftfreq(nz) * nz
    yy = np.fft.fftfreq(ny) * ny
    xx = np.fft.fftfreq(nx) * nx
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing='ij')
    R = np.sqrt((X*dx)**2 + (Y*dy)**2 + (Z*dz)**2)

    r_edges = np.geomspace(dx, box_size/2, nbins+1)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])
    bin_idx = np.searchsorted(r_edges, R.ravel(), side='right') - 1
    valid = (bin_idx >= 0) & (bin_idx < nbins)
    sums = np.bincount(bin_idx[valid], weights=corr.ravel()[valid], minlength=nbins)
    counts = np.bincount(bin_idx[valid], minlength=nbins)
    xi_r = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return r_centers, xi_r

def credible_contour_levels(H: np.ndarray, levels=(0.6827, 0.9545, 0.9973)):
    """
    Highest-density credible contour levels for a 2D histogram H.
    Returns density thresholds whose superlevel sets integrate to given probs.
    """
    pdf = H / max(H.sum(), 1.0)
    flat = np.sort(pdf.ravel())[::-1]
    cdf = np.cumsum(flat)
    thr: List[float] = []
    for p in levels:
        k = np.searchsorted(cdf, p)
        thr.append(flat[k] if k < len(flat) else flat[-1])
    return thr

# =========================
# Main compute
# =========================
def main(args):
    rng = np.random.default_rng(args.seed)

    # discover indices
    idxs = common_indices(args.base_dir, args.unet_dir, args.vit_dir, args.truth_tpl)
    if args.max_indices is not None:
        idxs = idxs[:args.max_indices]
    log_kv("INDICES", {"count": len(idxs), "examples": idxs[:10]})

    os.makedirs(args.outdir, exist_ok=True)

    # accumulators
    H_joint: Dict[str, np.ndarray] = {
        "Base": np.zeros((len(JOINT_BINS)-1, len(JOINT_BINS)-1), dtype=np.float64),
        "UNet": np.zeros((len(JOINT_BINS)-1, len(JOINT_BINS)-1), dtype=np.float64),
        "ViT":  np.zeros((len(JOINT_BINS)-1, len(JOINT_BINS)-1), dtype=np.float64),
    }

    pdf_truth_list: List[np.ndarray] = []
    pdf_pred_list: Dict[str, List[np.ndarray]] = {"Base": [], "UNet": [], "ViT": []}

    xi_truth_list: List[np.ndarray] = []
    xi_pred_list: Dict[str, List[np.ndarray]] = {"Base": [], "UNet": [], "ViT": []}
    r_ref: Optional[np.ndarray] = None

    with Progress(
        TextColumn("[bold cyan]Processing[/]"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.description}"),
    ) as prog:
        task = prog.add_task("indices", total=len(idxs))
        for t, idx in enumerate(idxs):
            # Load
            rho_t = load_true(args.truth_tpl, idx)
            rho_b = load_base(args.base_dir, idx)
            rho_u = load_unet(args.unet_dir, idx)
            rho_v = load_vit(args.vit_dir,  idx)

            # Harmonize
            ref_shape = rho_t.shape
            for name, arr in (("Base", rho_b), ("UNet", rho_u), ("ViT", rho_v)):
                h = harmonize_to_ref(arr, ref_shape)
                if h is None:
                    raise RuntimeError(f"Shape mismatch for {name} at idx={idx} "
                                       f"{arr.shape} vs {ref_shape}")
                if name == "Base": rho_b = h
                elif name == "UNet": rho_u = h
                else: rho_v = h

            # Means
            mu_t = rho_t.mean()
            mu_b = rho_b.mean(); mu_u = rho_u.mean(); mu_v = rho_v.mean()

            # -------- Joint 2D histogram (sampled voxels)
            Nvox = rho_t.size
            take = min(Nvox, max(1, MAX_VOX_SAMPLES // max(1, len(idxs))))
            flat_idx = rng.choice(Nvox, size=take, replace=False)

            x = (rho_t.ravel()[flat_idx] / mu_t).clip(JOINT_BINS[0], JOINT_BINS[-1])
            yB = (rho_b.ravel()[flat_idx] / mu_b).clip(JOINT_BINS[0], JOINT_BINS[-1])
            yU = (rho_u.ravel()[flat_idx] / mu_u).clip(JOINT_BINS[0], JOINT_BINS[-1])
            yV = (rho_v.ravel()[flat_idx] / mu_v).clip(JOINT_BINS[0], JOINT_BINS[-1])

            HB, _, _ = np.histogram2d(x, yB, bins=(JOINT_BINS, JOINT_BINS))
            HU, _, _ = np.histogram2d(x, yU, bins=(JOINT_BINS, JOINT_BINS))
            HV, _, _ = np.histogram2d(x, yV, bins=(JOINT_BINS, JOINT_BINS))
            H_joint["Base"] += HB; H_joint["UNet"] += HU; H_joint["ViT"] += HV

            # -------- Density PDF per-sample
            def one_pdf(arr: np.ndarray, mu: float) -> np.ndarray:
                h, _ = np.histogram((arr/mu).ravel(), bins=PDF_BINS)
                pdf = h / max(h.sum(), 1.0)
                return pdf

            pdf_truth_list.append(one_pdf(rho_t, mu_t))
            pdf_pred_list["Base"].append(one_pdf(rho_b, mu_b))
            pdf_pred_list["UNet"].append(one_pdf(rho_u, mu_u))
            pdf_pred_list["ViT"].append( one_pdf(rho_v, mu_v))

            # -------- 2PCF per-sample (δ = ρ/⟨ρ⟩ - 1)
            r_this, xi_t = xi_from_delta_fft(rho_t/mu_t - 1.0, BOX_SIZE, N_R_BINS)
            _,      xi_b = xi_from_delta_fft(rho_b/mu_b - 1.0, BOX_SIZE, N_R_BINS)
            _,      xi_u = xi_from_delta_fft(rho_u/mu_u - 1.0, BOX_SIZE, N_R_BINS)
            _,      xi_v = xi_from_delta_fft(rho_v/mu_v - 1.0, BOX_SIZE, N_R_BINS)
            if r_ref is None: r_ref = r_this

            xi_truth_list.append(xi_t)
            xi_pred_list["Base"].append(xi_b)
            xi_pred_list["UNet"].append(xi_u)
            xi_pred_list["ViT"].append( xi_v)

            prog.advance(task)
            if (t % LOG_EVERY) == 0:
                log_kv("progress", {"done": t+1, "total": len(idxs)})

    # ======================
    # Aggregate percentiles
    # ======================
    pdf_truth_arr = np.asarray(pdf_truth_list)          # (N, B)
    xi_truth_arr  = np.asarray(xi_truth_list)           # (N, R)
    p16_pdf_t, p50_pdf_t, p84_pdf_t = percentile_band(pdf_truth_arr, axis=0)
    p16_xi_t,  p50_xi_t,  p84_xi_t  = percentile_band(xi_truth_arr,  axis=0)

    pdf_pred_arr: Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in pdf_pred_list.items()}
    xi_pred_arr:  Dict[str, np.ndarray] = {k: np.asarray(v) for k, v in xi_pred_list.items()}

    pct_pdf: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    pct_xi:  Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for k in ("Base", "UNet", "ViT"):
        pct_pdf[k] = percentile_band(pdf_pred_arr[k], axis=0)
        pct_xi[k]  = percentile_band(xi_pred_arr[k],  axis=0)

    thr: Dict[str, List[float]] = {k: credible_contour_levels(H_joint[k]) for k in ("Base", "UNet", "ViT")}

    # ======================
    # Save CSV / NPZ
    # ======================
    out = args.outdir
    os.makedirs(out, exist_ok=True)

    np.savetxt(os.path.join(out, "bins_pdf.csv"),
               np.vstack([PDF_BINS[:-1], PDF_BINS[1:]]).T,
               delimiter=",", header="bin_low,bin_high", comments='')
    if r_ref is None:
        raise RuntimeError("r_ref is None (no indices?)")
    np.savetxt(os.path.join(out, "r_bins.csv"), r_ref, delimiter=",", header="r[h^-1 Mpc]", comments='')

    np.savetxt(os.path.join(out, "truth_pdf_percentiles.csv"),
               np.vstack([p16_pdf_t, p50_pdf_t, p84_pdf_t]).T,
               delimiter=",", header="p16,p50,p84", comments='')
    np.savetxt(os.path.join(out, "truth_xi_percentiles.csv"),
               np.vstack([p16_xi_t, p50_xi_t, p84_xi_t]).T,
               delimiter=",", header="p16,p50,p84", comments='')

    for k in ("Base","UNet","ViT"):
        p16, p50, p84 = pct_pdf[k]
        np.savetxt(os.path.join(out, f"{k.lower()}_pdf_percentiles.csv"),
                   np.vstack([p16, p50, p84]).T, delimiter=",", header="p16,p50,p84", comments='')
        p16, p50, p84 = pct_xi[k]
        np.savetxt(os.path.join(out, f"{k.lower()}_xi_percentiles.csv"),
                   np.vstack([p16, p50, p84]).T, delimiter=",", header="p16,p50,p84", comments='')

    np.savez(os.path.join(out, "joint2d_histograms.npz"),
             JOINT_BINS=JOINT_BINS,
             H_Base=H_joint["Base"], H_UNet=H_joint["UNet"], H_ViT=H_joint["ViT"],
             thr_Base=np.array(thr["Base"]), thr_UNet=np.array(thr["UNet"]), thr_ViT=np.array(thr["ViT"]))
    log("Saved CSV/NPZ files.")

    # ======================
    # Plot per model (3 panels)
    # ======================
    def panel_figure(model_name: str, color: str, H2d: np.ndarray, thr_levels: List[float]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

        # --- (L) Joint PDF with credible contours
        X, Y = np.meshgrid(JOINT_BINS, JOINT_BINS, indexing='xy')
        pcm = axs[0].pcolormesh(X, Y, H2d.T + 1e-12, norm=LogNorm(), cmap='viridis')
        axs[0].plot(JOINT_BINS, JOINT_BINS, ls="--", color="k", alpha=0.6)
        pdf2d = H2d / max(H2d.sum(), 1.0)
        axs[0].contour(0.5*(JOINT_BINS[:-1]+JOINT_BINS[1:]),
                       0.5*(JOINT_BINS[:-1]+JOINT_BINS[1:]),
                       pdf2d.T, levels=thr_levels,
                       colors=('orange','red','magenta'), linewidths=(1.6,1.4,1.2))
        axs[0].set_xscale("log"); axs[0].set_yscale("log")
        axs[0].set_xlabel(r"$\rho_{\mathrm{truth}}/\rho_0$")
        axs[0].set_ylabel(r"$\rho_{\mathrm{pred}}/\rho_0$")
        axs[0].set_title(f"Joint PDF ({model_name})")
        fig.colorbar(pcm, ax=axs[0], fraction=0.046, pad=0.04)

        # --- (C) Density PDF: truth (gray), pred (color)
        xx = 0.5*(PDF_BINS[:-1]+PDF_BINS[1:])
        p16, p50, p84 = p16_pdf_t, p50_pdf_t, p84_pdf_t
        m = (p50 > 0) & np.isfinite(p50)
        axs[1].fill_between(xx[m], p16[m], p84[m], color="0.5", alpha=0.25)
        axs[1].plot(xx[m], p50[m], color="k", lw=2, label="Truth")

        p16, p50, p84 = pct_pdf[model_name]
        m = (p50 > 0) & np.isfinite(p50)
        axs[1].fill_between(xx[m], p16[m], p84[m], color=color, alpha=0.25)
        axs[1].plot(xx[m], p50[m], color=color, lw=2, label="Prediction")
        axs[1].set_xscale("log"); axs[1].set_yscale("log")
        axs[1].set_xlabel(r"$\rho/\rho_0$")
        axs[1].set_ylabel(r"$\mathrm{d}f/\mathrm{d}\log_{10}\rho$")  # proportional
        axs[1].set_title("Density PDF (median + 1σ)")
        axs[1].legend()

        # --- (R) ξ(r): truth (gray), pred (color)
        def set_xi_scale(ax, p16_arr, p84_arr, p50_arr) -> np.ndarray:
            if USE_STRICT_LOG_FOR_XI:
                pos = (p16_arr > 0) & (p84_arr > 0) & (p50_arr > 0)
                frac = float(pos.sum()) / float(len(pos))
                ax.set_yscale("log" if frac >= 0.3 else "symlog")
                if frac < 0.3:
                    ax.set_yscale("symlog", linthresh=1e-5, linscale=1.0)
                    return np.isfinite(p50_arr)
                return pos
            else:
                ax.set_yscale("symlog", linthresh=1e-5, linscale=1.0)
                return np.isfinite(p50_arr)

        axs[2].set_xscale("log")
        # truth
        m = set_xi_scale(axs[2], p16_xi_t, p84_xi_t, p50_xi_t)
        axs[2].fill_between(r_ref[m], p16_xi_t[m], p84_xi_t[m], color="0.5", alpha=0.25)
        axs[2].plot(r_ref[m], p50_xi_t[m], color="k", lw=2, label="Truth")
        # model
        p16, p50, p84 = pct_xi[model_name]
        m = (p16 > 0) & (p84 > 0) & (p50 > 0) if axs[2].get_yscale() == "log" else np.isfinite(p50)
        axs[2].fill_between(r_ref[m], p16[m], p84[m], color=color, alpha=0.25)
        axs[2].plot(r_ref[m], p50[m], color=color, lw=2, label="Prediction")

        axs[2].set_xlabel(r"$r\ [h^{-1}\mathrm{Mpc}]$")
        axs[2].set_ylabel(r"$\langle \delta(\mathbf{x})\,\delta(\mathbf{x}+\mathbf{r}) \rangle$")
        axs[2].set_title(r"Two-Point Correlation (median + $1\sigma$)")
        axs[2].legend()

        fig.tight_layout()
        return fig

    colors = {"Base": "#1f77b4", "UNet": "#ff7f0e", "ViT": "#2ca02c"}
    for name in ("Base", "UNet", "ViT"):
        fig = panel_figure(name, colors[name], H_joint[name], thr[name])
        fig.savefig(os.path.join(out, f"panels_{name.lower()}.png"), dpi=220)
        plt.close(fig)

    log("All figures saved.")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir",  default=DEFAULT_BASE_PRED_DIR)
    ap.add_argument("--unet_dir",  default=DEFAULT_UNET_PRED_DIR)
    ap.add_argument("--vit_dir",   default=DEFAULT_VIT_PRED_DIR)
    ap.add_argument("--truth_tpl", default=DEFAULT_TRUE_PATH_TPL)
    ap.add_argument("--outdir",    default="./struct_outputs")
    ap.add_argument("--max_indices", type=int, default=None, help="limit for quick tests")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    log_kv("CONFIG", {
        "BOX_SIZE[h^-1Mpc]": BOX_SIZE,
        "N_R_BINS": N_R_BINS,
        "MAX_VOX_SAMPLES": MAX_VOX_SAMPLES,
        "JOINT_BINS": len(JOINT_BINS)-1,
        "PDF_BINS": len(PDF_BINS)-1,
        "STRICT_LOG_XI": USE_STRICT_LOG_FOR_XI,
    })
    log_kv("PATHS", {
        "base": args.base_dir,
        "unet": args.unet_dir,
        "vit": args.vit_dir,
        "truth_tpl": args.truth_tpl,
        "outdir": args.outdir,
    })
    main(args)
