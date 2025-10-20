#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing pipeline for:
“Enhancing Tool Wear State Identification in Imbalanced and Small Sample Scenarios
 through Conservative Adaptive Synthetic Sampling (CASS).”

Pipeline:
1) Wavelet denoising (per channel)
2) EEMD decomposition + feature extraction (per channel)
3) Feature scaling + KPCA dimensionality reduction
4) Visualizations for each stage

Input:
- CSV file where each column is a signal channel (time series).
- Default file path: "Original_3D.csv".
  If the file contains 9 columns named [x1,y1,z1, x2,y2,z2, x3,y3,z3],
  the script will optionally infer 3 class labels (one triplet per class)
  for coloring the KPCA scatter plot.

Outputs:
- Cleaned figures (.svg) and data (.csv) written to the working directory.

Requirements: numpy, pandas, matplotlib, scipy, PyEMD, scikit-learn, pywt.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from PyEMD import EEMD
import pywt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

# ------------------------------
# Basic matplotlib configuration
# ------------------------------
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.unicode_minus"] = False

# ------------------------------
# Utility helpers
# ------------------------------
def ensure_2d_array(arr: np.ndarray) -> np.ndarray:
    """Ensure shape (n_samples, n_channels)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr

def median_heuristic_gamma(X: np.ndarray, subsample: int = 5000) -> float:
    """
    Median heuristic for RBF gamma:
    gamma = 1 / (2 * median(||xi - xj||)^2)
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n > subsample:
        idx = np.random.choice(n, subsample, replace=False)
        X = X[idx]
    dists = np.sqrt(((X[None, :, :] - X[:, None, :]) ** 2).sum(-1))
    # Use upper triangle without zeros
    iu = np.triu_indices_from(dists, k=1)
    vals = dists[iu]
    med = np.median(vals[vals > 0]) if np.any(vals > 0) else 1.0
    gamma = 1.0 / (2.0 * (med ** 2))
    return float(gamma)

# ------------------------------
# Wavelet denoising
# ------------------------------
def wavelet_denoise_1d(x, wavelet="db6", level=None, mode="soft"):
    """
    Wavelet shrinkage denoising for a 1D signal.
    - level: if None, it is set based on signal length and wavelet filter length.
    - mode: 'soft' or 'hard' thresholding.
    """
    x = np.asarray(x, dtype=float)
    if level is None:
        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
        level = max(1, min(6, max_level))  # conservative cap
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # Universal threshold (VisuShrink)
    detail_coeffs = coeffs[1:]
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745 if len(detail_coeffs[-1]) > 0 else 0.0
    thr = sigma * np.sqrt(2.0 * np.log(len(x))) if sigma > 0 else 0.0
    coeffs_d = [coeffs[0]] + [pywt.threshold(c, thr, mode=mode) for c in detail_coeffs]
    x_rec = pywt.waverec(coeffs_d, wavelet)
    # Match original length (waverec can change by a sample)
    return x_rec[: len(x)]

def wavelet_denoise_matrix(X, wavelet="db6", level=None, mode="soft"):
    """Apply wavelet denoising column-wise to a 2D array."""
    X = ensure_2d_array(X)
    out = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = wavelet_denoise_1d(X[:, j], wavelet=wavelet, level=level, mode=mode)
    return out

# ------------------------------
# EEMD + feature extraction
# ------------------------------
def imf_features(imf: np.ndarray) -> dict:
    """Basic statistics for one IMF."""
    return {
        "mean": float(np.mean(imf)),
        "std": float(np.std(imf)),
        "skew": float(stats.skew(imf, bias=False)) if len(imf) > 2 else 0.0,
        "kurt": float(stats.kurtosis(imf, bias=False)) if len(imf) > 3 else 0.0,
        "energy": float(np.sum(imf ** 2)),
    }

def eemd_features_one_signal(x: np.ndarray, eemd: EEMD, max_imfs: int = None) -> pd.Series:
    """
    EEMD decomposition for a single 1D signal and feature extraction.
    Returns a Series with flattened features across IMFs.
    Also includes energy ratios per IMF.
    """
    IMFs = eemd(x)
    if IMFs.ndim == 1:
        IMFs = IMFs[None, :]
    n_imf = IMFs.shape[0] if max_imfs is None else min(max_imfs, IMFs.shape[0])

    feats = {}
    energies = []
    for i in range(n_imf):
        feats_i = imf_features(IMFs[i])
        energies.append(feats_i["energy"])
        # Store per-IMF stats
        for k, v in feats_i.items():
            feats[f"IMF{i+1}_{k}"] = v

    total_energy = np.sum(energies) if np.sum(energies) > 0 else 1.0
    for i, e in enumerate(energies):
        feats[f"IMF{i+1}_energy_ratio"] = float(e / total_energy)

    feats["n_imf_used"] = n_imf
    return pd.Series(feats), IMFs[:n_imf, :]

def eemd_features_matrix(X: np.ndarray, eemd_params: dict = None, max_imfs: int = None):
    """
    Apply EEMD per column and build a feature matrix (one row per channel).
    Returns (feature_df, list_of_IMF_arrays) so you can visualize IMFs later.
    """
    X = ensure_2d_array(X)
    if eemd_params is None:
        eemd_params = dict(trials=50, noise_width=0.2)
    eemd = EEMD(**eemd_params)

    rows = []
    imf_bank = []  # keep IMFs per channel for visualization
    for j in range(X.shape[1]):
        s = X[:, j]
        feats_j, IMFs_j = eemd_features_one_signal(s, eemd, max_imfs=max_imfs)
        feats_j.index = [f"ch{j+1}_{name}" for name in feats_j.index]
        rows.append(feats_j)
        imf_bank.append(IMFs_j)

    # Align indices across channels by concatenating then filling missing with 0
    feat_df = pd.DataFrame(rows).fillna(0.0)
    return feat_df, imf_bank

# ------------------------------
# KPCA projection
# ------------------------------
def run_kpca(features: pd.DataFrame, n_components=2, gamma=None, kernel="rbf", random_state=0):
    """Standardize features then apply KPCA."""
    X = features.to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if gamma is None and kernel == "rbf":
        gamma = median_heuristic_gamma(Xs)
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=random_state)
    Z = kpca.fit_transform(Xs)
    return Z, kpca, scaler, gamma

# ------------------------------
# Plotting helpers
# ------------------------------
def plot_raw_vs_denoised(X_raw, X_den, max_channels=3, outname="raw_vs_denoised.svg"):
    """Overlay raw and denoised for the first few channels."""
    X_raw = ensure_2d_array(X_raw)
    X_den = ensure_2d_array(X_den)
    n_show = min(max_channels, X_raw.shape[1])
    plt.figure(figsize=(9, 6))
    for i in range(n_show):
        plt.subplot(n_show, 1, i + 1)
        plt.plot(X_raw[:, i], linewidth=1, label=f"Raw ch{i+1}", alpha=0.8)
        plt.plot(X_den[:, i], linewidth=1, label=f"Denoised ch{i+1}", alpha=0.8)
        plt.legend(loc="upper right", fontsize=9)
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(outname, format="svg")
    plt.show()

def plot_imfs_and_heatmap(IMFs, x, outprefix="eemd"):
    """
    Plot IMF stack (line plots) and IMF heatmap for a single channel.
    IMFs: array [n_imf, n_samples]
    """
    n_imf = IMFs.shape[0]
    # IMF stack
    plt.figure(figsize=(9, 1.2 * (n_imf + 1)))
    plt.subplot(n_imf + 1, 1, 1)
    plt.plot(x, label="Denoised signal", linewidth=1)
    plt.legend(loc="upper right", fontsize=9)
    for i in range(n_imf):
        plt.subplot(n_imf + 1, 1, i + 2)
        plt.plot(IMFs[i], linewidth=1, label=f"IMF {i+1}")
        plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{outprefix}_stack.svg", format="svg")
    plt.show()

    # Heatmap
    plt.figure(figsize=(9, 3.5))
    plt.imshow(IMFs, aspect="auto")
    plt.colorbar()
    plt.xlabel("Sample index")
    plt.ylabel("IMF index")
    plt.title("IMF heatmap")
    plt.tight_layout()
    plt.savefig(f"{outprefix}_heatmap.svg", format="svg")
    plt.show()

def plot_imf_energy_bar(feat_row: pd.Series, outname="eemd_energy.svg"):
    """Bar chart of IMF energy ratios for one channel's features."""
    keys = [k for k in feat_row.index if k.endswith("energy_ratio")]
    vals = [feat_row[k] for k in keys]
    plt.figure(figsize=(7, 3))
    plt.bar(range(1, len(vals) + 1), vals)
    plt.xlabel("IMF index")
    plt.ylabel("Energy ratio")
    plt.title("IMF energy distribution")
    plt.tight_layout()
    plt.savefig(outname, format="svg")
    plt.show()

def plot_kpca_scatter(Z, labels=None, outname="kpca_scatter.svg"):
    """
    Plot KPCA embedding. If Z has 2 cols → 2D; if >=3 → 3D using first 3 components.
    Labels (optional) can color points by class.
    """
    Z = np.asarray(Z)
    if Z.shape[1] == 2:
        plt.figure(figsize=(6, 5))
        if labels is None:
            plt.scatter(Z[:, 0], Z[:, 1], s=35)
        else:
            for lab in np.unique(labels):
                idx = (labels == lab)
                plt.scatter(Z[idx, 0], Z[idx, 1], s=35, label=str(lab))
            plt.legend()
        plt.xlabel("KPCA-1")
        plt.ylabel("KPCA-2")
        plt.title("KPCA embedding (2D)")
        plt.tight_layout()
        plt.savefig(outname, format="svg")
        plt.show()
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        if labels is None:
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=35)
        else:
            for lab in np.unique(labels):
                idx = (labels == lab)
                ax.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2], s=35, label=str(lab))
            ax.legend()
        ax.set_xlabel("KPCA-1")
        ax.set_ylabel("KPCA-2")
        ax.set_zlabel("KPCA-3")
        ax.set_title("KPCA embedding (3D)")
        plt.tight_layout()
        plt.savefig(outname, format="svg")
        plt.show()

def plot_kpca_heatmap(Z, outname="kpca_heatmap.svg"):
    """Heatmap of KPCA scores."""
    plt.figure(figsize=(8, 3))
    plt.imshow(Z, aspect="auto")
    plt.colorbar()
    plt.xlabel("Component")
    plt.ylabel("Sample (channel)")
    plt.title("Heatmap of KPCA scores")
    plt.tight_layout()
    plt.savefig(outname, format="svg")
    plt.show()

# ------------------------------
# Optional label inference for Original_3D.csv
# ------------------------------
def infer_triplet_labels_if_possible(df: pd.DataFrame):
    """
    If the CSV looks like 9 columns in the order:
    [x1,y1,z1, x2,y2,z2, x3,y3,z3], assign labels 1/2/3 to those 3 groups
    (each column = a 'channel sample'), so we can color the KPCA scatter.
    Otherwise return None.
    """
    if df.shape[1] == 9:
        labels = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        return labels
    return None

# ------------------------------
# Main
# ------------------------------
def main(
    csv_path="Original_3D.csv",
    n_components=2,
    wavelet="db6",
    wavelet_level=None,
    eemd_trials=50,
    eemd_noise=0.2,
    eemd_max_imfs=None,
    kpca_gamma=None,
    random_state=0,
):
    # 1) Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    data_raw = df.values
    data_raw = np.nan_to_num(data_raw)
    data_raw = ensure_2d_array(data_raw)  # shape: (n_samples, n_channels)

    # Try to infer labels (for 9-column "Original_3D.csv")
    labels = infer_triplet_labels_if_possible(df)

    # 2) Wavelet denoising
    data_den = wavelet_denoise_matrix(
        data_raw, wavelet=wavelet, level=wavelet_level, mode="soft"
    )

    # 3) EEMD features (per channel)
    feat_df, imf_bank = eemd_features_matrix(
        data_den,
        eemd_params=dict(trials=eemd_trials, noise_width=eemd_noise),
        max_imfs=eemd_max_imfs,
    )

    # 4) KPCA on features
    Z, kpca, scaler, used_gamma = run_kpca(
        feat_df, n_components=n_components, gamma=kpca_gamma, kernel="rbf", random_state=random_state
    )

    # ------------------------------
    # Save intermediates
    # ------------------------------
    pd.DataFrame(data_raw).to_csv("stage_raw.csv", index=False, header=False)
    pd.DataFrame(data_den).to_csv("stage_denoised.csv", index=False, header=False)
    feat_df.to_csv("stage_eemd_features.csv", index=False)
    pd.DataFrame(Z).to_csv("stage_kpca_scores.csv", index=False)

    # ------------------------------
    # Visualizations
    # ------------------------------
    # (a) Raw vs Denoised (first 3 channels)
    plot_raw_vs_denoised(data_raw, data_den, max_channels=3, outname="raw_vs_denoised.svg")

    # (b) IMF stack + heatmap for channel 1 (if exists)
    if len(imf_bank) > 0 and imf_bank[0].size > 0:
        plot_imfs_and_heatmap(imf_bank[0], data_den[:, 0], outprefix="eemd_ch1")
        # (c) IMF energy distribution for channel 1
        # Find that row in feat_df
        row = feat_df.iloc[0]
        # Select only the IMF energy ratios of ch1
        row_ch1 = row[[c for c in feat_df.columns if c.startswith("ch1_")]]
        plot_imf_energy_bar(row_ch1, outname="eemd_ch1_energy.svg")

    # (d) KPCA scatter
    plot_kpca_scatter(Z, labels=labels, outname="kpca_scatter.svg")

    # (e) KPCA heatmap
    plot_kpca_heatmap(Z, outname="kpca_heatmap.svg")

    # Console summary
    print("=== Pipeline summary ===")
    print(f"Input file: {csv_path}")
    print(f"Wavelet: {wavelet}, level: {wavelet_level}")
    print(f"EEMD: trials={eemd_trials}, noise_width={eemd_noise}, max_imfs={eemd_max_imfs}")
    print(f"KPCA: n_components={n_components}, RBF gamma={used_gamma:.4g}")
    print("Saved: stage_raw.csv, stage_denoised.csv, stage_eemd_features.csv, stage_kpca_scores.csv")
    print("Saved figures: raw_vs_denoised.svg, eemd_ch1_stack.svg, eemd_ch1_heatmap.svg, "
          "eemd_ch1_energy.svg, kpca_scatter.svg, kpca_heatmap.svg")

if __name__ == "__main__":
    """
    Notes & tips

Labels for KPCA plot: If your Original_3D.csv truly encodes three classes as [x1,y1,z1, x2,y2,z2, x3,y3,z3], the script colors points by class automatically. If not, labels are omitted.

Feature design: The EEMD feature set is intentionally compact (mean/std/skew/kurtosis + energy ratios). You can extend with, e.g., entropy, zero-crossing rate, spectral centroid per IMF.

Gamma selection: Default uses a median heuristic over standardized features. If you’ve found a good fixed value (e.g., gamma=5 in your drafts), pass it via kpca_gamma=5.0.

Speed vs. quality: EEMD can be slow. Reduce trials, lower noise_width, or set eemd_max_imfs to cap the number of IMFs.
    """


    # You can adjust parameters here if needed.
    main(
        csv_path="your file",   # <- Your raw data must be in .csv format.
        n_components=2,               # set to 3 for 3D scatter
        wavelet="db6",
        wavelet_level=None,           # None = auto
        eemd_trials=50,
        eemd_noise=0.2,
        eemd_max_imfs=None,           # None = use all IMFs returned
        kpca_gamma=None,              # None = median heuristic; or set a value (e.g., 5.0)
        random_state=0,
    )
