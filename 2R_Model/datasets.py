# datasets.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import convolve1d
from utils import get_lds_kernel_window

# 18 個基礎特徵名稱
BASE_FEATURE_NAMES = [
    "laplacian_var", "edge_density", "tv_density", "noise_estimation",
    "gradient_mean", "gradient_std", "contrast", "skewness", "kurtosis",
    "hsv_s_mean", "hsv_v_mean", "lbp_var", "glcm_contrast",
    "glcm_homogeneity", "high_freq_energy_ratio", "usm_0_5_energy",
    "usm_1_1_energy", "total_pixels",
]

# 完整 54 維欄位
TABULAR_COLS = []
for prefix in ["global", "in", "out"]:
    for feat in BASE_FEATURE_NAMES:
        if feat == "total_pixels" and prefix == "global":
            TABULAR_COLS.append("total_pixels")
        else:
            TABULAR_COLS.append(f"{prefix}_{feat}")

# 專屬特徵欄位定義 (各 36 維: Global 18 + Region 18)
IN_COLS = ["total_pixels"] + [f"global_{f}" for f in BASE_FEATURE_NAMES if f != "total_pixels"] + [f"in_{f}" for f in
                                                                                                   BASE_FEATURE_NAMES]
OUT_COLS = ["total_pixels"] + [f"global_{f}" for f in BASE_FEATURE_NAMES if f != "total_pixels"] + [f"out_{f}" for f in
                                                                                                    BASE_FEATURE_NAMES]


def extract_tabular_matrices(df):
    """同時萃取 In 專屬 (36D) 與 Out 專屬 (36D) 的特徵矩陣"""
    missing_in = [c for c in IN_COLS if c not in df.columns]
    missing_out = [c for c in OUT_COLS if c not in df.columns]
    if missing_in or missing_out:
        raise ValueError(f"DataFrame 缺少 2-Region 特徵欄位! In缺: {missing_in}, Out缺: {missing_out}")

    mat_in = df[IN_COLS].values.astype(np.float32)
    mat_out = df[OUT_COLS].values.astype(np.float32)
    return mat_in, mat_out


def compute_sample_weights(labels, min_val, num_bins, reweight='none', lds=False, lds_kernel='gaussian', lds_ks=5,
                           lds_sigma=1.0):
    if reweight == 'none' and not lds:
        return np.ones(len(labels), dtype=np.float32)
    eff_reweight = 'sqrt_inv' if (lds and reweight == 'none') else reweight
    bins = np.clip(np.round((labels - min_val) * 10.0).astype(int), 0, num_bins - 1)
    value_counts = np.bincount(bins, minlength=num_bins).astype(np.float32)
    if eff_reweight == 'sqrt_inv':
        value_counts = np.sqrt(value_counts)
    elif eff_reweight == 'inverse':
        value_counts = np.clip(value_counts, 1.0, None)
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed_counts = convolve1d(value_counts, weights=lds_kernel_window, mode='constant')
    else:
        smoothed_counts = value_counts
    num_per_label = np.clip(smoothed_counts[bins], 1e-6, None)
    weights = 1.0 / num_per_label
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


class IMDBWIKI(Dataset):
    def __init__(self, df, tab_features_in, tab_features_out, tol=0.3, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=1.0, is_2region=True):
        self.df = df
        self.tab_features_in = tab_features_in
        self.tab_features_out = tab_features_out
        self.tol = tol
        self.split = split
        self.is_2region = is_2region

        self.y_center_in = df['in_alpha_center'].values.astype(np.float32)
        self.orig_min_in = df['best_alpha_in'].values.astype(np.float32)
        self.orig_max_in = df['best_alpha_in'].values.astype(np.float32)
        self.tol_min_in = df['in_alpha_min'].values.astype(np.float32)
        self.tol_max_in = df['in_alpha_max'].values.astype(np.float32)

        self.y_center_out = df['out_alpha_center'].values.astype(np.float32)
        self.orig_min_out = df['best_alpha_out'].values.astype(np.float32)
        self.orig_max_out = df['best_alpha_out'].values.astype(np.float32)
        self.tol_min_out = df['out_alpha_min'].values.astype(np.float32)
        self.tol_max_out = df['out_alpha_max'].values.astype(np.float32)

        if split == 'train' and (lds or reweight != 'none'):
            w_in = compute_sample_weights(self.orig_min_in, 1.0, 11, reweight, lds, lds_kernel, lds_ks, lds_sigma)
            w_out = compute_sample_weights(self.orig_min_out, 0.1, 10, reweight, lds, lds_kernel, lds_ks, lds_sigma)
            combined_w = np.sqrt(w_in * w_out)
            self.weights = (combined_w / combined_w.sum() * len(combined_w)).astype(np.float32)
        else:
            self.weights = np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 💡 分別回傳 Foreground 專屬特徵 (36D) 與 Background 專屬特徵 (36D)
        tab_feat_in = torch.tensor(self.tab_features_in[idx], dtype=torch.float32)
        tab_feat_out = torch.tensor(self.tab_features_out[idx], dtype=torch.float32)

        target_in = torch.tensor([
            self.y_center_in[idx], self.orig_min_in[idx], self.orig_max_in[idx],
            self.tol_min_in[idx], self.tol_max_in[idx]
        ], dtype=torch.float32)

        target_out = torch.tensor([
            self.y_center_out[idx], self.orig_min_out[idx], self.orig_max_out[idx],
            self.tol_min_out[idx], self.tol_max_out[idx]
        ], dtype=torch.float32)

        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return tab_feat_in, tab_feat_out, target_in, target_out, weight