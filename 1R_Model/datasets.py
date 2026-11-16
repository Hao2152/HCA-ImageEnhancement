# datasets.py
import logging
import numpy as np
from scipy.ndimage import convolve1d
import torch
from torch.utils import data

from utils import get_lds_kernel_window

print = logging.info

TABULAR_COLS = [
    'original_musiq', 'global_laplacian_var', 'global_edge_density',
    'global_tv_density', 'global_noise_estimation', 'global_gradient_mean',
    'global_gradient_std', 'global_contrast', 'global_skewness', 'global_kurtosis',
    'global_hsv_s_mean', 'global_hsv_v_mean', 'global_lbp_var', 'global_glcm_contrast',
    'global_glcm_homogeneity', 'global_high_freq_energy_ratio',
    'global_usm_0_5_energy', 'global_usm_1_1_energy', 'total_pixels'
]

LOG_COLS = [
    'global_laplacian_var', 'global_edge_density', 'global_tv_density',
    'global_noise_estimation', 'global_gradient_mean', 'global_gradient_std',
    'global_glcm_contrast', 'global_high_freq_energy_ratio',
    'global_usm_0_5_energy', 'global_usm_1_1_energy', 'total_pixels'
]


def extract_tabular_matrix(df):
    feat_df = df[TABULAR_COLS].copy().fillna(0.0)
    for c in LOG_COLS:
        feat_df[f'log_{c}'] = np.log1p(np.maximum(0.0, feat_df[c].values))
    return feat_df.values.astype(np.float32)


class IMDBWIKI(data.Dataset):
    def __init__(self, df, tab_features, tol=0.3, split='train',
                 reweight='none', lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=1, **kwargs):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.tol = float(tol)
        self.tab_features = tab_features.astype(np.float32)

        # 預先組裝所有標籤，免除訓練迴圈中的 DataFrame iloc 開銷
        orig_min = self.df['acceptable_alpha_min'].values.astype(np.float32)
        orig_max = self.df['acceptable_alpha_max'].values.astype(np.float32)
        if 'acceptable_alpha_center' in self.df.columns:
            y_center = self.df['acceptable_alpha_center'].values.astype(np.float32)
        else:
            y_center = 0.5 * (orig_min + orig_max)
        tol_min = np.clip(orig_min - self.tol, 0.1, 2.0).astype(np.float32)
        tol_max = np.clip(orig_max + self.tol, 0.1, 2.0).astype(np.float32)

        self.labels = np.column_stack([y_center, orig_min, orig_max, tol_min, tol_max]).astype(np.float32)

        if self.split == 'train':
            self.weights = self._prepare_weights(
                reweight=reweight, max_target=20, lds=lds,
                lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma
            )
        else:
            self.weights = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tab_feat = torch.from_numpy(self.tab_features[index])
        label = torch.from_numpy(self.labels[index])
        w = np.float32(self.weights[index]) if self.weights is not None else np.float32(1.0)
        weight = torch.tensor([w], dtype=torch.float32)
        return tab_feat, label, weight

    @staticmethod
    def _alpha_to_idx(alpha_val, max_target=20):
        idx = int(np.round(float(alpha_val) * 10)) - 1
        return min(max_target - 1, max(0, idx))

    def _prepare_weights(self, reweight='none', max_target=20, lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=1):
        if lds and reweight == 'none':
            reweight = 'sqrt_inv'
        if reweight == 'none':
            print("Using re-weighting: [NONE]")
            return None

        value_dict = {x: 0 for x in range(max_target)}
        centers = self.labels[:, 0]
        for c in centers:
            value_dict[self._alpha_to_idx(c, max_target)] += 1

        if reweight == 'sqrt_inv':
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == 'inverse':
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

        num_per_label = [value_dict[self._alpha_to_idx(c, max_target)] for c in centers]
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
            num_per_label = [smoothed_value[self._alpha_to_idx(c, max_target)] for c in centers]

        weights = [np.float32(1.0 / x) if x > 0 else np.float32(0.0) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        return [scaling * x for x in weights]