#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批次提取 18 個全域影像統計特徵並直接附加 (Append) 至輸入 CSV：
- 嚴格對齊訓練集 (`Inference.py` / `datasets.py`) 的特徵計算公式與欄位名稱：
  'global_laplacian_var', 'global_edge_density', 'global_tv_density',
  'global_noise_estimation', 'global_gradient_mean', 'global_gradient_std',
  'global_contrast', 'global_skewness', 'global_kurtosis',
  'global_hsv_s_mean', 'global_hsv_v_mean', 'global_lbp_var',
  'global_glcm_contrast', 'global_glcm_homogeneity',
  'global_high_freq_energy_ratio', 'global_usm_0_5_energy',
  'global_usm_1_1_energy', 'total_pixels'
- 支援多核心平行運算 (`multiprocessing.Pool`) 與 `tqdm` 進度條，2,000 張圖片僅需數十秒即可完成。
- 自動對重複圖片（例如重複出現的 `003253.jpg`）進行快取，避免重複讀取與計算。
"""

import os
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

FEATURE_COLS = [
    'global_laplacian_var', 'global_edge_density',
    'global_tv_density', 'global_noise_estimation', 'global_gradient_mean',
    'global_gradient_std', 'global_contrast', 'global_skewness', 'global_kurtosis',
    'global_hsv_s_mean', 'global_hsv_v_mean', 'global_lbp_var', 'global_glcm_contrast',
    'global_glcm_homogeneity', 'global_high_freq_energy_ratio',
    'global_usm_0_5_energy', 'global_usm_1_1_energy', 'total_pixels'
]


def compute_single_image_features(task):
    """
    計算單張影像的 18 個全域特徵 (與訓練集提取公式 100% 一致)
    """
    img_name, img_path_str = task
    if not img_path_str or not os.path.isfile(img_path_str):
        return img_name, None, f"找不到圖片檔案: {img_path_str}"

    img = cv2.imread(img_path_str, cv2.IMREAD_COLOR)
    if img is None:
        return img_name, None, f"OpenCV 無法讀取圖片: {img_path_str}"

    try:
        h, w, _ = img.shape
        total_pixels = int(h * w)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0

        # 1. 影像統計與清晰度特徵
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        global_laplacian_var = float(laplacian.var())

        edges = cv2.Canny(gray, 100, 200)
        global_edge_density = float(np.count_nonzero(edges) / total_pixels)

        tv = np.sum(np.abs(np.diff(gray_float, axis=0))) + np.sum(np.abs(np.diff(gray_float, axis=1)))
        global_tv_density = float(tv / total_pixels)

        blur = cv2.GaussianBlur(gray_float, (3, 3), 0)
        global_noise_estimation = float(np.mean(np.abs(gray_float - blur)))

        gx = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        global_gradient_mean = float(grad_mag.mean())
        global_gradient_std = float(grad_mag.std())

        global_contrast = float(gray_float.std())

        flat = gray_float.flatten()
        mean_val = np.mean(flat)
        std_val = np.std(flat) + 1e-7
        global_skewness = float(np.mean(((flat - mean_val) / std_val) ** 3))
        global_kurtosis = float(np.mean(((flat - mean_val) / std_val) ** 4) - 3.0)

        # 2. 色彩空間特徵
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        global_hsv_s_mean = float(hsv[:, :, 1].mean() / 255.0)
        global_hsv_v_mean = float(hsv[:, :, 2].mean() / 255.0)

        # 3. 紋理與頻域特徵
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        global_lbp_var = float(lbp.var())

        gray_16 = (gray // 16).astype(np.uint8)
        glcm = graycomatrix(gray_16, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
        global_glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
        global_glcm_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])

        f = np.fft.fft2(gray_float)
        fshift = np.fft.fftshift(f)
        mag_spectrum = np.abs(fshift)
        cy, cx = h // 2, w // 2
        r = min(cy, cx) // 4
        y_grid, x_grid = np.ogrid[:h, :w]
        mask = (x_grid - cx)**2 + (y_grid - cy)**2 <= r**2
        low_energy = np.sum(mag_spectrum[mask]**2)
        total_energy = np.sum(mag_spectrum**2) + 1e-7
        global_high_freq_energy_ratio = float(1.0 - (low_energy / total_energy))

        # 4. USM 濾波能量特徵
        def usm_energy(image_gray, sigma):
            blurred = cv2.GaussianBlur(image_gray, (0, 0), sigma)
            usm = cv2.addWeighted(image_gray, 1.5, blurred, -0.5, 0)
            return float(np.var(usm))

        global_usm_0_5_energy = usm_energy(gray_float, 0.5)
        global_usm_1_1_energy = usm_energy(gray_float, 1.1)

        feats = {
            'global_laplacian_var': global_laplacian_var,
            'global_edge_density': global_edge_density,
            'global_tv_density': global_tv_density,
            'global_noise_estimation': global_noise_estimation,
            'global_gradient_mean': global_gradient_mean,
            'global_gradient_std': global_gradient_std,
            'global_contrast': global_contrast,
            'global_skewness': global_skewness,
            'global_kurtosis': global_kurtosis,
            'global_hsv_s_mean': global_hsv_s_mean,
            'global_hsv_v_mean': global_hsv_v_mean,
            'global_lbp_var': global_lbp_var,
            'global_glcm_contrast': global_glcm_contrast,
            'global_glcm_homogeneity': global_glcm_homogeneity,
            'global_high_freq_energy_ratio': global_high_freq_energy_ratio,
            'global_usm_0_5_energy': global_usm_0_5_energy,
            'global_usm_1_1_energy': global_usm_1_1_energy,
            'total_pixels': total_pixels
        }
        return img_name, feats, None

    except Exception as e:
        return img_name, None, str(e)


def resolve_image_path(img_name: str, candidate_dirs: list) -> str:
    if os.path.isfile(img_name):
        return img_name
    for d in candidate_dirs:
        p = os.path.join(d, img_name)
        if os.path.isfile(p):
            return p
    return ""


def main():
    parser = argparse.ArgumentParser(description="計算影像 18 個 Global Features 並附加到 CSV")
    parser.add_argument('--input_csv', type=str, default="E:/1DS_Result_New/1R_ALL_Best_Summary_New.csv",
                        help='輸入的 CSV 路徑 (需包含 image 欄位)')
    parser.add_argument('--image_dir', type=str, default="E:/1R_Labeling_Images_Denoised_New",
                        help='對應的圖片資料夾路徑 (可支援多個資料夾，以分號 ; 分隔)')
    parser.add_argument('--output_csv', type=str, default="",
                        help='輸出的 CSV 路徑 (若留空，預設輸出為 <input_csv>_with_features.csv；若要直接覆寫原檔可填相同路徑)')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1),
                        help='平行運算使用的 CPU 核心數')
    args, unknown = parser.parse_known_args()

    input_path = Path(args.input_csv)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入的 CSV 檔案: {input_path}")

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_with_features.csv")

    candidate_dirs = [p.strip() for p in args.image_dir.split(";") if p.strip()]

    print(f"📂 正在讀取 CSV: {input_path}")
    df = pd.read_csv(input_path)

    img_col = 'image' if 'image' in df.columns else ('Image' if 'Image' in df.columns else df.columns[0])
    df[img_col] = df[img_col].astype(str).str.strip()

    # 針對不重複圖片建立任務清單，避免重複計算
    unique_imgs = df[img_col].dropna().drop_duplicates().tolist()
    tasks = [(name, resolve_image_path(name, candidate_dirs)) for name in unique_imgs]

    print(f"🚀 開始計算 {len(unique_imgs)} 張不重複圖片的 18 個全域特徵 (使用 {args.workers} 核心)...")
    feat_map = {}
    errors = []

    with Pool(processes=args.workers) as pool:
        for img_name, feats, err in tqdm(
            pool.imap_unordered(compute_single_image_features, tasks),
            total=len(tasks),
            desc="Extracting Global Features",
            unit="img"
        ):
            if err is not None:
                errors.append((img_name, err))
                feat_map[img_name] = {col: np.nan for col in FEATURE_COLS}
            else:
                feat_map[img_name] = feats

    # 將計算好的特徵依照原 CSV 的 image 順序對應並 append 到後方
    feat_rows = [feat_map.get(name, {col: np.nan for col in FEATURE_COLS}) for name in df[img_col]]
    feat_df = pd.DataFrame(feat_rows, columns=FEATURE_COLS)

    # 若原表已有同名特徵欄位則先移除再覆蓋，確保無重複欄位
    existing_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if existing_feat_cols:
        df = df.drop(columns=existing_feat_cols)

    result_df = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("-" * 45)
    print(f"✅ 特徵提取並附加完成！共 {len(result_df)} 筆資料、{result_df.shape[1]} 個欄位")
    print(f"💾 已儲存至: {output_path}")

    if errors:
        print(f"⚠️ 共有 {len(errors)} 張圖片無法讀取或計算失敗（已填入 NaN）：")
        for name, err in errors[:10]:
            print(f"   - {name}: {err}")


if __name__ == '__main__':
    main()