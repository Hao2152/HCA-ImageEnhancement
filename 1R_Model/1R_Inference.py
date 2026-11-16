#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推論腳本 (Inference Script from Image-Only CSV or Image Folder)
假設情境：
- 待預測的資料尚未提取任何特徵，輸入的 CSV 只有 `image` (或 `Image`) 欄位。
- 腳本會根據 CSV 內的圖片名稱去 `--image_dir` 讀取原圖，即時提取 18 個特徵，
  並載入訓練好的 HCA 模型權重進行推論，最後輸出包含 `Image` 與 `Predicted_Alpha` 的 CSV。
"""

import os
os.environ["KMP_WARNINGS"] = "FALSE"
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler
import pyiqa
# 匯入訓練專案中的模組
from resnet_cls_mh import resnet50_cls_mh
from mh_utils import level_split, get_mh_weight
from datasets import TABULAR_COLS, extract_tabular_matrix
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ================= 輸入與輸出路徑設定 =================
parser.add_argument('--input_csv', type=str, default="E:/2RLabeling_Images/2R_labeling_image.csv",
                    help='待推論的 CSV 路徑 (裡面只需有 image 或 Image 欄位，尚未提取特徵)')
parser.add_argument('--image_dir', type=str, default='E:/2RLabeling_Images',
                    help='存放待推論圖片的資料夾路徑 (搭配 --input_csv 使用，或直接掃描此資料夾)')
parser.add_argument('--image_path', type=str, default='',
                    help='若只想測試單張圖片，可直接指定單張圖片完整路徑')
parser.add_argument('--output_csv', type=str, default='./prediction_results.csv',
                    help='輸出的預測結果 CSV 檔名')
parser.add_argument('--save_features', action='store_true', default=True,
                    help='是否在輸出的 CSV 中一併保留剛剛提取出的 18 個特徵欄位')

# ================= 訓練集參考與模型權重設定 =================
parser.add_argument('--train_csv', type=str, default='./alpha_data.csv',
                    help='訓練時使用的完整特徵 CSV (用來 fit StandardScaler 與初始化 Multi-Head 結構)')
parser.add_argument('--split_csv', type=str, default='./alpha_data.csv',
                    help='若 train_csv 沒有 split 欄位，可用此檔對齊 train split (選填)')
parser.add_argument('--resume', type=str, default='./checkpoints_ablation/base_ckpt.best.pth.tar',
                    help='訓練好的模型權重路徑 (.pth.tar)')

# ================= 模型超參數 (需與訓練時一致) =================
parser.add_argument('--batch_size', type=int, default=64, help='推論批次大小')
parser.add_argument('--feat_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--temp', type=float, default=0.35)
parser.add_argument('--head_num', type=int, default=4)
parser.add_argument('--plot_head', type=str, default='adjcls',
                    choices=['combadd', 'combmul', 'adjcls', 'singlecls'],
                    help='選擇使用哪一個預測頭輸出最終 Alpha')
parser.add_argument('--fds', action='store_true', default=False)
parser.add_argument('--fds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'])
parser.add_argument('--fds_ks', type=int, default=5)
parser.add_argument('--fds_sigma', type=float, default=1.0)
parser.add_argument('--start_update', type=int, default=0)
parser.add_argument('--start_smooth', type=int, default=1)
parser.add_argument('--bucket_num', type=int, default=20)
parser.add_argument('--bucket_start', type=int, default=0)
parser.add_argument('--fds_mmt', type=float, default=0.9)
args, unknown = parser.parse_known_args()

def process_single_image(args):
    img_name, full_path = args
    try:
        feats = extract_image_features(full_path)
        feats['image'] = img_name
        return feats, None
    except Exception as e:
        return None, (img_name, str(e))
def get_musiq_score(image_bgr: np.ndarray, musiq_metric, device) -> float:
    """
    使用完整影像或無損分塊計算 MUSIQ 分數
    """
    MAX_PIXELS = 2000000
    h, w = image_bgr.shape[:2]

    if h * w <= MAX_PIXELS:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                score = musiq_metric(img_tensor)
            return float(score.item())
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    else:
        PATCH_SIZE = 1024
        scores = []

        for y in range(0, h, PATCH_SIZE):
            for x in range(0, w, PATCH_SIZE):
                y_end = min(y + PATCH_SIZE, h)
                x_end = min(x + PATCH_SIZE, w)
                patch = image_bgr[y:y_end, x:x_end]

                if patch.shape[0] < 256 or patch.shape[1] < 256:
                    continue

                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_tensor = torch.from_numpy(patch_rgb).permute(2, 0, 1).float() / 255.0
                patch_tensor = patch_tensor.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        patch_score = musiq_metric(patch_tensor)
                    scores.append(float(patch_score.item()))
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        if not scores:
            return 0.0
        return sum(scores) / len(scores)


def extract_image_features(image_path, musiq_metric, device):
    """
    從單張原始圖片提取 original_musiq 與 18 個全局統計特徵
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"無法讀取圖片: {image_path}")

    # 1. 計算 original_musiq 分數
    original_musiq = get_musiq_score(img, musiq_metric, device)

    h, w, _ = img.shape
    total_pixels = int(h * w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32) / 255.0

    # 2. 影像統計與清晰度特徵
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

    # 3. 色彩空間特徵
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    global_hsv_s_mean = float(hsv[:, :, 1].mean() / 255.0)
    global_hsv_v_mean = float(hsv[:, :, 2].mean() / 255.0)

    # 4. 紋理與頻域特徵
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

    # 5. USM 濾波能量特徵
    def usm_energy(image_gray, sigma):
        blurred = cv2.GaussianBlur(image_gray, (0, 0), sigma)
        usm = cv2.addWeighted(image_gray, 1.5, blurred, -0.5, 0)
        return float(np.var(usm))

    global_usm_0_5_energy = usm_energy(gray_float, 0.5)
    global_usm_1_1_energy = usm_energy(gray_float, 1.1)

    return {
        'original_musiq': original_musiq,
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


def resolve_image_list():
    if args.input_csv:
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"找不到輸入的待推論 CSV 檔: {args.input_csv}")
        df_in = pd.read_csv(args.input_csv)
        img_col = None
        for col in ['image', 'Image', 'filename', 'file']:
            if col in df_in.columns:
                img_col = col
                break
        if img_col is None:
            img_col = df_in.columns[0]
            print(f"⚠️ CSV 未找到 'image' 欄位，自動使用第一欄 '{img_col}' 作為圖片名稱。")

        image_list = []
        for raw_name in df_in[img_col].astype(str).tolist():
            raw_name = raw_name.strip()
            if os.path.isfile(raw_name):
                image_list.append((os.path.basename(raw_name), raw_name))
            else:
                full_p = os.path.join(args.image_dir, raw_name)
                image_list.append((raw_name, full_p))
        return image_list

    elif args.image_path:
        if not os.path.isfile(args.image_path):
            raise FileNotFoundError(f"找不到指定的單張圖片: {args.image_path}")
        return [(os.path.basename(args.image_path), args.image_path)]

    elif os.path.isdir(args.image_dir):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(valid_exts)])
        return [(f, os.path.join(args.image_dir, f)) for f in files]

    else:
        raise ValueError("請提供 --input_csv (搭配 --image_dir)、--image_path 或有效的 --image_dir！")


def prepare_training_context():
    if os.path.exists(args.split_csv):
        df = pd.read_csv(args.split_csv)
        missing_cols = [c for c in TABULAR_COLS if c not in df.columns]
        if len(missing_cols) > 0:
            if not os.path.exists(args.train_csv):
                raise FileNotFoundError(f"需要 {args.train_csv} 來取得訓練集特徵的正規化參數 (StandardScaler)！")
            raw_df = pd.read_csv(args.train_csv)
            cols_to_merge = ['image'] + [
                c for c in TABULAR_COLS + ['acceptable_alpha_min']
                if c not in df.columns and c in raw_df.columns
            ]
            df = pd.merge(df, raw_df[cols_to_merge].drop_duplicates(subset=['image']), on='image', how='left')
    elif os.path.exists(args.train_csv):
        df = pd.read_csv(args.train_csv)
    else:
        raise FileNotFoundError(f"找不到訓練集參考檔 ({args.split_csv} 或 {args.train_csv})，無法初始化 StandardScaler。")

    df['acceptable_alpha_min'] = df['acceptable_alpha_min'].round(1)
    df_train = df[df['split'] == 'train'].reset_index(drop=True) if 'split' in df.columns else df

    scaler = StandardScaler()
    scaler.fit(extract_tabular_matrix(df_train))

    max_cls = 19
    train_orig_y_min = df_train['acceptable_alpha_min'].values
    global_count_dict = {x: 0 for x in range(max_cls + 1)}
    for label in train_orig_y_min:
        idx = int(np.round(float(label) * 10)) - 1
        global_count_dict[min(max_cls, max(0, idx))] += 1
    global_count_dict_sqrt = {k: np.sqrt(v) for k, v in global_count_dict.items()}

    _, sindice_idx, sclass2count_idx = level_split(
        [0, max_cls + 0.01], global_count_dict_sqrt, args.head_num, if_age=True, ob_vmax=max_cls
    )
    sweight = get_mh_weight(global_count_dict, sindice_idx)
    sindice = [[round((x + 1) * 0.1, 4) for x in idx_list] for idx_list in sindice_idx]
    sclass2count = [[round((x + 1) * 0.1, 4) for x in c2c_list] for c2c_list in sclass2count_idx]

    return scaler, sindice, sclass2count, sweight


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0. 初始化 pyiqa MUSIQ 評估模型
    print("正在載入 pyiqa MUSIQ 評估模型...")
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    musiq_metric.eval()

    # 1. 解析待推論圖片清單
    image_list = resolve_image_list()
    if len(image_list) == 0:
        print("未找到任何待處理的圖片。")
        return

    # 2. 【進度條一】提取 original_musiq 與 18 個全域特徵
    print(f"\n[1/2] 開始平行提取影像特徵 (使用 {cpu_count()} 核心)...")
    data_records = []
    valid_image_names = []

    tasks = [(name, path) for name, path in image_list]
    num_workers = max(1, cpu_count() - 1)  # 保留 1 核心給系統

    with Pool(num_workers) as pool:
        # 結合 tqdm 顯示平行處理進度
        results = list(
            tqdm(pool.imap(process_single_image, tasks), total=len(tasks), desc="Parallel Extracting", unit="img"))

    for feats, err in results:
        if err is not None:
            tqdm.write(f"⚠️ 略過圖片 {err[0]}: {err[1]}")
        elif feats is not None:
            data_records.append(feats)
            valid_image_names.append(feats['image'])

    if len(data_records) == 0:
        print("❌ 錯誤: 沒有成功提取任何圖片特徵，請確認圖片路徑是否正確。")
        return

    df_infer = pd.DataFrame(data_records)

    # 3. 準備訓練集 StandardScaler 並轉換特徵矩陣（此時包含 original_musiq）
    scaler, sindice, sclass2count, sweight = prepare_training_context()
    X_infer = scaler.transform(extract_tabular_matrix(df_infer))

    infer_dataset = TensorDataset(torch.tensor(X_infer, dtype=torch.float32))
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. 建立模型並載入訓練好的權重
    global_class_indice = np.round(np.linspace(0.1, 1.9, 19) + 0.05, 4)
    global_class2count = np.round(np.linspace(0.1, 2.0, 20), 4)

    model = resnet50_cls_mh(
        in_dim=len(TABULAR_COLS), feat_dim=args.feat_dim, dropout=args.dropout,
        fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
        start_update=args.start_update, start_smooth=args.start_smooth,
        kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
        class_indice=global_class_indice, class2count=global_class2count,
        head_class_indice=sindice, head_class2count=sclass2count,
        head_weight=sweight, cmax=2.0
    )
    model = torch.nn.DataParallel(model).to(device)

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"找不到模型權重檔: {args.resume}")

    checkpoint = torch.load(args.resume, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 5. 【進度條二】模型批次推論
    preds_list = []
    print(f"\n[2/2] 開始執行模型推論 (Batch Size = {args.batch_size}, Head = {args.plot_head})...")
    infer_pbar = tqdm(infer_loader, desc="Model Inference", unit="batch", ncols=100)

    with torch.no_grad():
        for (batch_x,) in infer_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            allpre = model(batch_x)
            pred_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)
            pred_val = pred_dict[args.plot_head]
            preds_list.extend(pred_val.cpu().view(-1).numpy())

    # 6. 整理輸出 CSV
    pred_alpha = np.clip(np.round(preds_list, 1), 0.1, 2.0)
    if args.save_features:
        results_df = df_infer[['image', 'original_musiq'] + TABULAR_COLS].copy()
        results_df.rename(columns={'image': 'Image'}, inplace=True)
        results_df.insert(1, 'Predicted_Alpha', pred_alpha)
    else:
        results_df = pd.DataFrame({
            'Image': valid_image_names,
            'Original_MUSIQ': df_infer['original_musiq'].values,
            'Predicted_Alpha': pred_alpha
        })

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    print(f"\n✅ 推論完成！共成功預測 {len(results_df)} 張圖片，已儲存至: {args.output_csv}")
    print(results_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()