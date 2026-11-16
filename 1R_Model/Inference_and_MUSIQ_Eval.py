#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1R 推論 + Pred MUSIQ 評估 一條龍腳本 (Inference_and_MUSIQ_Eval.py)
功能：
1. 讀取 `2R_Better_1R_All_Future.csv` 的 global 特徵與 `1R_original_musiq`，預測 `1R_Predicted_Alpha`。
2. 緊接著根據 `1R_Predicted_Alpha` 到圖片資料夾 (`--image_dir`) 讀取原圖、套用 USM 並計算 `1R_Pred_MUSIQ`。
3. 直接將 `1R_Predicted_Alpha`、`1R_Pred_MUSIQ`、`1R_Pred_improvement`、`1R_Pred_vs_GT_Diff` 寫入 `--output_csv`。
"""

import os
os.environ["KMP_WARNINGS"] = "FALSE"
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pyiqa

from resnet_cls_mh import resnet50_cls_mh
from mh_utils import level_split, get_mh_weight
from datasets import TABULAR_COLS, extract_tabular_matrix

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ================= 輸入與輸出路徑設定 =================
parser.add_argument('--input_csv', type=str, default="E:/1DS_Result_2R_Data/2R_Better_1R_All_Future.csv",
                    help='待推論的 CSV 路徑 (已包含 global 特徵與 original_musiq)')
parser.add_argument('--image_dir', type=str, default="E:/2RLabeling_Images",
                    help='存放原始圖片的資料夾路徑 (可填多個路徑並以分號 ; 分隔)')
parser.add_argument('--output_csv', type=str, default='./2R_Better_1R_with_1R_Pred.csv',
                    help='輸出的預測與 MUSIQ 評估結果 CSV 檔名')
parser.add_argument('--eval_musiq', action='store_true', default=True,
                    help='是否在推論完 Alpha 後，直接計算 Pred Alpha 的 MUSIQ 分數並寫入 output_csv')

# ================= 訓練集參考與模型權重設定 =================
parser.add_argument('--train_csv', type=str, default='./alpha_data.csv',
                    help='訓練時使用的完整特徵 CSV (對應 main.py 的 --raw_csv_path)')
parser.add_argument('--split_csv', type=str, default='./alpha_data.csv',
                    help='訓練時使用的 split CSV (對應 main.py 的 --csv_path)')
parser.add_argument('--resume', type=str, default='./checkpoints_ablation/base_ckpt.best.pth.tar',
                    help='訓練好的模型權重路徑 (.pth.tar)')

# ================= 模型超參數 =================
parser.add_argument('--batch_size', type=int, default=64, help='推論批次大小')
parser.add_argument('--feat_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--temp', type=float, default=0.35)
parser.add_argument('--head_num', type=int, default=4)
parser.add_argument('--plot_head', type=str, default='adjcls',
                    choices=['combadd', 'combmul', 'adjcls', 'singlecls'])
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


def get_musiq_score(image_bgr: np.ndarray, musiq_metric, device) -> float:
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
        return sum(scores) / len(scores) if scores else 0.0


def apply_usm(orig_bgr_uint8: np.ndarray, alpha: float) -> np.ndarray:
    denoised = cv2.bilateralFilter(orig_bgr_uint8, d=5, sigmaColor=20, sigmaSpace=20)
    blurred = cv2.GaussianBlur(denoised, (5, 5), sigmaX=1.1, sigmaY=1.1)
    den_f32 = denoised.astype(np.float32)
    blur_f32 = blurred.astype(np.float32)
    final_f32 = den_f32 + alpha * (den_f32 - blur_f32)
    return np.clip(final_f32, 0.0, 255.0).astype(np.uint8)


def load_and_align_input_csv(csv_path):
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace("Future", "Feature") if "Future" in csv_path else csv_path.replace("Feature", "Future")
        if os.path.exists(alt_path):
            csv_path = alt_path
        elif os.path.exists("./2R_Better_1R_All_Future.csv"):
            csv_path = "./2R_Better_1R_All_Future.csv"
        else:
            raise FileNotFoundError(f"找不到輸入的待推論 CSV 檔: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📂 成功讀取輸入檔案: {csv_path} (共 {len(df)} 筆資料)")

    if 'image' not in df.columns:
        for cand in ['Image', 'filename', 'file']:
            if cand in df.columns:
                df['image'] = df[cand]
                break
        else:
            df['image'] = df.iloc[:, 0]

    if 'original_musiq' not in df.columns:
        if '1R_original_musiq' in df.columns:
            df['original_musiq'] = df['1R_original_musiq']
        elif '2R_original_musiq' in df.columns:
            df['original_musiq'] = df['2R_original_musiq']
        elif 'Original_MUSIQ' in df.columns:
            df['original_musiq'] = df['Original_MUSIQ']
        else:
            raise KeyError("CSV 中找不到 'original_musiq' 或 '1R_original_musiq' 欄位！")

    missing_cols = [c for c in TABULAR_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"輸入 CSV 缺少模型所需的特徵欄位: {missing_cols}")

    return df


def prepare_training_context():
    if os.path.exists(args.split_csv):
        df = pd.read_csv(args.split_csv)
        missing_cols = [c for c in TABULAR_COLS if c not in df.columns]
        if len(missing_cols) > 0:
            raw_df = pd.read_csv(args.train_csv)
            cols_to_merge = ['image'] + [
                c for c in TABULAR_COLS + ['acceptable_alpha_min']
                if c not in df.columns and c in raw_df.columns
            ]
            df = pd.merge(df, raw_df[cols_to_merge].drop_duplicates(subset=['image']), on='image', how='left')
    else:
        df = pd.read_csv(args.train_csv)

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
    print(f"🚀 使用運算裝置: {device}")

    # 1. 讀取 CSV 與特徵轉換 (30 維)
    df_infer = load_and_align_input_csv(args.input_csv)
    scaler, sindice, sclass2count, sweight = prepare_training_context()
    X_infer = scaler.transform(extract_tabular_matrix(df_infer))
    in_dim = X_infer.shape[1]

    infer_dataset = TensorDataset(torch.tensor(X_infer, dtype=torch.float32))
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 載入 HCA 1R 模型並推論 Alpha
    global_class_indice = np.round(np.linspace(0.1, 1.9, 19) + 0.05, 4)
    global_class2count = np.round(np.linspace(0.1, 2.0, 20), 4)

    model = resnet50_cls_mh(
        in_dim=in_dim, feat_dim=args.feat_dim, dropout=args.dropout,
        fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
        start_update=args.start_update, start_smooth=args.start_smooth,
        kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
        class_indice=global_class_indice, class2count=global_class2count,
        head_class_indice=sindice, head_class2count=sclass2count,
        head_weight=sweight, cmax=2.0
    )
    model = torch.nn.DataParallel(model).to(device)

    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)
    model.eval()

    preds_list = []
    print(f"\n[1/2] 執行 1R Alpha 模型推論 (輸入維度 = {in_dim})...")
    for (batch_x,) in tqdm(infer_loader, desc="Alpha Inference", unit="batch", ncols=100):
        with torch.no_grad():
            batch_x = batch_x.to(device, non_blocking=True)
            allpre = model(batch_x)
            pred_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)
            preds_list.extend(pred_dict[args.plot_head].cpu().view(-1).numpy())

    pred_alpha = np.clip(np.round(preds_list, 1), 0.1, 2.0)

    results_df = df_infer.copy()
    if '1R_original_musiq' in results_df.columns and 'original_musiq' in results_df.columns:
        results_df.drop(columns=['original_musiq'], inplace=True)

    insert_pos = results_df.columns.get_loc('1R_best_alpha') + 1 if '1R_best_alpha' in results_df.columns else 1
    results_df.insert(insert_pos, '1R_Predicted_Alpha', pred_alpha)

    # 先存一次推論結果，避免中途停止遺失進度
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    # 3. 計算 1R_Predicted_Alpha 的 MUSIQ 分數並直接寫回 output_csv
    if args.eval_musiq:
        print(f"\n[2/2] 載入 pyiqa MUSIQ 模型並計算預測參數的增強後 MUSIQ...")
        musiq_metric = pyiqa.create_metric("musiq", device=device)
        musiq_metric.eval()

        candidate_dirs = [Path(p.strip()) for p in args.image_dir.split(";") if p.strip()]
        pred_musiq_list, pred_gain_list, pred_diff_gt_list = [], [], []

        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating MUSIQ", unit="img", ncols=100):
            img_name = str(row['image']).strip()
            a_pred = float(row['1R_Predicted_Alpha'])
            a_gt = float(row['1R_best_alpha']) if ('1R_best_alpha' in row and pd.notna(row['1R_best_alpha'])) else np.nan
            orig_m = float(row['1R_original_musiq']) if ('1R_original_musiq' in row and pd.notna(row['1R_original_musiq'])) else np.nan
            gt_m = float(row['1R_after_musiq']) if ('1R_after_musiq' in row and pd.notna(row['1R_after_musiq'])) else np.nan

            # 若預測 Alpha 與最佳 GT Alpha 一致且已有 1R_after_musiq，直接沿用分數免重算！
            if pd.notna(gt_m) and pd.notna(a_gt) and abs(a_pred - a_gt) < 1e-5:
                m_pred = gt_m
            else:
                img_path = None
                for d in candidate_dirs:
                    cand = d / img_name
                    if cand.exists():
                        img_path = cand
                        break
                if img_path is None:
                    tqdm.write(f"⚠️ 找不到圖片: {img_name}，請確認 --image_dir 路徑")
                    pred_musiq_list.append(np.nan)
                    pred_gain_list.append(np.nan)
                    pred_diff_gt_list.append(np.nan)
                    continue

                orig_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if orig_img is None:
                    pred_musiq_list.append(np.nan)
                    pred_gain_list.append(np.nan)
                    pred_diff_gt_list.append(np.nan)
                    continue

                img_pred = apply_usm(orig_img, a_pred)
                m_pred = get_musiq_score(img_pred, musiq_metric, device)

            pred_musiq_list.append(m_pred)
            pred_gain_list.append((m_pred - orig_m) if pd.notna(orig_m) else np.nan)
            pred_diff_gt_list.append((m_pred - gt_m) if pd.notna(gt_m) else np.nan)

        pos = results_df.columns.get_loc('1R_Predicted_Alpha') + 1
        results_df.insert(pos, '1R_Pred_MUSIQ', pred_musiq_list)
        results_df.insert(pos + 1, '1R_Pred_improvement', pred_gain_list)
        results_df.insert(pos + 2, '1R_Pred_vs_GT_Diff', pred_diff_gt_list)

        results_df.to_csv(args.output_csv, index=False)

    print(f"\n✅ 全部完成！結果已儲存至: {args.output_csv}")
    preview_cols = [c for c in ['image', '1R_original_musiq', '1R_best_alpha', '1R_after_musiq', '1R_Predicted_Alpha', '1R_Pred_MUSIQ', '1R_Pred_improvement', '1R_Pred_vs_GT_Diff'] if c in results_df.columns]
    print(results_df[preview_cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
