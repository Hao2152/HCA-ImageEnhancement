#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1R 推論腳本 (直接讀取已計算好特徵的 CSV：2R_Better_1R_All_Future.csv / 2R_Better_1R_All_Feature.csv)
特點：
- 直接從輸入 CSV 讀取 `original_musiq` (若欄位名為 `1R_original_musiq` 或 `2R_original_musiq` 會自動對應) 與 18 個 global 特徵。
- 透過 datasets.py 的 extract_tabular_matrix 自動產生 19 個原始特徵 + 11 個 log 衍生特徵 = 30 維特徵。
- 修正 in_dim=X_infer.shape[1] (30 維)，完全對齊 main.py 訓練出的模型權重。
"""

import os
os.environ["KMP_WARNINGS"] = "FALSE"
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# 匯入訓練專案中的模組
from resnet_cls_mh import resnet50_cls_mh
from mh_utils import level_split, get_mh_weight
from datasets import TABULAR_COLS, extract_tabular_matrix

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ================= 輸入與輸出路徑設定 =================
parser.add_argument('--input_csv', type=str, default="E:/1DS_Result_2R_Data/2R_Better_1R_All_Future.csv",
                    help='待推論的 CSV 路徑 (已包含 global 特徵與 original_musiq)')
parser.add_argument('--output_csv', type=str, default='./2R_Better_1R_with_1R_Pred.csv',
                    help='輸出的預測結果 CSV 檔名')
parser.add_argument('--keep_all_cols', action='store_true', default=True,
                    help='是否在輸出 CSV 中保留輸入 CSV 的所有原始欄位 (包含 1R/2R GT 與 in/out 特徵)')

# ================= 訓練集參考與模型權重設定 =================
parser.add_argument('--train_csv', type=str, default='./ALL_Best_Summary_ExtendedFeatures_Advanced.csv',
                    help='訓練時使用的完整特徵 CSV (對應 main.py 的 --raw_csv_path)')
parser.add_argument('--split_csv', type=str, default='./alpha_data.csv',
                    help='訓練時使用的 split CSV (對應 main.py 的 --csv_path)')
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


def load_and_align_input_csv(csv_path):
    """
    讀取待推論 CSV，並自動對齊 `original_musiq` 與 18 個 global 特徵欄位
    """
    if not os.path.exists(csv_path):
        alt_path = csv_path.replace("Future", "Feature") if "Future" in csv_path else csv_path.replace("Feature", "Future")
        if os.path.exists(alt_path):
            print(f"⚠️ 找不到 {csv_path}，自動改為讀取: {alt_path}")
            csv_path = alt_path
        elif os.path.exists("./2R_Better_1R_All_Future.csv"):
            csv_path = "./2R_Better_1R_All_Future.csv"
        elif os.path.exists("./2R_Better_1R_All_Feature.csv"):
            csv_path = "./2R_Better_1R_All_Feature.csv"
        else:
            raise FileNotFoundError(f"找不到輸入的待推論 CSV 檔: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📂 成功讀取輸入檔案: {csv_path} (共 {len(df)} 筆資料)")

    # 1. 對齊圖片名稱欄位
    if 'image' not in df.columns:
        for cand in ['Image', 'filename', 'file']:
            if cand in df.columns:
                df['image'] = df[cand]
                break
        else:
            df['image'] = df.iloc[:, 0]

    # 2. 對齊 original_musiq 欄位 (支援 1R_original_musiq / 2R_original_musiq)
    if 'original_musiq' not in df.columns:
        if '1R_original_musiq' in df.columns:
            df['original_musiq'] = df['1R_original_musiq']
            print("ℹ️ 自動將 '1R_original_musiq' 對應為模型輸入的 'original_musiq'")
        elif '2R_original_musiq' in df.columns:
            df['original_musiq'] = df['2R_original_musiq']
            print("ℹ️ 自動將 '2R_original_musiq' 對應為模型輸入的 'original_musiq'")
        elif 'Original_MUSIQ' in df.columns:
            df['original_musiq'] = df['Original_MUSIQ']
        else:
            raise KeyError("CSV 中找不到 'original_musiq' 或 '1R_original_musiq' 欄位！")

    # 3. 檢查 TABULAR_COLS (來自 datasets.py) 所需的所有欄位是否齊全
    missing_cols = [c for c in TABULAR_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"輸入 CSV 缺少模型所需的特徵欄位: {missing_cols}")

    return df


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
    print(f"🚀 使用運算裝置: {device}")

    # 1. 直接從 CSV 讀取已經算好的 Global 特徵與 original_musiq
    df_infer = load_and_align_input_csv(args.input_csv)

    # 2. 準備訓練集 StandardScaler 並轉換特徵矩陣 (19 個原始特徵 + 11 個 log 特徵 = 30 維)
    print("⚙️ 正在載入訓練集分佈並正規化特徵 (StandardScaler)...")
    scaler, sindice, sclass2count, sweight = prepare_training_context()
    X_infer = scaler.transform(extract_tabular_matrix(df_infer))
    in_dim = X_infer.shape[1]

    infer_dataset = TensorDataset(torch.tensor(X_infer, dtype=torch.float32))
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. 建立模型並載入訓練好的權重 (關鍵修正: in_dim=X_infer.shape[1]，即 30 維)
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

    if not os.path.exists(args.resume):
        raise FileNotFoundError(f"找不到模型權重檔: {args.resume}")

    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 4. 模型批次推論
    preds_list = []
    print(f"\n開始執行模型推論 (Batch Size = {args.batch_size}, Head = {args.plot_head}, 實際輸入維度 = {in_dim})...")
    infer_pbar = tqdm(infer_loader, desc="Model Inference", unit="batch", ncols=100)

    with torch.no_grad():
        for (batch_x,) in infer_pbar:
            batch_x = batch_x.to(device, non_blocking=True)
            allpre = model(batch_x)
            pred_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)
            pred_val = pred_dict[args.plot_head]
            preds_list.extend(pred_val.cpu().view(-1).numpy())

    # 5. 整理輸出 CSV (同時保留小數點第一位 Predicted_Alpha 與原始連續值 Predicted_Alpha_Raw)
    pred_alpha_raw = np.round(preds_list, 4)
    pred_alpha = np.clip(np.round(preds_list, 1), 0.1, 2.0)

    if args.keep_all_cols:
        results_df = df_infer.copy()
        # 若原本沒有 'original_musiq' 而是自動從 '1R_original_musiq' 複製出來的，移除臨時欄以免重複
        if '1R_original_musiq' in results_df.columns and 'original_musiq' in results_df.columns:
            results_df.drop(columns=['original_musiq'], inplace=True)
        if '1R_best_alpha' in results_df.columns:
            insert_pos = results_df.columns.get_loc('1R_best_alpha') + 1
            results_df.insert(insert_pos, '1R_Predicted_Alpha', pred_alpha)
        else:
            results_df.insert(1, '1R_Predicted_Alpha', pred_alpha)
    else:
        results_df = pd.DataFrame({
            'Image': df_infer['image'].values,
            'Original_MUSIQ': df_infer['original_musiq'].values,
            '1R_Predicted_Alpha': pred_alpha,
            '1R_Predicted_Alpha_Raw': pred_alpha_raw
        })

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)

    print(f"\n✅ 推論完成！共成功預測 {len(results_df)} 筆資料，已儲存至: {args.output_csv}")
    preview_cols = [c for c in ['image', '1R_original_musiq', '1R_best_alpha', '1R_Predicted_Alpha'] if c in results_df.columns]
    print(results_df[preview_cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()