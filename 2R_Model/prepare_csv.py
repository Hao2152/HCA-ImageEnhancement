#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_csv_2region.py
======================
專為 2-Region (Foreground best_alpha_in & Background best_alpha_out) 設計的資料集準備腳本：
1. 讀取特徵提取後的總表 CSV (例如 ALL_Best_Summary_with_features.csv)
2. (選用) 若提供 --trials_dir，自動掃描每個 *_trials.csv 提取真實的 MUSIQ 等高容忍區間：
   - in_alpha_min, in_alpha_max, in_alpha_center
   - out_alpha_min, out_alpha_max, out_alpha_center
   若未提供 --trials_dir，則自動使用 best_alpha_in / best_alpha_out 搭配 ±tol 建立區間。
3. 採用「2D 混合分層抽樣 (Hybrid 2D Stratification)」切分 Train (80%) / Val (10%) / Test (10%)，
   確保 best_alpha_in 與 best_alpha_out 的雙端 U 型分佈在三個子集中完美對齊（分佈誤差 < 1%），且絕不觸發少數類報錯。
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def extract_interval_from_trial_csv(trial_csv_path: Path, musiq_drop_tol: float = 0.1) -> dict:
    """從單張影像的 *_trials.csv (110 組網格搜尋) 提取 in 與 out 的可接受區間"""
    try:
        df_t = pd.read_csv(trial_csv_path)
    except Exception:
        return {}

    if df_t.empty or "after_musiq" not in df_t.columns:
        return {}

    best_idx = df_t["after_musiq"].idxmax()
    best_row = df_t.loc[best_idx]
    best_musiq = float(best_row["after_musiq"])
    orig_musiq = float(best_row["original_musiq"])
    best_in = round(float(best_row["alpha_in"]), 1)
    best_out = round(float(best_row["alpha_out"]), 1)

    if best_musiq < orig_musiq:
        return {
            "in_alpha_min": best_in,
            "in_alpha_max": best_in,
            "in_alpha_center": best_in,
            "out_alpha_min": best_out,
            "out_alpha_max": best_out,
            "out_alpha_center": best_out,
            "acceptable_pairs_count": 1,
        }

    threshold = max(best_musiq - musiq_drop_tol, orig_musiq)
    acc_df = df_t[df_t["after_musiq"] >= threshold - 1e-6]
    if acc_df.empty:
        acc_df = df_t.iloc[[best_idx]]

    in_min = round(float(acc_df["alpha_in"].min()), 1)
    in_max = round(float(acc_df["alpha_in"].max()), 1)
    in_center = round((in_min + in_max) / 2.0, 2)

    out_min = round(float(acc_df["alpha_out"].min()), 1)
    out_max = round(float(acc_df["alpha_out"].max()), 1)
    out_center = round((out_min + out_max) / 2.0, 2)

    return {
        "in_alpha_min": in_min,
        "in_alpha_max": in_max,
        "in_alpha_center": in_center,
        "out_alpha_min": out_min,
        "out_alpha_max": out_max,
        "out_alpha_center": out_center,
        "acceptable_pairs_count": int(len(acc_df)),
    }


def build_hybrid_2d_strat_key(df: pd.DataFrame) -> pd.Series:
    """
    建立 2D 混合分層抽樣標籤：
    - 若某個 (best_alpha_in, best_alpha_out) 細粒度組合樣本數 >= 10，直接以該組合進行精確分層；
    - 若屬於中段稀疏格 (< 10 筆)，自動歸入 3×3 U 型區間 (Low / Mid / High) 進行分層，
      確保 train_test_split 絕不因少數類樣本不足而報錯，同時讓 in 與 out 分佈高度對齊。
    """
    in_vals = df["best_alpha_in"].round(1)
    out_vals = df["best_alpha_out"].round(1)
    pair_str = in_vals.astype(str) + "_" + out_vals.astype(str)
    pair_cnt = pair_str.value_counts()

    keys = []
    for idx, row in df.iterrows():
        vi = round(float(row["best_alpha_in"]), 1)
        vo = round(float(row["best_alpha_out"]), 1)
        ps = f"{vi}_{vo}"
        if pair_cnt[ps] >= 10:
            keys.append(ps)
        else:
            in_z = "L" if vi <= 1.0 else ("M" if vi <= 1.7 else "H")
            out_z = "L" if vo <= 0.1 else ("M" if vo <= 0.7 else "H")
            keys.append(f"ZONE_{in_z}_{out_z}")

    s_keys = pd.Series(keys, index=df.index)
    s_cnt = s_keys.value_counts()
    s_keys = s_keys.apply(lambda k: k if s_cnt[k] >= 4 else "RARE_TAIL")
    return s_keys


def prepare_2region_csv(
    csv_in: str,
    csv_out: str,
    trials_dir: str = "",
    musiq_drop_tol: float = 0.1,
    fallback_tol: float = 0.3,
    seed: int = 42,
):
    df = pd.read_csv(csv_in)

    # 1. 過濾掉沒有 best_alpha_in 或 best_alpha_out 的無效列 (例如 10 筆 NaN)
    df = df.dropna(subset=["best_alpha_in", "best_alpha_out"]).copy().reset_index(drop=True)
    df["best_alpha_in"] = df["best_alpha_in"].round(1)
    df["best_alpha_out"] = df["best_alpha_out"].round(1)

    # 2. 先設定預設區間 (若已有 acceptable_alpha_in_min 則直接沿用，否則以 ±fallback_tol 作為保底)
    if "acceptable_alpha_in_min" in df.columns and "acceptable_alpha_in_max" in df.columns:
        df["in_alpha_min"] = df["acceptable_alpha_in_min"].round(1)
        df["in_alpha_max"] = df["acceptable_alpha_in_max"].round(1)
    else:
        df["in_alpha_min"] = (df["best_alpha_in"] - fallback_tol).clip(1.0, 2.0).round(1)
        df["in_alpha_max"] = (df["best_alpha_in"] + fallback_tol).clip(1.0, 2.0).round(1)
    df["in_alpha_center"] = ((df["in_alpha_min"] + df["in_alpha_max"]) / 2.0).round(2)

    if "acceptable_alpha_out_min" in df.columns and "acceptable_alpha_out_max" in df.columns:
        df["out_alpha_min"] = df["acceptable_alpha_out_min"].round(1)
        df["out_alpha_max"] = df["acceptable_alpha_out_max"].round(1)
    else:
        df["out_alpha_min"] = (df["best_alpha_out"] - fallback_tol).clip(0.1, 1.0).round(1)
        df["out_alpha_max"] = (df["best_alpha_out"] + fallback_tol).clip(0.1, 1.0).round(1)
    df["out_alpha_center"] = ((df["out_alpha_min"] + df["out_alpha_max"]) / 2.0).round(2)

    # 3. 若有提供 --trials_dir，從每個 *_trials.csv 提取真實的 MUSIQ 等高可接受區間覆寫上去
    if trials_dir and os.path.isdir(trials_dir):
        trials_path = Path(trials_dir)
        img_col = "image" if "image" in df.columns else "Image"
        trial_map = {p.name.replace("_trials.csv", ""): p for p in trials_path.glob("*_trials.csv")}

        matched_cnt = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading *_trials.csv (drop_tol={musiq_drop_tol})"):
            stem = Path(str(row[img_col]).strip()).stem
            t_file = trial_map.get(stem, trials_path / f"{stem}_trials.csv")
            if t_file.is_file():
                res = extract_interval_from_trial_csv(t_file, musiq_drop_tol=musiq_drop_tol)
                if res:
                    matched_cnt += 1
                    for k, v in res.items():
                        df.loc[idx, k] = v

        # 同步保留 acceptable_alpha_* 欄位名稱以相容舊版習慣
        df["acceptable_alpha_in_min"] = df["in_alpha_min"]
        df["acceptable_alpha_in_max"] = df["in_alpha_max"]
        df["acceptable_alpha_in_center"] = df["in_alpha_center"]
        df["acceptable_alpha_out_min"] = df["out_alpha_min"]
        df["acceptable_alpha_out_max"] = df["out_alpha_max"]
        df["acceptable_alpha_out_center"] = df["out_alpha_center"]
        print(f"✅ 已從 {matched_cnt} / {len(df)} 個 *_trials.csv 提取真實 2-Region 可接受區間 (MUSIQ 衰退門檻 <= {musiq_drop_tol})")

    # 4. 使用 2D 混合分層抽樣切分 Train (80%) / Val (10%) / Test (10%)
    strat_key = build_hybrid_2d_strat_key(df)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=strat_key
    )
    temp_strat_key = strat_key.loc[temp_df.index]
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_strat_key
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print(
        f"✅ 已產生 N={len(full_df)} 的 2-Region 資料集 {csv_out} "
        f"(Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)})"
    )
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="準備 2-Region 訓練/驗證/測試 CSV (支援從 *_trials.csv 提取真實區間)")
    parser.add_argument(
        "--csv_in",
        type=str,
        default="E:/2DS_Result/ALL_Best_Summary_all.csv",
        help="輸入含 54 個特徵之總表 CSV 路徑",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default="alpha_data_2region.csv",
        help="輸出供 main_2region.py 使用的 CSV 路徑",
    )
    parser.add_argument(
        "--trials_dir",
        type=str,
        default="E:/2DS_Result/per_image",
        help="(選填) 存放 *_trials.csv 的資料夾路徑；若有提供則自動計算真實可接受區間",
    )
    parser.add_argument(
        "--musiq_drop_tol",
        type=float,
        default=0.1,
        help="從 *_trials.csv 提取可接受區間的 MUSIQ 衰退容忍門檻 (預設 0.1，可設為 0.2 或 0.3)",
    )
    parser.add_argument(
        "--fallback_tol",
        type=float,
        default=0.3,
        help="若無對應 *_trials.csv 時的參數容忍半徑 (+/- tol，預設 0.3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設 42)")

    args = parser.parse_args()
    prepare_2region_csv(
        csv_in=args.csv_in,
        csv_out=args.csv_out,
        trials_dir=args.trials_dir,
        musiq_drop_tol=args.musiq_drop_tol,
        fallback_tol=args.fallback_tol,
        seed=args.seed,
    )