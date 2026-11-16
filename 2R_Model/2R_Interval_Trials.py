#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從 2-Region *_trials.csv 資料夾自動提取真實 Acceptable Alpha Intervals (in & out)
並支援將模型預測之 (In_Predicted_Alpha, Out_Predicted_Alpha) 直接查表取得真實 MUSIQ_Pred！
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_interval_from_single_trial_csv(
    trial_csv_path: Path, musiq_drop_tol: float = 0.1
) -> dict:
    """
    讀取單張影像的 *_trials.csv (110 列)，
    找出滿足 after_musiq >= max(after_musiq) - musiq_drop_tol 的可接受區間。
    """
    df_t = pd.read_csv(trial_csv_path)
    if df_t.empty or "after_musiq" not in df_t.columns:
        return {}

    best_idx = df_t["after_musiq"].idxmax()
    best_row = df_t.loc[best_idx]
    best_musiq = float(best_row["after_musiq"])
    orig_musiq = float(best_row["original_musiq"])

    # 若最高分還低於原圖 (無效增強)，回傳邊界保守值
    if best_musiq < orig_musiq:
        best_in = float(best_row["alpha_in"])
        best_out = float(best_row["alpha_out"])
        return {
            "best_alpha_in": best_in,
            "best_alpha_out": best_out,
            "acceptable_alpha_in_min": best_in,
            "acceptable_alpha_in_max": best_in,
            "acceptable_alpha_in_center": best_in,
            "acceptable_alpha_out_min": best_out,
            "acceptable_alpha_out_max": best_out,
            "acceptable_alpha_out_center": best_out,
            "acceptable_pairs_count": 1,
        }

    # 篩選出 MUSIQ 衰退小於等於 musiq_drop_tol (且不低於原圖) 的所有合格試驗
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
        "image_from_trial": str(best_row["image"]),
        "best_alpha_in": round(float(best_row["alpha_in"]), 1),
        "best_alpha_out": round(float(best_row["alpha_out"]), 1),
        "acceptable_alpha_in_min": in_min,
        "acceptable_alpha_in_max": in_max,
        "acceptable_alpha_in_center": in_center,
        "acceptable_alpha_out_min": out_min,
        "acceptable_alpha_out_max": out_max,
        "acceptable_alpha_out_center": out_center,
        "acceptable_pairs_count": int(len(acc_df)),
    }


def build_2region_intervals_dataset(
    trials_dir: str,
    summary_csv: str,
    output_csv: str,
    musiq_drop_tol: float = 0.1,
):
    """
    掃描 trials_dir 下所有 *_trials.csv，計算真實可接受區間並合併回 summary_csv。
    """
    trials_path = Path(trials_dir)
    df_sum = pd.read_csv(summary_csv)

    img_col = "image" if "image" in df_sum.columns else "Image"

    # 建立 stem -> trial_csv_path 的快速索引表
    trial_files_map = {}
    for p in trials_path.glob("*_trials.csv"):
        stem = p.name.replace("_trials.csv", "")
        trial_files_map[stem] = p

    records = {}
    for idx, row in tqdm(
        df_sum.iterrows(), total=len(df_sum), desc=f"Extracting Intervals (tol={musiq_drop_tol})"
    ):
        img_name = str(row[img_col]).strip()
        stem = Path(img_name).stem
        t_path = trial_files_map.get(stem, trials_path / f"{stem}_trials.csv")
        if t_path.is_file():
            info = extract_interval_from_single_trial_csv(t_path, musiq_drop_tol=musiq_drop_tol)
            if info:
                records[idx] = info

    if records:
        interval_df = pd.DataFrame.from_dict(records, orient="index")
        cols_to_write = [
            "acceptable_alpha_in_min",
            "acceptable_alpha_in_max",
            "acceptable_alpha_in_center",
            "acceptable_alpha_out_min",
            "acceptable_alpha_out_max",
            "acceptable_alpha_out_center",
            "acceptable_pairs_count",
        ]
        for c in cols_to_write:
            df_sum.loc[interval_df.index, c] = interval_df[c]

    df_sum.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"完成！已從 {len(records)} / {len(df_sum)} 個 trials.csv 提取雙區域真實容忍區間，輸出至: {output_csv}")
    return df_sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 *_trials.csv 批次提取 2-Region Acceptable Intervals")
    parser.add_argument("--trials_dir", type=str, default="E:/2DS_Result/per_image", help="存放 *_trials.csv 的資料夾路徑")
    parser.add_argument("--summary_csv", type=str, default="E:/2DS_Result/ALL_Best_Summary_with_features.csv", help="輸入總表 CSV")
    parser.add_argument("--output_csv", type=str, default="E:/2DS_Result/ALL_Best_Summary_all.csv", help="輸出含區間之 CSV")
    parser.add_argument("--musiq_drop_tol", type=float, default=0.1, help="容許之 MUSIQ 衰退門檻 (預設 0.1，可調為 0.2 或 0.3)")
    args = parser.parse_args()

    build_2region_intervals_dataset(
        trials_dir=args.trials_dir,
        summary_csv=args.summary_csv,
        output_csv=args.output_csv,
        musiq_drop_tol=args.musiq_drop_tol,
    )