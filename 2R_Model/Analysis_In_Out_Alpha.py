#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析資料夾中所有 *_trials.csv 檔案，量化評估 alpha_in (前景) 與 alpha_out (背景)
對 after_musiq 分數的影響力大小 (變異數佔比與最大分差)。
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定字型與畫布參數
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def analyze_all_trials(trials_dir: str = ".", output_dir: str = "./trials_analysis_output"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    search_pattern = os.path.join(trials_dir, "*trials*.csv")
    trial_files = sorted(glob.glob(search_pattern))

    if not trial_files:
        print(f"⚠ 找不到任何符合 {search_pattern} 的檔案！")
        return

    print(f"找到 {len(trial_files)} 個 trials 檔案，開始進行影響力分析...")

    summary_records = []

    for f_path in trial_files:
        df = pd.read_csv(f_path)
        if df.empty or 'after_musiq' not in df.columns or 'alpha_in' not in df.columns or 'alpha_out' not in df.columns:
            continue

        img_name = str(df['image'].iloc[0]) if 'image' in df.columns else Path(f_path).name
        orig_musiq = float(df['original_musiq'].iloc[0])
        best_row = df.loc[df['after_musiq'].idxmax()]
        best_musiq = float(best_row['after_musiq'])

        # 1. 建立 2D 樞紐分析表 (Pivot Table: alpha_in vs alpha_out)
        piv = df.pivot_table(index='alpha_in', columns='alpha_out', values='after_musiq')

        # 2. 計算單獨調動 alpha_in / alpha_out 造成的最大分差 (Swing: Max - Min)
        # alpha_in 造成的平均最大落差 (固定 alpha_out 時的變動量平均)
        in_mean_swing = float((piv.max(axis=1) - piv.min(axis=1)).mean())
        # alpha_out 造成的平均最大落差 (固定 alpha_in 時的變動量平均)
        out_mean_swing = float((piv.max(axis=0) - piv.min(axis=0)).mean())

        # 3. 雙因子變異數拆解 (Two-Way ANOVA Sum of Squares)
        grand_mean = df['after_musiq'].mean()
        ss_total = float(((df['after_musiq'] - grand_mean) ** 2).sum())

        in_group_means = df.groupby('alpha_in')['after_musiq'].transform('mean')
        out_group_means = df.groupby('alpha_out')['after_musiq'].transform('mean')

        ss_in = float(((in_group_means - grand_mean) ** 2).sum())
        ss_out = float(((out_group_means - grand_mean) ** 2).sum())
        ss_interaction = max(0.0, ss_total - ss_in - ss_out)

        pct_in = (ss_in / ss_total * 100) if ss_total > 0 else 0.0
        pct_out = (ss_out / ss_total * 100) if ss_total > 0 else 0.0
        pct_inter = (ss_interaction / ss_total * 100) if ss_total > 0 else 0.0

        # 4. 判定誰的影響力更大
        dominant_factor = "Background (alpha_out)" if pct_out > pct_in else "Foreground (alpha_in)"

        summary_records.append({
            'file_name': Path(f_path).name,
            'image': img_name,
            'original_musiq': orig_musiq,
            'best_musiq': best_musiq,
            'max_gain': best_musiq - orig_musiq,
            'in_mean_swing': in_mean_swing,
            'out_mean_swing': out_mean_swing,
            'pct_var_in': pct_in,
            'pct_var_out': pct_out,
            'pct_var_interaction': pct_inter,
            'dominant_factor': dominant_factor
        })

    summary_df = pd.DataFrame(summary_records)
    out_csv = out_path / "trials_impact_summary.csv"
    summary_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 統計總表已儲存至: {out_csv}")

    # 印出終端機摘要報告
    print("\n📊 --- 所有 Trials 影響力分析總結 ---")
    print(f"分析影像總數: {len(summary_df)}")
    print(f"平均背景影響力佔比 (alpha_out ANOVA %): {summary_df['pct_var_out'].mean():.2f}%")
    print(f"平均前景影響力佔比 (alpha_in ANOVA %):  {summary_df['pct_var_in'].mean():.2f}%")
    print(f"平均背景造成的最大分差 (Out Swing):     {summary_df['out_mean_swing'].mean():.2f} 分")
    print(f"平均前景造成的最大分差 (In Swing):      {summary_df['in_mean_swing'].mean():.2f} 分")

    out_dominant_count = (summary_df['pct_var_out'] > summary_df['pct_var_in']).sum()
    print(f"以背景影響力為主導的影像數: {out_dominant_count} / {len(summary_df)}")

    # ==========================================
    # 5. 繪製統整視覺化圖表 (改用分佈直方圖與 Top 30 橫向長條圖，避免畫布超載)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'【2001 張影像之 2-Region 參數影響力大數據分析】(總計 {len(summary_df)} 筆 Trials)',
                 fontsize=14, fontweight='bold', y=1.02, color='#1A252F')

    # Panel A: 變異數佔比百分比分佈直方圖 (Distribution of Variance Contribution)
    ax1 = axes[0]
    ax1.hist(summary_df['pct_var_out'], bins=30, color='#E67E22', edgecolor='black', alpha=0.8,
             label='Background alpha_out (%)')
    ax1.hist(summary_df['pct_var_in'], bins=30, color='#2E86C1', edgecolor='black', alpha=0.5,
             label='Foreground alpha_in (%)')
    ax1.axvline(summary_df['pct_var_out'].mean(), color='#D35400', linestyle='-', linewidth=2,
                label=f'Mean Out: {summary_df["pct_var_out"].mean():.1f}%')
    ax1.axvline(summary_df['pct_var_in'].mean(), color='#1B4F72', linestyle='-', linewidth=2,
                label=f'Mean In: {summary_df["pct_var_in"].mean():.1f}%')
    ax1.set_title('【A】2001 張圖之變異數解釋佔比整體分佈', fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlabel('Explained Variance Ratio (%)', fontsize=10.5, fontweight='bold')
    ax1.set_ylabel('Image Count (Frequency)', fontsize=10.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=9, frameon=True)

    # Panel B: 抽樣前 30 張代表性影像的 Swing 比較
    ax2 = axes[1]
    sample_df = summary_df.head(30).copy()  # 取前 30 張避免畫面過載
    labels = [str(img).replace('.jpg', '') for img in sample_df['image']]
    y_pos = np.arange(len(labels))
    width = 0.38

    ax2.barh(y_pos - width / 2, sample_df['in_mean_swing'], width, label='Foreground Swing (1.0~2.0)', color='#2E86C1',
             edgecolor='black')
    ax2.barh(y_pos + width / 2, sample_df['out_mean_swing'], width, label='Background Swing (0.1~1.0)', color='#E67E22',
             edgecolor='black')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('MUSIQ Score Max-Min Swing (Score Drop/Gain)', fontsize=10.5, fontweight='bold')
    ax2.set_title('【B】前 30 張代表性影像之參數分數震盪幅度 (Swing)', fontsize=11.5, fontweight='bold', pad=10)
    ax2.grid(axis='x', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=9, frameon=True)

    plt.tight_layout()
    out_plot = out_path / "trials_impact_distribution_chart.png"
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 視覺化分析圖表已成功儲存至: {out_plot}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze alpha_in vs alpha_out impact from *_trials.csv files")
    parser.add_argument("--trials-dir", type=str, default="E:/2DS_Result/per_image", help="Directory containing *_trials.csv files")
    parser.add_argument("--out-dir", type=str, default="./trials_analysis_output", help="Output directory")
    args = parser.parse_args()

    analyze_all_trials(trials_dir=args.trials_dir, output_dir=args.out_dir)