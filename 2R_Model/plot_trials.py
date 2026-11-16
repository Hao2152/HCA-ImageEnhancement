#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自動讀取資料夾內所有的 *_trials.csv 檔案，
為每一張影像繪製對應的 2-Region (alpha_in vs alpha_out) 等高線與熱力圖。
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


def plot_trials_contours(trials_dir: str = ".", output_dir: str = "./trials_contour_plots"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    search_pattern = os.path.join(trials_dir, "*trials*.csv")
    trial_files = sorted(glob.glob(search_pattern))

    if not trial_files:
        print(f"⚠ 找不到任何符合 {search_pattern} 的檔案！")
        return

    print(f"找到 {len(trial_files)} 個 trials 檔案，開始繪製等高解圖片...")

    for idx, f_path in enumerate(trial_files):
        df = pd.read_csv(f_path)
        if df.empty or 'after_musiq' not in df.columns or 'alpha_in' not in df.columns or 'alpha_out' not in df.columns:
            continue

        img_name = str(df['image'].iloc[0]) if 'image' in df.columns else Path(f_path).stem
        orig_musiq = float(df['original_musiq'].iloc[0])

        # 找出最佳點 (Max MUSIQ)
        best_row = df.loc[df['after_musiq'].idxmax()]
        best_musiq = float(best_row['after_musiq'])
        best_in = float(best_row['alpha_in'])
        best_out = float(best_row['alpha_out'])

        # 建立 2D 樞紐分析表 (Pivot Table) 供等高線與熱力圖使用
        piv = df.pivot_table(index='alpha_in', columns='alpha_out', values='after_musiq').sort_index(ascending=True)

        X = piv.columns.values  # alpha_out (0.1 ~ 1.0)
        Y = piv.index.values  # alpha_in (1.0 ~ 2.0)
        Z = piv.values  # after_musiq 分數矩陣

        XX, YY = np.meshgrid(X, Y)

        # 建立畫布
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('white')

        # 1. 繪製填色熱力圖 (Filled Contour / Colormap Background)
        im = ax.imshow(
            Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto', cmap='YlGnBu', alpha=0.85
        )
        cbar = plt.colorbar(im, ax=ax, pad=0.03)
        cbar.set_label('MUSIQ Score (after_musiq)', fontsize=10, fontweight='bold')

        # 2. 繪製精細等高線 (Isocontours)
        num_levels = 10
        contours = ax.contour(XX, YY, Z, levels=num_levels, colors='black', alpha=0.4, linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

        # 3. 標記衰退 <= 0.3 的可接受等高邊界 (Acceptable Region Threshold)
        threshold = best_musiq - 0.1
        accept_contour = ax.contour(
            XX, YY, Z, levels=[threshold],
            colors='#E74C3C', linewidths=2.0, linestyles='--'
        )

        # 4. 標記全域最佳解 (Star Marker)
        ax.scatter([best_out], [best_in], marker='*', s=220, color='#FFD700', edgecolor='#C0392B', linewidths=1.5,
                   zorder=6, label=f'Best: in={best_in:.1f}, out={best_out:.1f}')

        # 圖表標題與軸標籤
        ax.set_title(
            f'Image: {img_name} | Orig: {orig_musiq:.1f} → Max: {best_musiq:.2f}\n'
            f'2-Region MUSIQ Response Surface & Isocontours (Red dashed: Drop ≤ 0.1)',
            fontsize=11, fontweight='bold', pad=12, color='#1A252F'
        )
        ax.set_xlabel('Background Parameter: alpha_out (0.1 ~ 1.0)', fontsize=10.5, fontweight='bold')
        ax.set_ylabel('Foreground Parameter: alpha_in (1.0 ~ 2.0)', fontsize=10.5, fontweight='bold')

        ax.set_xticks(X)
        ax.set_yticks(Y)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', frameon=True, fontsize=9)

        plt.tight_layout()

        # 儲存圖片
        safe_name = Path(f_path).stem.replace('_trials', '')
        out_img_path = out_path / f"contour_{safe_name}.png"
        plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[{idx + 1}/{len(trial_files)}] 已產出等高線圖: {out_img_path}")

    print(f"\n✅ 所有等高解圖片已成功儲存至資料夾: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot contour and response surface for *_trials.csv files")
    parser.add_argument("--trials-dir", type=str, default="E:/2DS_Result/per_image", help="Directory containing *_trials.csv files")
    parser.add_argument("--out-dir", type=str, default="./trials_contour_plots",
                        help="Output directory for contour plots")
    args = parser.parse_args()

    plot_trials_contours(trials_dir=args.trials_dir, output_dir=args.out_dir)