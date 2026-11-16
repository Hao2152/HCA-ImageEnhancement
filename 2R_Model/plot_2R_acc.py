#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. 自動判斷 In_Is_In_Range 與 Out_Is_In_Range 並繪製 2-Region 聯立區間命中圖。
2. 自動篩選出 In & Out 皆 Out-of-Range 的失敗樣本，結合 plot_trials.py 邏輯，
   在等高線圖上標出 Best GT、GT 可接受矩形框與模型預測位置 (Pred)，輸出至指定資料夾。
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 設定字型與畫布參數
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Microsoft JhengHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def prepare_and_evaluate_ranges(df: pd.DataFrame, raw_csv_path: str = None, fallback_tol: float = 0.1) -> pd.DataFrame:
    """
    自動對齊欄位名稱，並在程式碼內自行計算 In_Is_In_Range 與 Out_Is_In_Range
    """
    col_mapping = {
        'Image': 'Image', 'image': 'Image',
        'In_Predicted_Alpha': 'In_Pred', 'Predicted_Alpha_In': 'In_Pred', 'Alpha_Pred_In': 'In_Pred',
        'Out_Predicted_Alpha': 'Out_Pred', 'Predicted_Alpha_Out': 'Out_Pred', 'Alpha_Pred_Out': 'Out_Pred',
        'In_Center_GT': 'In_GT', 'Alpha_GT_In': 'In_GT', 'in_alpha_center': 'In_GT',
        'acceptable_alpha_in_center': 'In_GT', 'best_alpha_in': 'In_GT',
        'Out_Center_GT': 'Out_GT', 'Alpha_GT_Out': 'Out_GT', 'out_alpha_center': 'Out_GT',
        'acceptable_alpha_out_center': 'Out_GT', 'best_alpha_out': 'Out_GT',
        'In_Min_Alpha': 'In_Min', 'in_alpha_min': 'In_Min', 'acceptable_alpha_in_min': 'In_Min',
        'In_Max_Alpha': 'In_Max', 'in_alpha_max': 'In_Max', 'acceptable_alpha_in_max': 'In_Max',
        'Out_Min_Alpha': 'Out_Min', 'out_alpha_min': 'Out_Min', 'acceptable_alpha_out_min': 'Out_Min',
        'Out_Max_Alpha': 'Out_Max', 'out_alpha_max': 'Out_Max', 'acceptable_alpha_out_max': 'Out_Max',
    }

    for src_col, dst_col in col_mapping.items():
        if src_col in df.columns and dst_col not in df.columns:
            df[dst_col] = df[src_col]

    # 若主 CSV 缺少上下界 (In_Min, In_Max, Out_Min, Out_Max)，從 raw_csv 根據檔名合併
    bound_cols = ['In_Min', 'In_Max', 'Out_Min', 'Out_Max']
    if not all(c in df.columns for c in bound_cols) and raw_csv_path and Path(raw_csv_path).exists():
        print(f"從 {raw_csv_path} 合併 GT 上下界區間欄位...")
        raw_df = pd.read_csv(raw_csv_path)
        for src_col, dst_col in col_mapping.items():
            if src_col in raw_df.columns and dst_col not in raw_df.columns:
                raw_df[dst_col] = raw_df[src_col]
        avail_bounds = [c for c in bound_cols if c in raw_df.columns]
        if 'Image' in raw_df.columns and len(avail_bounds) == 4:
            df = pd.merge(
                df,
                raw_df[['Image'] + avail_bounds].drop_duplicates(subset=['Image']),
                on='Image', how='left'
            )

    if not all(c in df.columns for c in bound_cols):
        if 'In_GT' in df.columns and 'Out_GT' in df.columns:
            print(f"⚠ CSV 中無 Min/Max 區間欄位，改以 GT ± {fallback_tol} 計算區間上下界...")
            df['In_Min'] = df['In_GT'] - fallback_tol
            df['In_Max'] = df['In_GT'] + fallback_tol
            df['Out_Min'] = df['Out_GT'] - fallback_tol
            df['Out_Max'] = df['Out_GT'] + fallback_tol
        else:
            raise ValueError(f"缺少計算區間所需的欄位！目前欄位: {list(df.columns)}")

    # === 自行判斷是否落在 GT 區間內 (加入 1e-5 避免浮點數誤差) ===
    eps = 1e-5
    df['In_Is_In_Range'] = (df['In_Pred'] >= df['In_Min'] - eps) & (df['In_Pred'] <= df['In_Max'] + eps)
    df['Out_Is_In_Range'] = (df['Out_Pred'] >= df['Out_Min'] - eps) & (df['Out_Pred'] <= df['Out_Max'] + eps)

    # 四種聯立命中狀態
    df['Both_In_Range'] = df['In_Is_In_Range'] & df['Out_Is_In_Range']
    df['In_Only'] = df['In_Is_In_Range'] & (~df['Out_Is_In_Range'])
    df['Out_Only'] = (~df['In_Is_In_Range']) & df['Out_Is_In_Range']
    df['Neither'] = (~df['In_Is_In_Range']) & (~df['Out_Is_In_Range'])

    return df


def find_trial_csv_for_image(trials_dir: Path, img_name: str) -> Path:
    """
    根據影像名稱 (例如 '000438.jpg') 在 trials_dir 中尋找對應的 '*_trials.csv'
    """
    stem = Path(img_name).stem
    candidates = [
        trials_dir / f"{stem}_trials.csv",
        trials_dir / f"{img_name}_trials.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # 若檔名有特殊前綴/後綴，使用 glob 模糊比對
    matches = list(trials_dir.glob(f"*{stem}*trials*.csv"))
    if matches:
        return matches[0]
    return None


def plot_failed_contours_with_predictions(
    failed_df: pd.DataFrame,
    trials_dir: str,
    fail_out_dir: str,
    drop_tol: float = 0.1
):
    """
    結合 plot_trials.py 的邏輯，針對失敗樣本繪製 2D 等高線圖 + GT 可接受框 + 預測位置標記
    """
    trials_path = Path(trials_dir)
    out_path = Path(fail_out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 先將失敗樣本的統計清單存成 CSV 方便檢視
    failed_csv_path = out_path / "failed_both_out_of_range_list.csv"
    failed_df.to_csv(failed_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 已將 {len(failed_df)} 筆失敗樣本明細匯出至: {failed_csv_path}")

    success_count = 0
    for idx, (_, row) in enumerate(failed_df.iterrows()):
        img_name = str(row['Image'])
        trial_file = find_trial_csv_for_image(trials_path, img_name)

        if trial_file is None or not trial_file.exists():
            print(f"⚠ [{idx + 1}/{len(failed_df)}] 找不到 {img_name} 對應的 trials CSV (搜尋路徑: {trials_path})，跳過繪圖。")
            continue

        t_df = pd.read_csv(trial_file)
        if t_df.empty or 'after_musiq' not in t_df.columns:
            continue

        orig_musiq = float(t_df['original_musiq'].iloc[0])
        best_row = t_df.loc[t_df['after_musiq'].idxmax()]
        best_musiq = float(best_row['after_musiq'])
        best_in = float(best_row['alpha_in'])
        best_out = float(best_row['alpha_out'])

        # 讀取模型預測位置與 GT 區間
        pred_in = float(row['In_Pred'])
        pred_out = float(row['Out_Pred'])
        in_min, in_max = float(row['In_Min']), float(row['In_Max'])
        out_min, out_max = float(row['Out_Min']), float(row['Out_Max'])

        # 建立 2D 樞紐分析表 (Pivot Table)
        piv = t_df.pivot_table(index='alpha_in', columns='alpha_out', values='after_musiq').sort_index(ascending=True)
        X = piv.columns.values  # alpha_out (0.1 ~ 1.0)
        Y = piv.index.values    # alpha_in (1.0 ~ 2.0)
        Z = piv.values          # after_musiq 矩陣
        XX, YY = np.meshgrid(X, Y)

        # 取得預測位置在 trials 網格上的 MUSIQ 分數 (若不在網格點上則用 CSV 的 MUSIQ_Pred)
        if 'MUSIQ_Pred' in row and pd.notna(row['MUSIQ_Pred']):
            pred_musiq = float(row['MUSIQ_Pred'])
        else:
            # 找最接近的網格點
            closest_in = Y[np.argmin(np.abs(Y - pred_in))]
            closest_out = X[np.argmin(np.abs(X - pred_out))]
            pred_musiq = float(piv.loc[closest_in, closest_out])

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        fig.patch.set_facecolor('white')

        # 1. 繪製填色熱力圖 (Colormap Background)
        im = ax.imshow(
            Z, extent=[X.min(), X.max(), Y.min(), Y.max()],
            origin='lower', aspect='auto', cmap='YlGnBu', alpha=0.85
        )
        cbar = plt.colorbar(im, ax=ax, pad=0.03)
        cbar.set_label('MUSIQ Score (after_musiq)', fontsize=10, fontweight='bold')

        # 2. 繪製精細等高線 (Isocontours)
        contours = ax.contour(XX, YY, Z, levels=10, colors='black', alpha=0.4, linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

        # 3. 標記衰退 <= drop_tol 的可接受等高邊界 (紅色虛線)
        threshold = best_musiq - drop_tol
        ax.contour(XX, YY, Z, levels=[threshold], colors='#E74C3C', linewidths=2.0, linestyles='--')


        # 5. 繪製從 Best GT 指向 Pred 的偏移向量箭頭
        ax.annotate(
            '', xy=(pred_out, pred_in), xytext=(best_out, best_in),
            arrowprops=dict(arrowstyle='->', color='#C0392B', lw=2.0, ls='-', shrinkA=6, shrinkB=6),
            zorder=6
        )

        # 6. 標記全域最佳解 (金色星號) 與 模型預測位置 (紅叉)
        ax.scatter(
            [best_out], [best_in], marker='*', s=240, color='#FFD700', edgecolor='#C0392B',
            linewidths=1.5, zorder=7, label=f'Best GT: in={best_in:.1f}, out={best_out:.1f} ({best_musiq:.2f})'
        )
        ax.scatter(
            [pred_out], [pred_in], marker='X', s=190, color='#FF1744', edgecolor='white',
            linewidths=1.2, zorder=8, label=f'Pred (Miss): in={pred_in:.1f}, out={pred_out:.1f} ({pred_musiq:.2f})'
        )

        diff_score = pred_musiq - best_musiq
        ax.set_title(
            f'[Both Out-of-Range] {img_name} | Orig: {orig_musiq:.1f} → GT: {best_musiq:.2f} vs Pred: {pred_musiq:.2f} (Diff: {diff_score:+.2f})\n'
            f'2-Region MUSIQ Contour (Red dashed: Drop ≤ {drop_tol:.1f} | Green box: GT Acceptable Range)',
            fontsize=10, fontweight='bold', pad=12, color='#1A252F'
        )
        ax.set_xlabel('Background Parameter: alpha_out (0.1 ~ 1.0)', fontsize=10.5, fontweight='bold')
        ax.set_ylabel('Foreground Parameter: alpha_in (1.0 ~ 2.0)', fontsize=10.5, fontweight='bold')

        ax.set_xticks(X)
        ax.set_yticks(Y)
        ax.set_xlim(X.min() - 0.03, X.max() + 0.03)
        ax.set_ylim(Y.min() - 0.03, Y.max() + 0.03)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='best', frameon=True, fontsize=8.8, facecolor='white', framealpha=0.92)

        plt.tight_layout()
        safe_stem = Path(img_name).stem
        out_img_path = out_path / f"failed_contour_{safe_stem}.png"
        plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
        plt.close()

        success_count += 1
        print(f"[{idx + 1}/{len(failed_df)}] 已輸出失敗診斷等高線圖: {out_img_path}")

    print(f"\n✅ 共成功產出 {success_count}/{len(failed_df)} 張失敗樣本等高解診斷圖至: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="繪製 2-Region 聯立命中圖，並針對雙未命中樣本輸出等高線 + 預測位置診斷圖")
    parser.add_argument("--csv", type=str, default="Z:\HCA/2R_Model/test_2region\hca_2region_all_musiq_eval_output/2region_musiq_comparison_results.csv", help="預測結果 CSV 路徑")
    parser.add_argument("--raw-csv", type=str, default="alpha_data_2region.csv", help="用來合併 GT 上下界區間的總表 CSV")
    parser.add_argument("--trials-dir", type=str, default="E:/2DS_Result/per_image", help="存放 *_trials.csv 的資料夾路徑")
    parser.add_argument("--fail-dir", type=str, default="./failed_both_missed_contours", help="輸出失敗樣本等高線圖的指定資料夾")
    parser.add_argument("--out-plot", type=str, default="joint_interval_accuracy_combined.png", help="聯立命中總圖輸出路徑")
    parser.add_argument("--drop-tol", type=float, default=0.1, help="等高線圖上紅色虛線標示的 MUSIQ 衰退容忍值 (預設 0.1)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = prepare_and_evaluate_ranges(df, raw_csv_path=args.raw_csv, fallback_tol=args.drop_tol)

    n_total = len(df)
    print(f"\n📊 --- 2-Region 區間命中統計 (N = {n_total}) ---")
    print(f"前景命中率 (In Accuracy):         {df['In_Is_In_Range'].mean():.2%} ({df['In_Is_In_Range'].sum()}/{n_total})")
    print(f"背景命中率 (Out Accuracy):        {df['Out_Is_In_Range'].mean():.2%} ({df['Out_Is_In_Range'].sum()}/{n_total})")
    print(f"雙參數同時命中 (Both In-Range):   {df['Both_In_Range'].mean():.2%} ({df['Both_In_Range'].sum()}/{n_total})")
    print(f"僅前景命中 (In Only):             {df['In_Only'].mean():.2%} ({df['In_Only'].sum()}/{n_total})")
    print(f"僅背景命中 (Out Only):            {df['Out_Only'].mean():.2%} ({df['Out_Only'].sum()}/{n_total})")
    print(f"皆未命中 (Both Missed):           {df['Neither'].mean():.2%} ({df['Neither'].sum()}/{n_total})")

    # =========================================================
    # 第一部分：繪製單張整合之 2D 聯立區間命中評估圖
    # =========================================================
    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    np.random.seed(42)
    jitter_x = np.random.uniform(-0.012, 0.012, size=n_total)
    jitter_y = np.random.uniform(-0.012, 0.012, size=n_total)

    cat_data = [
        ('Both In-Range (Joint Hit)', df['Both_In_Range'], '#27AE60', 'o', 55, 0.85, 4),
        ('Foreground (In) Only Hit', df['In_Only'], '#2980B9', 's', 60, 0.85, 3),
        ('Background (Out) Only Hit', df['Out_Only'], '#E67E22', '^', 65, 0.85, 3),
        ('Both Missed', df['Neither'], '#C0392B', 'X', 75, 0.95, 5),
    ]

    for label, mask, color, marker, size, alpha, zorder in cat_data:
        sub_x = df.loc[mask, 'Out_Pred'] + jitter_x[mask]
        sub_y = df.loc[mask, 'In_Pred'] + jitter_y[mask]
        count = mask.sum()
        pct = count / n_total * 100
        ax1.scatter(
            sub_x, sub_y, c=color, marker=marker, s=size, alpha=alpha,
            edgecolors='black', linewidth=0.7, label=f'{label}: {count} ({pct:.1f}%)', zorder=zorder
        )

    ax1.set_xlim(0.05, 1.05)
    ax1.set_ylim(0.95, 2.05)
    ax1.set_xticks(np.round(np.linspace(0.1, 1.0, 10), 1))
    ax1.set_yticks(np.round(np.linspace(1.0, 2.0, 11), 1))
    ax1.set_xlabel('Predicted Background (alpha_out)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Foreground (alpha_in)', fontsize=11, fontweight='bold')
    ax1.set_title(f'(A) 2D Joint Prediction Scatter with Hit Status (N={n_total})', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5, facecolor='white', framealpha=0.9)

    matrix = np.array([
        [df['Both_In_Range'].sum(), df['In_Only'].sum()],
        [df['Out_Only'].sum(), df['Neither'].sum()]
    ])
    matrix_pct = matrix / n_total * 100

    im = ax2.imshow(matrix_pct, cmap='Blues', vmin=0, vmax=max(75.0, matrix_pct.max()))
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Out: In-Range', 'Out: Out-of-Range'], fontsize=10, fontweight='bold')
    ax2.set_yticklabels(['In: In-Range', 'In: Out-of-Range'], fontsize=10, fontweight='bold')

    for i in range(2):
        for j in range(2):
            c = "white" if matrix_pct[i, j] > 35 else "black"
            ax2.text(j, i, f"{matrix[i, j]} samples\n({matrix_pct[i, j]:.1f}%)",
                     ha="center", va="center", color=c, fontsize=12, fontweight='bold')

    ax2.set_title('(B) Joint Interval Accuracy Matrix\n'
                  f'Joint Hit: {df["Both_In_Range"].mean():.1%} | In Hit: {df["In_Is_In_Range"].mean():.1%} | Out Hit: {df["Out_Is_In_Range"].mean():.1%}',
                  fontsize=11.5, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='Percentage (%)')

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=300)
    plt.close()
    print(f"✅ 聯立命中總圖已儲存至: {args.out_plot}")

    # =========================================================
    # 第二部分：針對 In & Out 皆 Out-of-Range 的 13 張樣本繪製等高線診斷圖
    # =========================================================
    failed_df = df[df['Neither']].copy()
    if not failed_df.empty:
        plot_failed_contours_with_predictions(
            failed_df=failed_df,
            trials_dir=args.trials_dir,
            fail_out_dir=args.fail_dir,
            drop_tol=args.drop_tol
        )
    else:
        print("🎉 沒有任何 In & Out 同時未命中的樣本！")


if __name__ == "__main__":
    main()