#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解決點位重疊與 GT 上下界不一致導致看不清楚的問題：
1. 產出 2D 網格樣本數/命中率熱力圖 + 歸一化 GT 靶心區間圖 (PNG)
2. 產出對齊式 3D GT 容忍區間綠色方管隧道圖 (HTML)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def calc_signed_bound_offset(pred, mi, ma, inner_half_width=0.4, box_bound=0.5):
    """
    將每張圖各自的 [mi, ma] 對齊至 [-box_bound, +box_bound] 內：
    - 若落在 [mi, ma] 區間內：線性映射至 [-inner_half_width, +inner_half_width]
    - 若小於 mi：-box_bound - (mi - pred)
    - 若大於 ma：+box_bound + (pred - ma)
    """
    res = []
    for p, l, u in zip(pred, mi, ma):
        if p < l - 1e-5:
            res.append(-box_bound - (l - p))
        elif p > u + 1e-5:
            res.append(+box_bound + (p - u))
        else:
            if u - l < 1e-5:
                res.append(0.0)
            else:
                res.append(((p - l) / (u - l) - 0.5) * (2 * inner_half_width))
    return np.array(res)


def generate_clear_visualizations(
    pred_csv: str = "2region_musiq_comparison_results.csv",
    raw_csv: str = "ALL_Best_Summary_all.csv",
    out_png: str = "clear_2region_interval_accuracy_overview.png",
    out_html: str = "3d_2region_interval_tunnel_clear.html"
):
    df_pred = pd.read_csv(pred_csv)
    df_raw = pd.read_csv(raw_csv)

    merged = pd.merge(
        df_pred,
        df_raw[['image', 'acceptable_alpha_in_min', 'acceptable_alpha_in_max',
                'acceptable_alpha_out_min', 'acceptable_alpha_out_max']].drop_duplicates(subset=['image']),
        left_on='Image', right_on='image', how='left'
    )

    # 自行判斷 In 與 Out 是否落在 GT 區間內
    eps = 1e-5
    merged['In_Hit'] = (merged['Alpha_Pred_In'] >= merged['acceptable_alpha_in_min'] - eps) & (
        merged['Alpha_Pred_In'] <= merged['acceptable_alpha_in_max'] + eps
    )
    merged['Out_Hit'] = (merged['Alpha_Pred_Out'] >= merged['acceptable_alpha_out_min'] - eps) & (
        merged['Alpha_Pred_Out'] <= merged['acceptable_alpha_out_max'] + eps
    )

    def get_status(row):
        if row['In_Hit'] and row['Out_Hit']:
            return 'Both In-Range (Joint Hit)'
        elif row['In_Hit'] and not row['Out_Hit']:
            return 'Foreground (In) Only Hit'
        elif not row['In_Hit'] and row['Out_Hit']:
            return 'Background (Out) Only Hit'
        return 'Both Missed'

    merged['Status'] = merged.apply(get_status, axis=1)

    # =========================================================================
    # Part 1: 繪製 2D 網格命中率熱力圖 (左) + 歸一化 GT 區間靶心圖 (右)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    fig.patch.set_facecolor('white')

    ax1 = axes[0]
    in_bins = np.round(np.linspace(1.0, 2.0, 11), 1)
    out_bins = np.round(np.linspace(0.1, 1.0, 10), 1)

    count_grid = np.zeros((11, 10), dtype=int)
    hit_grid = np.full((11, 10), np.nan)

    for i, a_in in enumerate(in_bins):
        for j, a_out in enumerate(out_bins):
            sub = merged[(np.isclose(merged['Alpha_Pred_In'], a_in)) & (np.isclose(merged['Alpha_Pred_Out'], a_out))]
            if len(sub) > 0:
                count_grid[i, j] = len(sub)
                hit_grid[i, j] = (sub['In_Hit'] & sub['Out_Hit']).mean() * 100.0

    im1 = ax1.imshow(hit_grid, origin='lower', aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('Joint Interval Hit Rate (%)', fontsize=10, fontweight='bold')

    for i in range(11):
        for j in range(10):
            cnt = count_grid[i, j]
            if cnt > 0:
                hr = hit_grid[i, j]
                txt_color = 'white' if (hr < 35 or hr > 88) else 'black'
                ax1.text(j, i, f"n={cnt}\n{hr:.0f}%", ha='center', va='center', fontsize=7.5, fontweight='bold', color=txt_color)
            else:
                ax1.text(j, i, "·", ha='center', va='center', fontsize=10, color='#BDC3C7')

    ax1.set_xticks(range(10))
    ax1.set_xticklabels([f"{x:.1f}" for x in out_bins], fontsize=9.5, fontweight='bold')
    ax1.set_yticks(range(11))
    ax1.set_yticklabels([f"{y:.1f}" for y in in_bins], fontsize=9.5, fontweight='bold')
    ax1.set_xlabel('Predicted Background Alpha (α_out: 0.1 ~ 1.0)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Foreground Alpha (α_in: 1.0 ~ 2.0)', fontsize=11, fontweight='bold')
    ax1.set_title(f'(A) 2D Grid Joint Hit Rate & Sample Count (N = {len(merged):,})\nColor = Joint Hit % | Text = Sample Count & Hit Rate', fontsize=11.5, fontweight='bold', pad=10)

    ax2 = axes[1]
    np.random.seed(42)
    jx = np.random.uniform(-0.025, 0.025, len(merged))
    jy = np.random.uniform(-0.025, 0.025, len(merged))

    merged['Norm_Out'] = calc_signed_bound_offset(merged['Alpha_Pred_Out'], merged['acceptable_alpha_out_min'], merged['acceptable_alpha_out_max'], 0.4, 0.5) + jx
    merged['Norm_In'] = calc_signed_bound_offset(merged['Alpha_Pred_In'], merged['acceptable_alpha_in_min'], merged['acceptable_alpha_in_max'], 0.4, 0.5) + jy

    ax2.axvspan(-0.5, 0.5, color='#E8F8F5', alpha=0.6, zorder=1)
    ax2.axhspan(-0.5, 0.5, color='#E8F8F5', alpha=0.6, zorder=1)
    target_rect = plt.Rectangle((-0.5, -0.5), 1.0, 1.0, facecolor='#A3E4D7', alpha=0.55, edgecolor='#16A085', linewidth=2.5, zorder=2, label='GT Acceptable Interval Box (Target Zone)')
    ax2.add_patch(target_rect)

    ax2.axvline(-0.5, color='#16A085', linestyle='--', lw=1.2)
    ax2.axvline(0.5, color='#16A085', linestyle='--', lw=1.2)
    ax2.axhline(-0.5, color='#16A085', linestyle='--', lw=1.2)
    ax2.axhline(0.5, color='#16A085', linestyle='--', lw=1.2)

    cat_specs = [
        ('Both In-Range (Joint Hit)', '#27AE60', 'o', 28, 0.75, 4),
        ('Foreground (In) Only Hit', '#2980B9', 's', 40, 0.85, 5),
        ('Background (Out) Only Hit', '#E67E22', '^', 48, 0.90, 5),
        ('Both Missed', '#C0392B', 'X', 75, 1.0, 6),
    ]

    for s_key, col, mk, sz, al, zo in cat_specs:
        sub = merged[merged['Status'] == s_key]
        cnt = len(sub)
        pct = cnt / len(merged) * 100
        ax2.scatter(sub['Norm_Out'], sub['Norm_In'], c=col, marker=mk, s=sz, alpha=al, edgecolors='black', linewidth=0.5,
                    label=f"{s_key}: {cnt} ({pct:.1f}%)", zorder=zo)

    ax2.set_xlim(-1.45, 1.45)
    ax2.set_ylim(-1.45, 1.45)
    ax2.set_xticks([-1.2, -0.9, -0.6, 0.0, 0.6, 0.9, 1.2])
    ax2.set_xticklabels(['-0.7\n(Under)', '-0.4\n(Under)', '-0.1\n(Under)', 'Inside GT\n[Min, Max]', '+0.1\n(Over)', '+0.4\n(Over)', '+0.7\n(Over)'], fontsize=8.5, fontweight='bold')
    ax2.set_yticks([-1.2, -0.9, -0.6, 0.0, 0.6, 0.9, 1.2])
    ax2.set_yticklabels(['-0.7 (Under)', '-0.4 (Under)', '-0.1 (Under)', 'Inside GT\n[Min, Max]', '+0.1 (Over)', '+0.4 (Over)', '+0.7 (Over)'], fontsize=8.5, fontweight='bold')

    ax2.set_xlabel('Background (α_out) Position Relative to GT Interval [Min, Max]', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Foreground (α_in) Position Relative to GT Interval [Min, Max]', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Normalized Target-Box Interval Accuracy View\nCenter Green Box = Inside [Min, Max] | Outside = Exact Distance Missed', fontsize=11.5, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', frameon=True, fontsize=8.8, facecolor='white', framealpha=0.92)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"✅ 已儲存清晰版 2D 區間準確率總覽圖: {out_png}")

    # =========================================================================
    # Part 2: 繪製 3D 對齊式 GT 容忍區間綠色方管隧道 (HTML)
    # =========================================================================
    df_sorted = merged.sort_values(by=['original_musiq']).reset_index(drop=True)
    df_sorted['Sample_Idx'] = np.arange(1, len(df_sorted) + 1)
    df_sorted['Tunnel_Y_In'] = calc_signed_bound_offset(df_sorted['Alpha_Pred_In'], df_sorted['acceptable_alpha_in_min'], df_sorted['acceptable_alpha_in_max'], 0.23, 0.25)
    df_sorted['Tunnel_Z_Out'] = calc_signed_bound_offset(df_sorted['Alpha_Pred_Out'], df_sorted['acceptable_alpha_out_min'], df_sorted['acceptable_alpha_out_max'], 0.23, 0.25)

    fig3d = go.Figure()
    x_min, x_max = 1, len(df_sorted)
    vx = [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min]
    vy = [-0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25]
    vz = [-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25]

    fig3d.add_trace(go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=[0, 0, 4, 4, 0, 0, 2, 2],
        j=[1, 2, 5, 6, 1, 5, 3, 7],
        k=[2, 3, 6, 7, 5, 4, 7, 6],
        color='#2ECC71', opacity=0.18,
        name='GT Acceptable Tunnel (In & Out Range)',
        showlegend=True, hoverinfo='skip'
    ))

    edge_x = [x_min, x_max, None, x_min, x_max, None, x_min, x_max, None, x_min, x_max, None,
              x_min, x_min, x_min, x_min, x_min, None, x_max, x_max, x_max, x_max, x_max]
    edge_y = [-0.25, -0.25, None, 0.25, 0.25, None, 0.25, 0.25, None, -0.25, -0.25, None,
              -0.25, 0.25, 0.25, -0.25, -0.25, None, -0.25, 0.25, 0.25, -0.25, -0.25]
    edge_z = [-0.25, -0.25, None, -0.25, -0.25, None, 0.25, 0.25, None, 0.25, 0.25, None,
              -0.25, -0.25, 0.25, 0.25, -0.25, None, -0.25, -0.25, 0.25, 0.25, -0.25]

    fig3d.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        name='GT Interval Boundary Edges',
        line=dict(color='#16A085', width=4), hoverinfo='skip'
    ))

    miss_df = df_sorted[df_sorted['Status'] != 'Both In-Range (Joint Hit)']
    err_x, err_y, err_z = [], [], []
    for _, r in miss_df.iterrows():
        sx = r['Sample_Idx']
        ty = np.clip(r['Tunnel_Y_In'], -0.25, 0.25)
        tz = np.clip(r['Tunnel_Z_Out'], -0.25, 0.25)
        err_x.extend([sx, sx, None])
        err_y.extend([r['Tunnel_Y_In'], ty, None])
        err_z.extend([r['Tunnel_Z_Out'], tz, None])

    fig3d.add_trace(go.Scatter3d(
        x=err_x, y=err_y, z=err_z, mode='lines',
        name='Out-of-Range Deviation Stem',
        line=dict(color='rgba(192, 57, 43, 0.55)', width=2.5), hoverinfo='skip'
    ))

    status_list = [
        ('Both In-Range (Joint Hit)', '#27AE60', 'circle', 3.5, 0.65),
        ('Foreground (In) Only Hit', '#2980B9', 'square', 5.0, 0.90),
        ('Background (Out) Only Hit', '#E67E22', 'diamond', 5.5, 0.95),
        ('Both Missed', '#E74C3C', 'x', 7.0, 1.0),
    ]

    for s_name, color, sym, sz, op in status_list:
        sub = df_sorted[df_sorted['Status'] == s_name]
        cnt = len(sub)
        pct = cnt / len(df_sorted) * 100
        htext = [
            f"<b>Image:</b> {r['Image']} (Rank #{r['Sample_Idx']})<br>"
            f"<b>Status:</b> {r['Status']}<br>"
            f"<b>Foreground α_in (Y):</b> Pred={r['Alpha_Pred_In']:.1f} | GT=[{r['acceptable_alpha_in_min']:.1f}, {r['acceptable_alpha_in_max']:.1f}] (Best: {r['Alpha_GT_In']:.1f})<br>"
            f"<b>Background α_out (Z):</b> Pred={r['Alpha_Pred_Out']:.1f} | GT=[{r['acceptable_alpha_out_min']:.1f}, {r['acceptable_alpha_out_max']:.1f}] (Best: {r['Alpha_GT_Out']:.1f})<br>"
            f"<b>MUSIQ:</b> Orig={r['original_musiq']:.2f} → GT={r['MUSIQ_GT']:.2f} vs Pred={r['MUSIQ_Pred']:.2f} (Diff={r['MUSIQ_Diff']:+.2f})"
            for _, r in sub.iterrows()
        ]
        fig3d.add_trace(go.Scatter3d(
            x=sub['Sample_Idx'], y=sub['Tunnel_Y_In'], z=sub['Tunnel_Z_Out'],
            mode='markers', name=f"{s_name}: {cnt} ({pct:.1f}%)",
            marker=dict(size=sz, color=color, symbol=sym, opacity=op, line=dict(color='black', width=0.5)),
            text=htext, hoverinfo='text'
        ))

    fig3d.update_layout(
        title=dict(
            text="<b>3D 雙參數容忍區間隧道圖 (Aligned 3D GT Interval Tunnel)</b>",
            x=0.5, xanchor='center', font=dict(size=17)
        ),
        scene=dict(
            xaxis=dict(title='<b>X: Sorted Sample Index (依原圖 MUSIQ 由低至高)</b>'),
            yaxis=dict(
                title='<b>Y: 前景 α_in 相對 GT 區間位置</b>', range=[-1.15, 1.15],
                tickvals=[-0.95, -0.55, -0.25, 0.0, 0.25, 0.55, 0.95],
                ticktext=['-0.7(低)', '-0.3', 'In_Min', '區間內', 'In_Max', '+0.3', '+0.7(高)']
            ),
            zaxis=dict(
                title='<b>Z: 背景 α_out 相對 GT 區間位置</b>', range=[-1.15, 1.15],
                tickvals=[-0.95, -0.55, -0.25, 0.0, 0.25, 0.55, 0.95],
                ticktext=['-0.7(低)', '-0.3', 'Out_Min', '區間內', 'Out_Max', '+0.3', '+0.7(高)']
            ),
            camera=dict(eye=dict(x=1.75, y=-1.55, z=1.15)),
            aspectratio=dict(x=1.5, y=1.0, z=1.0)
        ),
        template='plotly_white'
    )

    fig3d.write_html(out_html, include_plotlyjs='cdn')
    print(f"✅ 已儲存清晰版 3D 容忍區間隧道網頁: {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Z:\HCA/2R_Model/test_2region\hca_2region_all_musiq_eval_output/2region_musiq_comparison_results.csv")
    parser.add_argument("--raw-csv", type=str, default="alpha_data_2region.csv")
    parser.add_argument("--out-png", type=str, default="clear_2region_interval_accuracy_overview.png")
    parser.add_argument("--out-html", type=str, default="3d_2region_interval_tunnel_clear.html")
    args = parser.parse_args()
    generate_clear_visualizations(args.csv, args.raw_csv, args.out_png, args.out_html)