import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import pyiqa
import matplotlib.pyplot as plt

# ==========================================
# 1. MUSIQ 評估指標初始化與核心函數
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading MUSIQ metric on {device}...")
musiq_metric = pyiqa.create_metric("musiq", device=device)


def get_musiq_score(image_bgr: np.ndarray) -> float:
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


# ==========================================
# 2. 2-Region 影像處理管線與文字繪製函數
# ==========================================
def apply_2region_usm(
        orig_bgr_uint8: np.ndarray,
        mask_uint8: np.ndarray,
        alpha_in: float,
        alpha_out: float
) -> np.ndarray:
    """
    對整張影像進行雙邊濾波（Denoised），但根據 Saliency Mask 區分前景與背景，
    並各自套用不同的 Alpha 增益進行非銳化遮罩（USM）增強。
    """
    # 1. 針對整張影像進行雙邊濾波（去噪）與高斯模糊
    denoised = cv2.bilateralFilter(orig_bgr_uint8, d=5, sigmaColor=20, sigmaSpace=20)
    blurred = cv2.GaussianBlur(denoised, (5, 5), sigmaX=1.1, sigmaY=1.1)

    den_f32 = denoised.astype(np.float32)
    blur_f32 = blurred.astype(np.float32)

    # 2. 準備 Mask (將範圍標準化至 0.0 ~ 1.0)
    if len(mask_uint8.shape) == 2:
        mask_f32 = mask_uint8.astype(np.float32) / 255.0
        mask_f32 = np.expand_dims(mask_f32, axis=-1)
    else:
        mask_f32 = mask_uint8.astype(np.float32)[:, :, :1] / 255.0

    # 3. 計算 2-Region 自適應 Alpha 矩陣：前景 (Alpha_In) 與 背景 (Alpha_Out)
    pixel_alpha = mask_f32 * alpha_in + (1.0 - mask_f32) * alpha_out

    # 4. 套用銳化公式：Final = Denoised + Alpha * (Denoised - Blurred)
    sharpen_diff = den_f32 - blur_f32
    final_f32 = den_f32 + pixel_alpha * sharpen_diff

    final = np.clip(final_f32, 0.0, 255.0).astype(np.uint8)
    return final


def draw_text_with_background(img: np.ndarray, text1: str, text2: str, position=(20, 50)):
    """
    在影像上繪製帶有半透明黑色背景的文字
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (0, 255, 0)  # 綠色

    (w1, h1), baseline1 = cv2.getTextSize(text1, font, font_scale, thickness)
    (w2, h2), baseline2 = cv2.getTextSize(text2, font, font_scale, thickness)

    bg_w = max(w1, w2) + 20
    bg_h = h1 + h2 + 40

    x, y = position

    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - h1 - 10), (x + bg_w, y + h2 + 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, text1, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (x, y + h1 + 15), font, font_scale, color, thickness, cv2.LINE_AA)

    return img


# ==========================================
# 3. 主程式：批次處理與視覺化輸出
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Calculate and compare MUSIQ scores for 2-Region Original, GT and Predicted Alpha")
    parser.add_argument("--csv", type=str, default="Z:/HCA/2R_Model/test_2region/2region_hca_all_results.csv",
                        help="Path to validation/test CSV")
    parser.add_argument("--raw-csv", type=str, default="./alpha_data_2region.csv",
                        help="Optional fallback CSV to merge original_musiq/masks if missing")
    parser.add_argument("--img-dir", type=str, default='E:/2RLabeling_Images',
                        help="Directory containing the original images")
    parser.add_argument("--mask-dir", type=str, default='E:/2RLabeling_Masks',
                        help="Directory containing the saliency masks")
    parser.add_argument("--out-dir", type=str, default="Z:/HCA/2R_Model/test_2region/hca_2region_all_musiq_eval_output",
                        help="Output directory for results and plots")
    parser.add_argument("--save-pairs", action="store_true", default=True, help="Save side-by-side comparison images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)

    pairs_dir = out_dir / "comparison_pairs_2region"
    pairs_dir.mkdir(exist_ok=True)

    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    # 支援 main_2region.py 產出的 In_Center_GT / Out_Center_GT / In_Predicted_Alpha / Out_Predicted_Alpha
    col_mapping = {
        'Image': 'Image', 'image': 'Image',
        'In_Center_GT': 'Alpha_GT_In', 'in_alpha_center': 'Alpha_GT_In',
        'acceptable_alpha_in_center': 'Alpha_GT_In', 'Center_GT_In': 'Alpha_GT_In', 'best_alpha_in': 'Alpha_GT_In',
        'Out_Center_GT': 'Alpha_GT_Out', 'out_alpha_center': 'Alpha_GT_Out',
        'acceptable_alpha_out_center': 'Alpha_GT_Out', 'Center_GT_Out': 'Alpha_GT_Out', 'best_alpha_out': 'Alpha_GT_Out',
        'In_Predicted_Alpha': 'Alpha_Pred_In', 'Predicted_Alpha_In': 'Alpha_Pred_In',
        'Out_Predicted_Alpha': 'Alpha_Pred_Out', 'Predicted_Alpha_Out': 'Alpha_Pred_Out'
    }

    for raw_col, standard_col in col_mapping.items():
        if raw_col in df.columns and standard_col not in df.columns:
            df[standard_col] = df[raw_col]

    required_cols = ['Image', 'Alpha_GT_In', 'Alpha_GT_Out', 'Alpha_Pred_In', 'Alpha_Pred_Out']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV or mapping failed: {col}. Available columns: {list(df.columns)}")

    # 嘗試從 fallback-csv 合併 original_musiq
    if 'original_musiq' not in df.columns and Path(args.raw_csv).exists():
        raw_df = pd.read_csv(args.raw_csv)
        join_key = 'image' if 'image' in raw_df.columns else ('Image' if 'Image' in raw_df.columns else None)
        if join_key and 'original_musiq' in raw_df.columns:
            df = pd.merge(
                df,
                raw_df[[join_key, 'original_musiq']].drop_duplicates(subset=[join_key]),
                left_on='Image', right_on=join_key, how='left'
            )
            if join_key != 'Image' and join_key in df.columns:
                df.drop(columns=[join_key], inplace=True)

    results = []

    print(f"Processing {len(df)} images under 2-Region framework...")
    for idx, row in df.iterrows():
        img_name = row['Image']
        alpha_gt_in = float(row['Alpha_GT_In'])
        alpha_gt_out = float(row['Alpha_GT_Out'])
        alpha_pred_in = float(row['Alpha_Pred_In'])
        alpha_pred_out = float(row['Alpha_Pred_Out'])

        img_path = img_dir / img_name
        mask_path = mask_dir / img_name
        if not mask_path.exists():
            mask_path = mask_dir / f"{Path(img_name).stem}.png"

        if not img_path.exists():
            print(f"⚠ Warning: Image not found, skipping - {img_path}")
            continue
        if not mask_path.exists():
            print(f"⚠ Warning: Mask file not found, skipping - {mask_path}")
            continue

        orig_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if orig_img is None or mask_img is None:
            print(f"⚠ Warning: Failed to load image or mask, skipping - {img_name}")
            continue

        if mask_img.shape[:2] != orig_img.shape[:2]:
            mask_img = cv2.resize(mask_img, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if 'original_musiq' in row and pd.notna(row['original_musiq']):
            orig_musiq = float(row['original_musiq'])
        else:
            orig_musiq = get_musiq_score(orig_img)

        # --- 2-Region GT 與 預測 影像計算 ---
        img_gt = apply_2region_usm(orig_img, mask_img, alpha_gt_in, alpha_gt_out)
        musiq_gt = get_musiq_score(img_gt)

        img_pred = apply_2region_usm(orig_img, mask_img, alpha_pred_in, alpha_pred_out)
        musiq_pred = get_musiq_score(img_pred)

        diff = musiq_pred - musiq_gt
        gain_pred_vs_orig = musiq_pred - orig_musiq
        gain_gt_vs_orig = musiq_gt - orig_musiq

        results.append({
            'Image': img_name,
            'original_musiq': orig_musiq,
            'Alpha_GT_In': alpha_gt_in,
            'Alpha_GT_Out': alpha_gt_out,
            'Alpha_Pred_In': alpha_pred_in,
            'Alpha_Pred_Out': alpha_pred_out,
            'MUSIQ_GT': musiq_gt,
            'MUSIQ_Pred': musiq_pred,
            'MUSIQ_Diff': diff,
            'MUSIQ_Gain_Pred_vs_Orig': gain_pred_vs_orig,
            'MUSIQ_Gain_GT_vs_Orig': gain_gt_vs_orig
        })

        # --- 繪製文字與拼接 2-Region 對比圖 ---
        if args.save_pairs:
            text_orig_1 = "Original Image"
            text_orig_2 = f"MUSIQ: {orig_musiq:.2f}"
            img_orig_labeled = draw_text_with_background(orig_img.copy(), text_orig_1, text_orig_2)

            text_gt_1 = f"GT In/Out: {alpha_gt_in:.1f}/{alpha_gt_out:.1f}"
            text_gt_2 = f"MUSIQ: {musiq_gt:.2f}"
            img_gt_labeled = draw_text_with_background(img_gt.copy(), text_gt_1, text_gt_2)

            text_pred_1 = f"Pred In/Out: {alpha_pred_in:.1f}/{alpha_pred_out:.1f}"
            text_pred_2 = f"MUSIQ: {musiq_pred:.2f}"
            img_pred_labeled = draw_text_with_background(img_pred.copy(), text_pred_1, text_pred_2)

            combined_img = np.hstack((img_orig_labeled, img_gt_labeled, img_pred_labeled))

            out_pair_path = pairs_dir / f"compare_2region_{img_name}"
            cv2.imwrite(str(out_pair_path), combined_img)

        print(f"[{idx + 1}/{len(df)}] {img_name} | "
              f"GT In/Out: {alpha_gt_in:.1f}/{alpha_gt_out:.1f} | "
              f"Pred In/Out: {alpha_pred_in:.1f}/{alpha_pred_out:.1f} | "
              f"Orig Score: {orig_musiq:.2f} | GT Score: {musiq_gt:.2f} | Pred Score: {musiq_pred:.2f} | "
              f"Diff(P-G): {diff:+.2f} | Gain(P-O): {gain_pred_vs_orig:+.2f}")

    # ==========================================
    # 4. 產出 2-Region 評估報告與圖表
    # ==========================================
    if not results:
        print("No images were processed successfully.")
        return

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "2region_musiq_comparison_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")
    if args.save_pairs:
        print(f"✅ 2-Region Comparison images saved to {pairs_dir}")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(res_df['MUSIQ_GT'], res_df['MUSIQ_Pred'], alpha=0.6, color='royalblue', s=20,
                label='2-Region Pred vs GT')
    plt.scatter(res_df['original_musiq'], res_df['MUSIQ_Pred'], alpha=0.5, color='darkorange', s=20,
                label='2-Region Pred vs Original')
    min_val = min(res_df['original_musiq'].min(), res_df['MUSIQ_GT'].min(), res_df['MUSIQ_Pred'].min()) - 1
    max_val = max(res_df['original_musiq'].max(), res_df['MUSIQ_GT'].max(), res_df['MUSIQ_Pred'].max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y = x (Perfect Match)')
    plt.title('2-Region HCA MUSIQ Scores Comparison', fontsize=12, fontweight='bold')
    plt.xlabel('Reference MUSIQ Score (GT / Original)', fontsize=10)
    plt.ylabel('MUSIQ Score (Predicted)', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(res_df['MUSIQ_Diff'], bins=30, alpha=0.7, color='dodgerblue', edgecolor='black',
             label='2-Region (Pred - GT)')
    plt.hist(res_df['MUSIQ_Gain_Pred_vs_Orig'], bins=30, alpha=0.5, color='forestgreen', edgecolor='black',
             label='2-Region (Pred - Original)')
    plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5)
    plt.title('2-Region MUSIQ Score Improvements Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('MUSIQ Difference / Gain', fontsize=10)
    plt.ylabel('Image count (Frequency)', fontsize=10)
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.legend()

    out_plot = out_dir / "2region_musiq_difference_analysis.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    print(f"✅ Difference Analysis Plot saved to {out_plot}")

    print("\n📊 --- 2-Region MUSIQ 統計摘要 ---")
    print(f"原始影像平均 MUSIQ (original_musiq): {res_df['original_musiq'].mean():.4f}")
    print(f"GT 雙區增強平均 MUSIQ (MUSIQ_GT):      {res_df['MUSIQ_GT'].mean():.4f}")
    print(f"預測雙區增強平均 MUSIQ (MUSIQ_Pred):    {res_df['MUSIQ_Pred'].mean():.4f}")
    print(
        f"平均差異 (Pred - GT):                 {res_df['MUSIQ_Diff'].mean():.4f} (Std: {res_df['MUSIQ_Diff'].std():.4f})")
    print(f"平均提升 (Pred - Original):           {res_df['MUSIQ_Gain_Pred_vs_Orig'].mean():+.4f}")
    print(f"預測分數 優於或等於 原圖的比例:         {(res_df['MUSIQ_Gain_Pred_vs_Orig'] >= 0.0).mean():.2%}")
    print(f"預測分數 貼齊或超越 GT的比例 (Diff>=-0.15): {(res_df['MUSIQ_Diff'] >= -0.15).mean():.2%}")


if __name__ == "__main__":
    main()