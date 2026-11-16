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
# 2. 影像處理管線與文字繪製函數
# ==========================================
def apply_usm(orig_bgr_uint8: np.ndarray, alpha: float) -> np.ndarray:
    denoised = cv2.bilateralFilter(orig_bgr_uint8, d=5, sigmaColor=20, sigmaSpace=20)
    blurred = cv2.GaussianBlur(denoised, (5, 5), sigmaX=1.1, sigmaY=1.1)

    den_f32 = denoised.astype(np.float32)
    blur_f32 = blurred.astype(np.float32)

    final_f32 = den_f32 + alpha * (den_f32 - blur_f32)
    final = np.clip(final_f32, 0.0, 255.0).astype(np.uint8)

    return final


def draw_text_with_background(img: np.ndarray, text1: str, text2: str, position=(20, 50)):
    """
    在影像上繪製帶有半透明黑色背景的文字，確保在任何亮度的圖片上都能清晰辨識
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 255, 0)  # 綠色字體

    # 計算第一行文字大小
    (w1, h1), baseline1 = cv2.getTextSize(text1, font, font_scale, thickness)
    # 計算第二行文字大小
    (w2, h2), baseline2 = cv2.getTextSize(text2, font, font_scale, thickness)

    # 背景框的寬度與高度
    bg_w = max(w1, w2) + 20
    bg_h = h1 + h2 + 40

    x, y = position

    # 建立一個與原圖大小相同的黑色遮罩
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - h1 - 10), (x + bg_w, y + h2 + 20), (0, 0, 0), -1)

    # 混合原圖與黑色遮罩，產生半透明背景 (alpha=0.6)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # 繪製文字
    cv2.putText(img, text1, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, text2, (x, y + h1 + 15), font, font_scale, color, thickness, cv2.LINE_AA)

    return img


# ==========================================
# 3. 主程式：批次處理與視覺化輸出
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Calculate and compare MUSIQ scores for Original, GT and Predicted Alpha")
    parser.add_argument("--csv", type=str, default="Z:/HCA/1R_Model/test/hca_base_all_results.csv", help="Path to validation/test CSV")
    parser.add_argument("--raw-csv", type=str, default="E:/1DS_Result_New/1R_ALL_Best_Summary_New_with_features.csv",
                        help="Optional fallback CSV to merge original_musiq if missing in --csv")
    parser.add_argument("--img-dir", type=str, default='E:/1R_Labeling_Images_New',
                        help="Directory containing the original images")
    parser.add_argument("--out-dir", type=str, default="Z:/HCA/test/hca_base_all_musiq_eval_output",
                        help="Output directory for results and plots")
    parser.add_argument("--save-pairs", action="store_true",default=False, help="Save side-by-side comparison images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(args.img_dir)

    # 建立存放對比圖的資料夾
    pairs_dir = out_dir / "comparison_pairs"
    pairs_dir.mkdir(exist_ok=True)

    print(f"Reading CSV: {args.csv}")
    df = pd.read_csv(args.csv)

    required_cols = ['Image', 'Center_GT', 'Predicted_Alpha']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    # 若輸入的 CSV 尚未包含 original_musiq，嘗試從 raw_csv 自動合併
    if 'original_musiq' not in df.columns and Path(args.raw_csv).exists():
        raw_df = pd.read_csv(args.raw_csv)
        if 'image' in raw_df.columns and 'original_musiq' in raw_df.columns:
            df = pd.merge(
                df,
                raw_df[['image', 'original_musiq']].drop_duplicates(subset=['image']),
                left_on='Image', right_on='image', how='left'
            )
            if 'image' in df.columns:
                df.drop(columns=['image'], inplace=True)

    results = []

    print(f"Processing {len(df)} images...")
    for idx, row in df.iterrows():
        img_name = row['Image']
        alpha_gt = float(row['Center_GT'])
        alpha_pred = float(row['Predicted_Alpha'])

        img_path = img_dir / img_name
        if not img_path.exists():
            print(f"⚠ Warning: Image not found, skipping - {img_path}")
            continue

        orig_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if orig_img is None:
            print(f"⚠ Warning: Failed to load image, skipping - {img_path}")
            continue

        # --- 取得或計算原始影像的 original_musiq 分數 ---
        if 'original_musiq' in row and pd.notna(row['original_musiq']):
            orig_musiq = float(row['original_musiq'])
        else:
            orig_musiq = get_musiq_score(orig_img)

        # --- 計算 GT 影像 ---
        img_gt = apply_usm(orig_img, alpha_gt)
        musiq_gt = get_musiq_score(img_gt)

        # --- 計算 Pred 影像 ---
        img_pred = apply_usm(orig_img, alpha_pred)
        musiq_pred = get_musiq_score(img_pred)

        diff = musiq_pred - musiq_gt
        gain_pred_vs_orig = musiq_pred - orig_musiq
        gain_gt_vs_orig = musiq_gt - orig_musiq

        results.append({
            'Image': img_name,
            'original_musiq': orig_musiq,
            'Alpha_GT': alpha_gt,
            'Alpha_Pred': alpha_pred,
            'MUSIQ_GT': musiq_gt,
            'MUSIQ_Pred': musiq_pred,
            'MUSIQ_Diff': diff,
            'MUSIQ_Gain_Pred_vs_Orig': gain_pred_vs_orig,
            'MUSIQ_Gain_GT_vs_Orig': gain_gt_vs_orig
        })

        # --- 繪製文字與拼接對比圖 (Left: Orig, Mid: GT, Right: Pred) ---
        text_orig_1 = "Original Image"
        text_orig_2 = f"MUSIQ: {orig_musiq:.2f}"
        img_orig_labeled = draw_text_with_background(orig_img.copy(), text_orig_1, text_orig_2)

        text_gt_1 = f"GT Alpha: {alpha_gt:.2f}"
        text_gt_2 = f"MUSIQ: {musiq_gt:.2f}"
        img_gt_labeled = draw_text_with_background(img_gt.copy(), text_gt_1, text_gt_2)

        text_pred_1 = f"Pred Alpha: {alpha_pred:.2f}"
        text_pred_2 = f"MUSIQ: {musiq_pred:.2f}"
        #img_pred_labeled = draw_text_with_background(img_pred.copy(), text_pred_1, text_pred_2)

        # 水平拼接三張影像 (Left: Original, Middle: GT, Right: Pred)
        #combined_img = np.hstack((img_orig_labeled, img_gt_labeled, img_pred_labeled))

        # 儲存對比圖
        #out_pair_path = pairs_dir / f"compare_{img_name}"
        #cv2.imwrite(str(out_pair_path), combined_img)

        print(f"[{idx + 1}/{len(df)}] {img_name} | Orig: {orig_musiq:.2f} | GT: {musiq_gt:.2f} | Pred: {musiq_pred:.2f} | Diff(Pred-GT): {diff:+.2f} | Gain(Pred-Orig): {gain_pred_vs_orig:+.2f}")

    # ==========================================
    # 4. 產出報告與圖表
    # ==========================================
    if not results:
        print("No images were processed successfully.")
        return

    res_df = pd.DataFrame(results)
    out_csv = out_dir / "musiq_comparison_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")
    print(f"✅ Comparison images saved to {pairs_dir}")

    # 繪製圖表
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(res_df['MUSIQ_GT'], res_df['MUSIQ_Pred'], alpha=0.6, color='b', s=20, label='Pred vs GT')
    plt.scatter(res_df['original_musiq'], res_df['MUSIQ_Pred'], alpha=0.5, color='g', s=20, label='Pred vs Original')
    min_val = min(res_df['original_musiq'].min(), res_df['MUSIQ_GT'].min(), res_df['MUSIQ_Pred'].min()) - 1
    max_val = max(res_df['original_musiq'].max(), res_df['MUSIQ_GT'].max(), res_df['MUSIQ_Pred'].max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='y = x')
    plt.title('MUSIQ Scores Comparison')
    plt.xlabel('Reference MUSIQ Score (GT / Original)')
    plt.ylabel('MUSIQ Score (Predicted)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(res_df['MUSIQ_Diff'], bins=30, alpha=0.7, color='coral', edgecolor='black', label='Pred - GT')
    plt.hist(res_df['MUSIQ_Gain_Pred_vs_Orig'], bins=30, alpha=0.5, color='mediumseagreen', edgecolor='black', label='Pred - Original')
    plt.axvline(0, color='k', linestyle='dashed', linewidth=1.5)
    plt.title('Distribution of MUSIQ Score Differences')
    plt.xlabel('MUSIQ Difference')
    plt.ylabel('Frequency')
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.legend()

    out_plot = out_dir / "musiq_difference_analysis.png"
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    print(f"✅ Plots saved to {out_plot}")

    # 統計摘要
    print("\n📊 --- MUSIQ 統計摘要 ---")
    print(f"原始影像平均 MUSIQ (original_musiq): {res_df['original_musiq'].mean():.4f}")
    print(f"GT 增強後平均 MUSIQ (MUSIQ_GT):      {res_df['MUSIQ_GT'].mean():.4f}")
    print(f"預測增強後平均 MUSIQ (MUSIQ_Pred):    {res_df['MUSIQ_Pred'].mean():.4f}")
    print(f"平均差異 (Pred - GT):                 {res_df['MUSIQ_Diff'].mean():.4f} (Std: {res_df['MUSIQ_Diff'].std():.4f})")
    print(f"平均提升 (Pred - Original):           {res_df['MUSIQ_Gain_Pred_vs_Orig'].mean():+.4f}")
    print(f"預測分數優於或等於原圖的比例:         {(res_df['MUSIQ_Gain_Pred_vs_Orig'] >= 0.0).mean():.2%}")
    print(f"預測分數優於或等於 GT 的比例:         {(res_df['MUSIQ_Diff'] >= -0.1).mean():.2%}")


if __name__ == "__main__":
    main()