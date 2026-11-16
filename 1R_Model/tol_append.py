import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# 參數設定
# ============================================================================
# 主 CSV 檔案路徑
SUMMARY_CSV = 'E:/1DS_Result_New/1R_ALL_Best_Summary_New_with_features.csv'
# 修正後的輸出檔案路徑
OUTPUT_CSV = 'E:/1DS_Result_New/1R_ALL_Best_Summary_New_with_features_v2.csv'
# 存放各影像 trial 紀錄的資料夾 (請根據實際路徑修改，目前預設為同一層資料夾)
TRIALS_DIR = Path('E:/1DS_Result_New/per_image')

# 容許的 MUSIQ 下降幅度
MUSIQ_TOLERANCE = 0.1


def main():
    print("=" * 60)
    print("📊 開始更新 Best Alpha 與 Acceptable Range")
    print("=" * 60)

    # 1. 讀取 Summary CSV
    try:
        df = pd.read_csv(SUMMARY_CSV)
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {SUMMARY_CSV}")
        return

    # 確認影像檔名欄位
    img_col = next((col for col in ['image_name', 'image', 'filename', 'file_name'] if col in df.columns), None)
    if img_col is None:
        raise ValueError("找不到影像名稱欄位 (image, image_name, filename)。請確認 CSV 欄位名稱。")

    # 2. 遍歷 Trials Folder 並計算
    processed_count = 0
    missing_count = 0

    for idx, row in df.iterrows():
        img_name = str(row[img_col])
        stem = Path(img_name).stem

        # 尋找對應的 trial csv
        trial_file = TRIALS_DIR / f"{stem}_trials.csv"

        if not trial_file.exists():
            trial_file = TRIALS_DIR / f"{stem}.csv"

        if trial_file.exists():
            try:
                df_trial = pd.read_csv(trial_file)

                # 確保必要欄位存在
                alpha_col = 'alpha' if 'alpha' in df_trial.columns else 'Alpha'
                musiq_col = 'after_musiq' if 'after_musiq' in df_trial.columns else 'MUSIQ'

                # 尋找最高的 after_musiq 以及對應的 best_alpha
                best_idx = df_trial[musiq_col].idxmax()
                best_musiq = df_trial.loc[best_idx, musiq_col]
                best_alpha = df_trial.loc[best_idx, alpha_col]

                # 計算 Threshold 與 Range
                threshold = best_musiq - MUSIQ_TOLERANCE
                valid_trials = df_trial[df_trial[musiq_col] >= threshold]

                if not valid_trials.empty:
                    alpha_min = valid_trials[alpha_col].min()
                    alpha_max = valid_trials[alpha_col].max()

                    # 更新 DataFrame
                    df.at[idx, 'best_alpha'] = best_alpha
                    df.at[idx, 'acceptable_alpha_min'] = alpha_min
                    df.at[idx, 'acceptable_alpha_max'] = alpha_max
                    df.at[idx, 'acceptable_alpha_center'] = (alpha_min + alpha_max) / 2.0
                    df.at[idx, 'acceptable_alpha_width'] = alpha_max - alpha_min

                    processed_count += 1
            except Exception as e:
                print(f"處理 {trial_file.name} 時發生錯誤: {e}")
        else:
            missing_count += 1

    print(f"✅ 成功比對並更新: {processed_count} 筆")
    if missing_count > 0:
        print(f"⚠️ 找不到對應 Trials CSV: {missing_count} 筆")

    # 3. 輸出更新後的 CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n💾 結果已儲存至:\n{OUTPUT_CSV}")

    # ========================================================================
    # 4. 統計分析與視覺化
    # ========================================================================
    valid_widths = df['acceptable_alpha_width'].dropna()

    if not valid_widths.empty:
        mean_width = valid_widths.mean()
        median_width = valid_widths.median()
        max_width = valid_widths.max()

        print("\n【Acceptable Alpha Width 統計資料】")
        print("-" * 35)
        print(f"Mean   : {mean_width:.4f}")
        print(f"Median : {median_width:.4f}")
        print(f"Max    : {max_width:.4f}")
        print("-" * 35)

        # 繪製 Histogram
        plt.figure(figsize=(9, 6))
        plt.hist(valid_widths, bins=30, color='teal', alpha=0.75, edgecolor='black')
        plt.axvline(mean_width, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_width:.2f}')
        plt.axvline(median_width, color='orange', linestyle='dashed', linewidth=2, label=f'Median: {median_width:.2f}')

        plt.title('Distribution of Acceptable Alpha Width (Tolerance = 0.1)', fontsize=14)
        plt.xlabel('Acceptable Alpha Width (Max - Min)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle=':', alpha=0.7)

        output_plot = Path(OUTPUT_CSV).parent / 'acceptable_alpha_width_histogram.png'
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n📈 Histogram 已儲存至:\n{output_plot}")
    else:
        print("\n❌ 沒有足夠的有效資料來產生統計與圖表。")


if __name__ == '__main__':
    main()