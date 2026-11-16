# prepare_csv.py
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_IN = "E:/1DS_Result_New/1R_ALL_Best_Summary_New_with_features.csv"
CSV_OUT = "alpha_data.csv"

df = pd.read_csv(CSV_IN)
df = df.dropna(subset=['acceptable_alpha_min', 'acceptable_alpha_max']).copy()
df['acceptable_alpha_min'] = df['acceptable_alpha_min'].round(1)
df['acceptable_alpha_max'] = df['acceptable_alpha_max'].round(1)
df['acceptable_alpha_center'] = ((df['acceptable_alpha_min'] + df['acceptable_alpha_max']) / 2.0).round(2)
df['best_alpha'] = df['best_alpha'].fillna(df['acceptable_alpha_min']).round(1)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['acceptable_alpha_min'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['acceptable_alpha_min'])

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
full_df.to_csv(CSV_OUT, index=False)
print(f"✅ 已產生 N={len(full_df)} 的 {CSV_OUT} (Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)})")