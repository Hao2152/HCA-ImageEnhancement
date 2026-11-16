import os

os.environ["KMP_WARNINGS"] = "FALSE"
import random
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import IMDBWIKI, extract_tabular_matrices
from utils import AverageMeter
from resnet_cls_mh import resnet50_cls_mh
from mh_utils import level_split, get_mh_weight
from cls_funcs import Count2Class
from eval_metrics import shot_interval_metrics
from ranksim import batchwise_ranking_regularizer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# ================= LDS / FDS / RankSim 消融實驗參數 =================
parser.add_argument('--lds', action='store_true', default=False, help='enable LDS (Label Distribution Smoothing)')
parser.add_argument('--lds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'])
parser.add_argument('--lds_ks', type=int, default=5)
parser.add_argument('--lds_sigma', type=float, default=1.0)
parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'])

parser.add_argument('--fds', action='store_true', default=False, help='enable FDS (Feature Distribution Smoothing)')
parser.add_argument('--fds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'])
parser.add_argument('--fds_ks', type=int, default=5)
parser.add_argument('--fds_sigma', type=float, default=1.0)
parser.add_argument('--start_update', type=int, default=0)
parser.add_argument('--start_smooth', type=int, default=1)
parser.add_argument('--bucket_num', type=int, default=20)
parser.add_argument('--bucket_start', type=int, default=0)
parser.add_argument('--fds_mmt', type=float, default=0.9)

parser.add_argument('--ranksim', action='store_true', default=False, help='shortcut flag to enable RankSim')
parser.add_argument('--regularization_weight', type=float, default=0.0, help='RankSim loss weight gamma')
parser.add_argument('--interpolation_lambda', type=float, default=2.0, help='RankSim interpolation strength lambda')

# ================= 訓練與模型超參數 (2-Region 雙參數預測) =================
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--tol', type=float, default=0.3, help='interval tolerance (+/- 0.3)')
parser.add_argument('--csv_path', type=str, default='./alpha_data_2region.csv')
parser.add_argument('--raw_csv_path', type=str, default='E:/2DS_Result/ALL_Best_Summary_all.csv')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--feat_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--temp', type=float, default=0.35)
parser.add_argument('--save_best', action='store_true', default=True)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--evaluate_only', action='store_true', default=False)
parser.add_argument('--plot_head', type=str, default='combmul', choices=['combadd', 'combmul', 'adjcls', 'singlecls'])
parser.add_argument('--save_dir', type=str, default='./checkpoints_2region')
parser.add_argument('--log_dir', type=str, default='analy_log_2region')
parser.add_argument('--log_name', type=str, default='2region_hca.log')
parser.add_argument('--head_num', type=int, default=4)
args, unknown = parser.parse_known_args()

if args.ranksim and args.regularization_weight <= 0:
    args.regularization_weight = 1.0


def plot_2region_regression_results(df_results, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Foreground (best_alpha_in) - sorted by In_Center_GT
    df_in_sorted = df_results.sort_values('In_Center_GT').reset_index(drop=True)
    ax1 = axes[0]
    ax1.fill_between(df_in_sorted.index, df_in_sorted['In_Min_Alpha'], df_in_sorted['In_Max_Alpha'],
                     color='gray', alpha=0.3, label='Acceptable Range')
    ax1.plot(df_in_sorted.index, df_in_sorted['In_Center_GT'], 'k--', label='GT (Center)', linewidth=1.2)
    correct_in = df_in_sorted[df_in_sorted['In_Is_In_Range'] == True]
    incorrect_in = df_in_sorted[df_in_sorted['In_Is_In_Range'] == False]
    ax1.scatter(correct_in.index, correct_in['In_Predicted_Alpha'], color='green', s=15, label='Correct (In Range)',
                zorder=5)
    ax1.scatter(incorrect_in.index, incorrect_in['In_Predicted_Alpha'], color='red', s=15,
                label='Incorrect (Out of Range)', zorder=5)
    ax1.set_xlabel('Samples (Sorted by In GT Center)')
    ax1.set_ylabel('Foreground Alpha (best_alpha_in)')
    ax1.set_title(f'{title} - Foreground (In)')
    ax1.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    ax1.set_ylim(0.9, 2.1)

    # Right: Background (best_alpha_out) - sorted by Out_Center_GT
    df_out_sorted = df_results.sort_values('Out_Center_GT').reset_index(drop=True)
    ax2 = axes[1]
    ax2.fill_between(df_out_sorted.index, df_out_sorted['Out_Min_Alpha'], df_out_sorted['Out_Max_Alpha'],
                     color='gray', alpha=0.3, label='Acceptable Range')
    ax2.plot(df_out_sorted.index, df_out_sorted['Out_Center_GT'], 'k--', label='GT (Center)', linewidth=1.2)
    correct_out = df_out_sorted[df_out_sorted['Out_Is_In_Range'] == True]
    incorrect_out = df_out_sorted[df_out_sorted['Out_Is_In_Range'] == False]
    ax2.scatter(correct_out.index, correct_out['Out_Predicted_Alpha'], color='green', s=15, label='Correct (In Range)',
                zorder=5)
    ax2.scatter(incorrect_out.index, incorrect_out['Out_Predicted_Alpha'], color='red', s=15,
                label='Incorrect (Out of Range)', zorder=5)
    ax2.set_xlabel('Samples (Sorted by Out GT Center)')
    ax2.set_ylabel('Background Alpha (best_alpha_out)')
    ax2.set_title(f'{title} - Background (Out)')
    ax2.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    ax2.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


@torch.no_grad()
def collect_2region_predictions_df(loader, df_split, model_in, model_out, head_type='combmul'):
    model_in.eval()
    model_out.eval()

    in_preds, in_center, in_min, in_max = [], [], [], []
    out_preds, out_center, out_min, out_max = [], [], [], []

    for tab_feat_in, tab_feat_out, targets_in, targets_out, _ in loader:
        tab_feat_in = tab_feat_in.cuda(non_blocking=True)
        tab_feat_out = tab_feat_out.cuda(non_blocking=True)

        # Forward Foreground Branch
        allpre_in = model_in(tab_feat_in)
        pred_dict_in = model_in.module.hier_combine_continuous(allpre_in, temp=args.temp)
        pred_val_in = pred_dict_in[head_type]

        # Forward Background Branch
        allpre_out = model_out(tab_feat_out)
        pred_dict_out = model_out.module.hier_combine_continuous(allpre_out, temp=args.temp)
        pred_val_out = pred_dict_out[head_type]

        in_preds.extend(pred_val_in.cpu().view(-1).numpy())
        in_center.extend(targets_in[:, 0].numpy())
        in_min.extend(targets_in[:, 3].numpy())
        in_max.extend(targets_in[:, 4].numpy())

        out_preds.extend(pred_val_out.cpu().view(-1).numpy())
        out_center.extend(targets_out[:, 0].numpy())
        out_min.extend(targets_out[:, 3].numpy())
        out_max.extend(targets_out[:, 4].numpy())

    if 'original_musiq' in df_split.columns:
        orig_musiq_vals = np.round(df_split['original_musiq'].values.astype(float), 4)
    else:
        orig_musiq_vals = np.full(len(df_split), np.nan)

    split_vals = df_split['split'].values if 'split' in df_split.columns else ['unknown'] * len(df_split)

    results_df = pd.DataFrame({
        'Image': df_split['image'].values,
        'Split': split_vals,
        'original_musiq': orig_musiq_vals,

        'In_Center_GT': np.round(in_center, 1),
        'In_Min_Alpha': np.round(in_min, 1),
        'In_Max_Alpha': np.round(in_max, 1),
        'In_Predicted_Alpha': np.clip(np.round(in_preds, 1), 1.0, 2.0),

        'Out_Center_GT': np.round(out_center, 1),
        'Out_Min_Alpha': np.round(out_min, 1),
        'Out_Max_Alpha': np.round(out_max, 1),
        'Out_Predicted_Alpha': np.clip(np.round(out_preds, 1), 0.1, 1.0),
    })

    results_df['In_Is_In_Range'] = (
            (results_df['In_Predicted_Alpha'] >= results_df['In_Min_Alpha'] - 1e-4) &
            (results_df['In_Predicted_Alpha'] <= results_df['In_Max_Alpha'] + 1e-4)
    )
    results_df['Out_Is_In_Range'] = (
            (results_df['Out_Predicted_Alpha'] >= results_df['Out_Min_Alpha'] - 1e-4) &
            (results_df['Out_Predicted_Alpha'] <= results_df['Out_Max_Alpha'] + 1e-4)
    )
    results_df = results_df.sort_values('In_Center_GT').reset_index(drop=True)
    return results_df


def main():
    max_cls_in = 10  # 1.0 to 2.0 (11 bins)
    max_cls_out = 9  # 0.1 to 1.0 (10 bins)

    global_class_indice_in = np.round(np.linspace(1.0, 1.9, 10) + 0.05, 4)
    global_class2count_in = np.round(np.linspace(1.0, 2.0, 11), 4)

    global_class_indice_out = np.round(np.linspace(0.1, 0.9, 9) + 0.05, 4)
    global_class2count_out = np.round(np.linspace(0.1, 1.0, 10), 4)

    ablation_tag = []
    if args.reweight != 'none' and not args.lds:
        ablation_tag.append(args.reweight)
    if args.lds:
        ablation_tag.append('lds')
    if args.fds:
        ablation_tag.append('fds')
    if args.regularization_weight > 0:
        ablation_tag.append('ranksim')
    exp_tag = '_'.join(ablation_tag) if len(ablation_tag) > 0 else 'base'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('./test_2region', exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f"2region_{exp_tag}_{args.log_name}")),
            logging.StreamHandler()
        ])
    print = logging.info
    print(f"=====> 2-Region Dual-Branch HCA Feature Slicing: [{exp_tag.upper()}] | Args: {args}")

    df = pd.read_csv(args.csv_path)

    for col in ['best_alpha_in', 'best_alpha_out']:
        if col not in df.columns:
            raw_df = pd.read_csv(args.raw_csv_path)
            if col in raw_df.columns:
                df = pd.merge(df, raw_df[['image', col]].drop_duplicates(subset=['image']), on='image', how='left')

    df = df.dropna(subset=['best_alpha_in', 'best_alpha_out']).reset_index(drop=True)

    if 'in_alpha_min' in df.columns and 'in_alpha_max' in df.columns:
        df['in_alpha_min'] = df['in_alpha_min'].round(1)
        df['in_alpha_max'] = df['in_alpha_max'].round(1)
        df['in_alpha_center'] = ((df['in_alpha_min'] + df['in_alpha_max']) / 2.0).round(2)
    else:
        df['in_alpha_min'] = (df['best_alpha_in'] - args.tol).clip(1.0, 2.0).round(1)
        df['in_alpha_max'] = (df['best_alpha_in'] + args.tol).clip(1.0, 2.0).round(1)
        df['in_alpha_center'] = df['best_alpha_in'].round(2)

    if 'out_alpha_min' in df.columns and 'out_alpha_max' in df.columns:
        df['out_alpha_min'] = df['out_alpha_min'].round(1)
        df['out_alpha_max'] = df['out_alpha_max'].round(1)
        df['out_alpha_center'] = ((df['out_alpha_min'] + df['out_alpha_max']) / 2.0).round(2)
    else:
        df['out_alpha_min'] = (df['best_alpha_out'] - args.tol).clip(0.1, 1.0).round(1)
        df['out_alpha_max'] = (df['best_alpha_out'] + args.tol).clip(0.1, 1.0).round(1)
        df['out_alpha_center'] = df['best_alpha_out'].round(2)

    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    df_val = df[df['split'] == 'val'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)

    train_orig_y_min_in = df_train['in_alpha_min'].values
    train_orig_y_min_out = df_train['out_alpha_min'].values

    scaler_in = StandardScaler()
    scaler_out = StandardScaler()

    X_tr_in, X_tr_out = extract_tabular_matrices(df_train)
    X_va_in, X_va_out = extract_tabular_matrices(df_val)
    X_te_in, X_te_out = extract_tabular_matrices(df_test)

    X_tr_in = scaler_in.fit_transform(X_tr_in)
    X_va_in = scaler_in.transform(X_va_in)
    X_te_in = scaler_in.transform(X_te_in)

    X_tr_out = scaler_out.fit_transform(X_tr_out)
    X_va_out = scaler_out.transform(X_va_out)
    X_te_out = scaler_out.transform(X_te_out)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dataset = IMDBWIKI(df_train, X_tr_in, X_tr_out, tol=args.tol, split='train',
                             reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks,
                             lds_sigma=args.lds_sigma, is_2region=True
                             )
    train_eval_dataset = IMDBWIKI(df_train, X_tr_in, X_tr_out, tol=args.tol, split='val')
    val_dataset = IMDBWIKI(df_val, X_va_in, X_va_out, tol=args.tol, split='val')
    test_dataset = IMDBWIKI(df_test, X_te_in, X_te_out, tol=args.tol, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Foreground (In) HCA setup ---
    global_count_dict_in = {x: 0 for x in range(max_cls_in + 1)}
    for label in train_orig_y_min_in:
        idx = int(np.round((float(label) - 1.0) * 10))
        global_count_dict_in[min(max_cls_in, max(0, idx))] += 1
    global_count_dict_sqrt_in = {k: np.sqrt(v) for k, v in global_count_dict_in.items()}

    _, sindice_idx_in, sclass2count_idx_in = level_split(
        [0, max_cls_in + 0.01], global_count_dict_sqrt_in, args.head_num, if_age=True, ob_vmax=max_cls_in
    )
    sweight_in = get_mh_weight(global_count_dict_in, sindice_idx_in)
    sindice_in = [[round(1.0 + (x * 0.1), 4) for x in idx_list] for idx_list in sindice_idx_in]
    sclass2count_in = [[round(1.0 + (x * 0.1), 4) for x in c2c_list] for c2c_list in sclass2count_idx_in]

    # --- Background (Out) HCA setup ---
    global_count_dict_out = {x: 0 for x in range(max_cls_out + 1)}
    for label in train_orig_y_min_out:
        idx = int(np.round((float(label) - 0.1) * 10))
        global_count_dict_out[min(max_cls_out, max(0, idx))] += 1
    global_count_dict_sqrt_out = {k: np.sqrt(v) for k, v in global_count_dict_out.items()}

    _, sindice_idx_out, sclass2count_idx_out = level_split(
        [0, max_cls_out + 0.01], global_count_dict_sqrt_out, args.head_num, if_age=True, ob_vmax=max_cls_out
    )
    sweight_out = get_mh_weight(global_count_dict_out, sindice_idx_out)
    sindice_out = [[round(0.1 + (x * 0.1), 4) for x in idx_list] for idx_list in sindice_idx_out]
    sclass2count_out = [[round(0.1 + (x * 0.1), 4) for x in c2c_list] for c2c_list in sclass2count_idx_out]

    # Model In (Foreground) - in_dim=36 (Global + In)
    model_in = resnet50_cls_mh(
        in_dim=36, feat_dim=args.feat_dim, dropout=args.dropout,
        fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
        start_update=args.start_update, start_smooth=args.start_smooth,
        kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
        class_indice=global_class_indice_in, class2count=global_class2count_in,
        head_class_indice=sindice_in, head_class2count=sclass2count_in,
        head_weight=sweight_in, cmax=2.0
    )
    model_in = torch.nn.DataParallel(model_in).cuda()

    # Model Out (Background) - in_dim=36 (Global + Out)
    model_out = resnet50_cls_mh(
        in_dim=36, feat_dim=args.feat_dim, dropout=args.dropout,
        fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
        start_update=args.start_update, start_smooth=args.start_smooth,
        kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
        class_indice=global_class_indice_out, class2count=global_class2count_out,
        head_class_indice=sindice_out, head_class2count=sclass2count_out,
        head_weight=sweight_out, cmax=1.0
    )
    model_out = torch.nn.DataParallel(model_out).cuda()

    best_ckpt_in = os.path.join(args.save_dir, f'{exp_tag}_in_ckpt.best.pth.tar')
    best_ckpt_out = os.path.join(args.save_dir, f'{exp_tag}_out_ckpt.best.pth.tar')

    if not args.evaluate_only:
        optimizer_in = torch.optim.AdamW(model_in.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_in = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_in, T_max=args.epochs)

        optimizer_out = torch.optim.AdamW(model_out.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler_out = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_out, T_max=args.epochs)

        for epoch in range(args.epochs):
            train_one_epoch_2region(train_loader, model_in, model_out, optimizer_in, optimizer_out, epoch)
            scheduler_in.step()
            scheduler_out.step()

        torch.save({'state_dict': model_in.state_dict()}, best_ckpt_in)
        torch.save({'state_dict': model_out.state_dict()}, best_ckpt_out)
    else:
        if os.path.exists(best_ckpt_in):
            model_in.load_state_dict(torch.load(best_ckpt_in)['state_dict'])
        if os.path.exists(best_ckpt_out):
            model_out.load_state_dict(torch.load(best_ckpt_out)['state_dict'])

    print("Now evaluating 2-Region models on test set")
    test_res_in, _ = validate_2region(test_loader, model_in, train_orig_y_min_in, is_in=True,
                                      prefix='Test (Foreground)')
    test_res_out, _ = validate_2region(test_loader, model_out, train_orig_y_min_out, is_in=False,
                                       prefix='Test (Background)')

    for head_choice in ['singlecls', 'adjcls', 'combadd', 'combmul']:
        res_tr = collect_2region_predictions_df(train_eval_loader, df_train, model_in, model_out, head_type=head_choice)
        res_va = collect_2region_predictions_df(val_loader, df_val, model_in, model_out, head_type=head_choice)
        res_te = collect_2region_predictions_df(test_loader, df_test, model_in, model_out, head_type=head_choice)
        res_all = pd.concat([res_tr, res_va, res_te], ignore_index=True).sort_values('In_Center_GT').reset_index(
            drop=True)

        print(f"\\n==========================================")
        print(f"   2-Region HCA [{exp_tag.upper()}] ({head_choice}) Final Alignment Results")
        print(f"==========================================")
        print(f"[Test Set - In]  Interval Acc: {res_te['In_Is_In_Range'].mean():.2%}")
        print(f"[Test Set - Out] Interval Acc: {res_te['Out_Is_In_Range'].mean():.2%}")
        print(f"==========================================")

        if head_choice == args.plot_head:
            for prefix in ['2region_hca', f'2region_hca_{exp_tag}']:
                plot_2region_regression_results(res_tr, f'2Region HCA ({head_choice}): Training Set',
                                                f'./test_2region/{prefix}_train_results.png')
                plot_2region_regression_results(res_va, f'2Region HCA ({head_choice}): Validation Set',
                                                f'./test_2region/{prefix}_val_results.png')
                plot_2region_regression_results(res_te, f'2Region HCA ({head_choice}): Test Set',
                                                f'./test_2region/{prefix}_test_results.png')
                plot_2region_regression_results(res_all, f'2Region HCA ({head_choice}): ALL Set',
                                                f'./test_2region/{prefix}_all_results.png')

                res_tr.to_csv(f'./test_2region/{prefix}_train_results.csv', index=False)
                res_va.to_csv(f'./test_2region/{prefix}_validation_results.csv', index=False)
                res_te.to_csv(f'./test_2region/{prefix}_test_results.csv', index=False)
                res_all.to_csv(f'./test_2region/{prefix}_all_results.csv', index=False)


def train_one_epoch_2region(train_loader, model_in, model_out, optimizer_in, optimizer_out, epoch):
    model_in.train()
    model_out.train()

    head_num_in = model_in.module.head_num
    head_num_out = model_out.module.head_num

    if args.fds:
        model_in.module.FDS.update_last_epoch_stats(epoch)
        model_out.module.FDS.update_last_epoch_stats(epoch)

    fds_enc_in, fds_lab_in = [], []
    fds_enc_out, fds_lab_out = [], []

    for tab_feat_in, tab_feat_out, targets_in, targets_out, weights in train_loader:
        tab_feat_in = tab_feat_in.cuda(non_blocking=True)
        tab_feat_out = tab_feat_out.cuda(non_blocking=True)
        targets_in = targets_in.cuda(non_blocking=True)
        targets_out = targets_out.cuda(non_blocking=True)
        weights = weights.cuda(non_blocking=True).view(-1)

        # --- Foreground (In) Branch ---
        y_center_in = targets_in[:, 0:1]
        orig_min_in = targets_in[:, 1:2]
        orig_max_in = targets_in[:, 2:3]
        tol_min_in = targets_in[:, 3:4]
        tol_max_in = targets_in[:, 4:5]

        c_center_in = torch.clamp((y_center_in - 1.0) * 10.0, 0.0, 10.0)
        fds_targets_in = c_center_in.round()

        optimizer_in.zero_grad()
        allpre_in = model_in(tab_feat_in, fds_targets_in, epoch)

        if args.fds and epoch >= args.start_update:
            fds_enc_in.append(allpre_in['encoding'].detach())
            fds_lab_in.append(fds_targets_in.squeeze(1).detach())

        cls_grid_in = torch.arange(11, device=tab_feat_in.device, dtype=torch.float32).unsqueeze(0)
        soft_gt_in = torch.exp(-0.5 * ((cls_grid_in - c_center_in) / 0.65) ** 2)
        soft_gt_in = soft_gt_in / (soft_gt_in.sum(dim=1, keepdim=True) + 1e-7)

        loss_ce_x_in = (-(soft_gt_in * F.log_softmax(allpre_in['x'], dim=1)).sum(dim=1) * weights).mean()
        loss_ce_adj_in = (-(soft_gt_in * F.log_softmax(allpre_in['adjust_x'], dim=1)).sum(dim=1) * weights).mean()

        loss_mh_in = 0.0
        for hi in range(head_num_in):
            tmp_cindice = torch.tensor(model_in.module.head_class_indice[hi], dtype=torch.float32,
                                       device=tab_feat_in.device)
            gt_c = Count2Class(y_center_in.view(-1), tmp_cindice).long()
            ce_hi = F.cross_entropy(allpre_in['mhpre_dict'][f'cls_{hi}'], gt_c, reduction='none')
            loss_mh_in = loss_mh_in + (ce_hi * weights).mean()

        reg_weights_in = 0.7 + 0.3 * weights if (args.lds or args.reweight != 'none') else weights
        preds_dict_in = model_in.module.hier_combine_continuous(allpre_in, temp=args.temp)
        loss_reg_in = 0.0
        for k in ['singlecls', 'adjcls', 'combadd', 'combmul']:
            p_val = preds_dict_in[k]
            l1_center = (F.smooth_l1_loss(p_val, y_center_in, beta=0.08, reduction='none').view(
                -1) * reg_weights_in).mean()
            tube_core = ((F.relu(orig_min_in - p_val) + F.relu(p_val - orig_max_in)).view(-1) * reg_weights_in).mean()
            tube_tol = ((F.relu(tol_min_in - p_val) + F.relu(p_val - tol_max_in)).view(-1) * reg_weights_in).mean()
            loss_reg_in = loss_reg_in + (2.0 * l1_center + 3.0 * tube_core + 5.0 * tube_tol)

        loss_in = loss_ce_x_in + loss_ce_adj_in + 0.5 * loss_mh_in + loss_reg_in
        if args.regularization_weight > 0:
            loss_in = loss_in + args.regularization_weight * batchwise_ranking_regularizer(allpre_in['encoding'],
                                                                                           y_center_in,
                                                                                           lambda_val=args.interpolation_lambda)

        loss_in.backward()
        optimizer_in.step()

        # --- Background (Out) Branch ---
        y_center_out = targets_out[:, 0:1]
        orig_min_out = targets_out[:, 1:2]
        orig_max_out = targets_out[:, 2:3]
        tol_min_out = targets_out[:, 3:4]
        tol_max_out = targets_out[:, 4:5]

        c_center_out = torch.clamp((y_center_out - 0.1) * 10.0, 0.0, 9.0)
        fds_targets_out = c_center_out.round()

        optimizer_out.zero_grad()
        allpre_out = model_out(tab_feat_out, fds_targets_out, epoch)

        if args.fds and epoch >= args.start_update:
            fds_enc_out.append(allpre_out['encoding'].detach())
            fds_lab_out.append(fds_targets_out.squeeze(1).detach())

        cls_grid_out = torch.arange(10, device=tab_feat_out.device, dtype=torch.float32).unsqueeze(0)
        soft_gt_out = torch.exp(-0.5 * ((cls_grid_out - c_center_out) / 0.65) ** 2)
        soft_gt_out = soft_gt_out / (soft_gt_out.sum(dim=1, keepdim=True) + 1e-7)

        loss_ce_x_out = (-(soft_gt_out * F.log_softmax(allpre_out['x'], dim=1)).sum(dim=1) * weights).mean()
        loss_ce_adj_out = (-(soft_gt_out * F.log_softmax(allpre_out['adjust_x'], dim=1)).sum(dim=1) * weights).mean()

        loss_mh_out = 0.0
        for hi in range(head_num_out):
            tmp_cindice = torch.tensor(model_out.module.head_class_indice[hi], dtype=torch.float32,
                                       device=tab_feat_out.device)
            gt_c = Count2Class(y_center_out.view(-1), tmp_cindice).long()
            ce_hi = F.cross_entropy(allpre_out['mhpre_dict'][f'cls_{hi}'], gt_c, reduction='none')
            loss_mh_out = loss_mh_out + (ce_hi * weights).mean()

        reg_weights_out = 0.7 + 0.3 * weights if (args.lds or args.reweight != 'none') else weights
        preds_dict_out = model_out.module.hier_combine_continuous(allpre_out, temp=args.temp)
        loss_reg_out = 0.0
        for k in ['singlecls', 'adjcls', 'combadd', 'combmul']:
            p_val = preds_dict_out[k]
            l1_center = (F.smooth_l1_loss(p_val, y_center_out, beta=0.08, reduction='none').view(
                -1) * reg_weights_out).mean()
            tube_core = (
                        (F.relu(orig_min_out - p_val) + F.relu(p_val - orig_max_out)).view(-1) * reg_weights_out).mean()
            tube_tol = ((F.relu(tol_min_out - p_val) + F.relu(p_val - tol_max_out)).view(-1) * reg_weights_out).mean()
            loss_reg_out = loss_reg_out + (2.0 * l1_center + 3.0 * tube_core + 5.0 * tube_tol)

        loss_out = loss_ce_x_out + loss_ce_adj_out + 0.5 * loss_mh_out + loss_reg_out
        if args.regularization_weight > 0:
            loss_out = loss_out + args.regularization_weight * batchwise_ranking_regularizer(allpre_out['encoding'],
                                                                                             y_center_out,
                                                                                             lambda_val=args.interpolation_lambda)

        loss_out.backward()
        optimizer_out.step()

    if args.fds:
        if len(fds_enc_in) > 0:
            model_in.module.FDS.update_running_stats(torch.cat(fds_enc_in, dim=0), torch.cat(fds_lab_in, dim=0), epoch)
        if len(fds_enc_out) > 0:
            model_out.module.FDS.update_running_stats(torch.cat(fds_enc_out, dim=0), torch.cat(fds_lab_out, dim=0),
                                                      epoch)


class interval_evaluator():
    def __init__(self):
        self.pre, self.tol_min, self.tol_max, self.orig_min = [], [], [], []

    def update(self, tmp_pre, tmp_tol_min, tmp_tol_max, tmp_orig_min):
        self.pre.append(tmp_pre.view(-1, 1))
        self.tol_min.append(tmp_tol_min.view(-1, 1))
        self.tol_max.append(tmp_tol_max.view(-1, 1))
        self.orig_min.append(tmp_orig_min.view(-1, 1))

    def evaluate(self, train_orig_y_min):
        tmp_pre = torch.cat(self.pre, dim=0).view(-1)
        tmp_tol_min = torch.cat(self.tol_min, dim=0).view(-1)
        tmp_tol_max = torch.cat(self.tol_max, dim=0).view(-1)
        tmp_orig_min = torch.cat(self.orig_min, dim=0).view(-1)
        self.result = {'bal': shot_interval_metrics(tmp_pre, tmp_tol_min, tmp_tol_max, tmp_orig_min, train_orig_y_min)}
        return self.result

    def print_result(self, bstr='bal'):
        shot_dict = self.result[bstr]
        return_str = '%12s&\t%12s&\t%10s&\t%10s&\t%10s&\t%10s&\n' % (
            'Metrics', 'Raw_Overall', 'Bal_All', 'Many', 'Medium', 'Few'
        )
        return_str += '%12s&\t%11.2f%%&\t%9.2f%%&\t%9.2f%%&\t%9.2f%%&\t%9.2f%%&\n' % (
            'Int_Acc(%)', shot_dict['all']['raw_acc'], shot_dict['all']['acc'],
            shot_dict['many']['acc'], shot_dict['median']['acc'], shot_dict['low']['acc']
        )
        return return_str


def validate_2region(val_loader, model, train_orig_y_min=None, is_in=True, prefix='Val'):
    losses_mse = AverageMeter('Center_MSE', ':.4f')
    model.eval()
    eval_dict = {k: interval_evaluator() for k in ['singlecls', 'adjcls', 'combadd', 'combmul']}

    with torch.no_grad():
        for tab_feat_in, tab_feat_out, targets_in, targets_out, _ in val_loader:
            tab_feats = (tab_feat_in if is_in else tab_feat_out).cuda(non_blocking=True)
            targets = (targets_in if is_in else targets_out).cuda(non_blocking=True)

            y_center = targets[:, 0:1]
            orig_min = targets[:, 1:2]
            tol_min = targets[:, 3:4]
            tol_max = targets[:, 4:5]

            allpre = model(tab_feats)
            pred_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)

            min_val, max_val = (1.0, 2.0) if is_in else (0.1, 1.0)

            for k in ['singlecls', 'adjcls', 'combadd', 'combmul']:
                pred_1dec = torch.clamp(torch.round(pred_dict[k] * 10.0) / 10.0, min_val, max_val)
                eval_dict[k].update(pred_1dec.cpu(), tol_min.cpu(), tol_max.cpu(), orig_min.cpu())

            mse_err = F.mse_loss(pred_dict[args.plot_head], y_center)
            losses_mse.update(mse_err.item(), tab_feats.size(0))

        result_dict = dict()
        center_rmse = np.sqrt(losses_mse.avg)
        print_str = f'[{prefix}] Interval Accuracy (+-{args.tol:.2f}) (Center RMSE: {center_rmse:.4f})\n'
        for tkey in list(eval_dict.keys()):
            result_dict[tkey] = eval_dict[tkey].evaluate(train_orig_y_min)
            print_str += f'<{tkey}> \n' + eval_dict[tkey].print_result(bstr='bal') + '-*' * 25 + '\n'
        logging.info(print_str)
    return result_dict, center_rmse


if __name__ == '__main__':
    main()