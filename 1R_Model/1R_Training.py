# main.py
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

from datasets import IMDBWIKI, TABULAR_COLS, extract_tabular_matrix
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
parser.add_argument('--regularization_weight', type=float, default=0.0, help='RankSim loss weight gamma (e.g. 1.0)')
parser.add_argument('--interpolation_lambda', type=float, default=2.0, help='RankSim interpolation strength lambda')

# ================= 訓練與模型超參數 (預設 100% 對齊 Sandbox) =================
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--tol', type=float, default=0.3, help='interval tolerance (+/- 0.3)')
parser.add_argument('--csv_path', type=str, default='./alpha_data.csv')
parser.add_argument('--raw_csv_path', type=str, default='./ALL_Best_Summary_ExtendedFeatures_Advanced.csv')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--feat_dim', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--temp', type=float, default=0.35)
parser.add_argument('--save_best', action='store_true', default=False)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--evaluate_only', action='store_true', default=False)
parser.add_argument('--plot_head', type=str, default='adjcls', choices=['combadd', 'combmul', 'adjcls', 'singlecls'])
parser.add_argument('--save_dir', type=str, default='./checkpoints_ablation')
parser.add_argument('--log_dir', type=str, default='analy_log')
parser.add_argument('--log_name', type=str, default='ablation_hca.log')
parser.add_argument('--head_num', type=int, default=4)
args, unknown = parser.parse_known_args()

# 若使用者加上 --ranksim 但未指定 regularization_weight，自動設為 1.0
if args.ranksim and args.regularization_weight <= 0:
    args.regularization_weight = 1.0


def plot_regression_results(df_results, title, filename):
    plt.figure(figsize=(12, 6))
    plt.fill_between(df_results.index, df_results['Min_Alpha'], df_results['Max_Alpha'],
                     color='gray', alpha=0.3, label='Acceptable Range')
    plt.plot(df_results.index, df_results['Center_GT'], 'k--', label='GT (Center)', linewidth=1.2)

    correct = df_results[df_results['Is_In_Range'] == True]
    incorrect = df_results[df_results['Is_In_Range'] == False]

    plt.scatter(correct.index, correct['Predicted_Alpha'], color='green', s=15, label='Correct (In Range)', zorder=5)
    plt.scatter(incorrect.index, incorrect['Predicted_Alpha'], color='red', s=15, label='Incorrect (Out of Range)', zorder=5)

    plt.xlabel('Samples (Sorted by GT Center)')
    plt.ylabel('USM Alpha Value')
    plt.title(title)
    plt.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    plt.ylim(-0.1, 2.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_prediction_distribution(df_results, title, filename):
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(0.05, 2.05, 21)
    plt.hist(df_results['Center_GT'], bins=bin_edges, alpha=0.5, color='blue', edgecolor='black',
             label='Ground Truth (Center)')
    plt.hist(df_results['Predicted_Alpha'], bins=bin_edges, alpha=0.8, color='red', edgecolor='red',
             label='Predicted Alpha', histtype='step', linewidth=3)

    plt.title(f'Alpha Distribution: Ground Truth vs Predicted ({title})')
    plt.xlabel('USM Alpha Value')
    plt.ylabel('Number of Samples')
    plt.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0.0, 2.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


@torch.no_grad()
def collect_predictions_df(loader, df_split, model, head_type='combmul'):
    model.eval()
    preds_list, center_list, min_list, max_list = [], [], [], []

    for tab_feats, targets, _ in loader:
        tab_feats = tab_feats.cuda(non_blocking=True)
        allpre = model(tab_feats)
        pred_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)
        pred_val = pred_dict[head_type]

        preds_list.extend(pred_val.cpu().view(-1).numpy())
        center_list.extend(targets[:, 0].numpy())
        min_list.extend(targets[:, 3].numpy())
        max_list.extend(targets[:, 4].numpy())

    # 自動抓取 df_split 中的 original_musiq 欄位 (若無則相容大小寫或設為 NaN)
    if 'original_musiq' in df_split.columns:
        orig_musiq_vals = np.round(df_split['original_musiq'].values.astype(float), 4)
    elif 'Original_MUSIQ' in df_split.columns:
        orig_musiq_vals = np.round(df_split['Original_MUSIQ'].values.astype(float), 4)
    else:
        orig_musiq_vals = np.full(len(df_split), np.nan)

    split_vals = df_split['split'].values if 'split' in df_split.columns else ['unknown'] * len(df_split)
    results_df = pd.DataFrame({
        'Image': df_split['image'].values,
        'Split': split_vals,  # 👈 標註屬於 train / val / test
        'original_musiq': orig_musiq_vals,
        'Center_GT': np.round(center_list, 1),
        'Min_Alpha': np.round(min_list, 1),
        'Max_Alpha': np.round(max_list, 1),
        'Predicted_Alpha': np.clip(np.round(preds_list, 1), 0.1, 2.0),
    })
    results_df['Is_In_Range'] = (
        (results_df['Predicted_Alpha'] >= results_df['Min_Alpha'] - 1e-4) &
        (results_df['Predicted_Alpha'] <= results_df['Max_Alpha'] + 1e-4)
    )
    results_df = results_df.sort_values('Center_GT').reset_index(drop=True)
    return results_df


def main():
    max_cls = 19
    global_class_indice = np.round(np.linspace(0.1, 1.9, 19) + 0.05, 4)
    global_class2count = np.round(np.linspace(0.1, 2.0, 20), 4)

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
    os.makedirs('./test', exist_ok=True)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f"{exp_tag}_{args.log_name}")),
            logging.StreamHandler()
        ])
    print = logging.info
    print(f"=====> Pure Tabular HCA Ablation: [{exp_tag.upper()}] | Args: {args}")

    df = pd.read_csv(args.csv_path)
    missing_cols = [c for c in TABULAR_COLS if c not in df.columns]
    if len(missing_cols) > 0:
        raw_df = pd.read_csv(args.raw_csv_path)
        cols_to_merge = ['image'] + [c for c in TABULAR_COLS + ['acceptable_alpha_min', 'acceptable_alpha_max', 'acceptable_alpha_center'] if c not in df.columns]
        df = pd.merge(df, raw_df[cols_to_merge].drop_duplicates(subset=['image']), on='image', how='left')

    df['acceptable_alpha_min'] = df['acceptable_alpha_min'].round(1)
    df['acceptable_alpha_max'] = df['acceptable_alpha_max'].round(1)
    df['acceptable_alpha_center'] = ((df['acceptable_alpha_min'] + df['acceptable_alpha_max']) / 2.0).round(2)

    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    df_val = df[df['split'] == 'val'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    train_orig_y_min = df_train['acceptable_alpha_min'].values

    scaler = StandardScaler()
    X_tr_tab = scaler.fit_transform(extract_tabular_matrix(df_train))
    X_va_tab = scaler.transform(extract_tabular_matrix(df_val))
    X_te_tab = scaler.transform(extract_tabular_matrix(df_test))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dataset = IMDBWIKI(
        df_train, tab_features=X_tr_tab, tol=args.tol, split='train',
        reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma
    )
    train_eval_dataset = IMDBWIKI(df_train, tab_features=X_tr_tab, tol=args.tol, split='val')
    val_dataset = IMDBWIKI(df_val, tab_features=X_va_tab, tol=args.tol, split='val')
    test_dataset = IMDBWIKI(df_test, tab_features=X_te_tab, tol=args.tol, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    global_count_dict = {x: 0 for x in range(max_cls + 1)}
    for label in train_orig_y_min:
        idx = int(np.round(float(label) * 10)) - 1
        global_count_dict[min(max_cls, max(0, idx))] += 1
    global_count_dict_sqrt = {k: np.sqrt(v) for k, v in global_count_dict.items()}

    _, sindice_idx, sclass2count_idx = level_split(
        [0, max_cls + 0.01], global_count_dict_sqrt, args.head_num, if_age=True, ob_vmax=max_cls
    )
    sweight = get_mh_weight(global_count_dict, sindice_idx)
    sindice = [[round((x + 1) * 0.1, 4) for x in idx_list] for idx_list in sindice_idx]
    sclass2count = [[round((x + 1) * 0.1, 4) for x in c2c_list] for c2c_list in sclass2count_idx]

    model = resnet50_cls_mh(
        in_dim=X_tr_tab.shape[1], feat_dim=args.feat_dim, dropout=args.dropout,
        fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
        start_update=args.start_update, start_smooth=args.start_smooth,
        kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt,
        class_indice=global_class_indice, class2count=global_class2count,
        head_class_indice=sindice, head_class2count=sclass2count,
        head_weight=sweight, cmax=2.0
    )

    model = torch.nn.DataParallel(model).cuda()
    best_ckpt_path = os.path.join(args.save_dir, f'{exp_tag}_ckpt.best.pth.tar')
    best_epoch = args.epochs

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        best_epoch = checkpoint.get('epoch', args.epochs)
        model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)

    if not args.evaluate_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_raw_acc = -1.0
        best_val_rmse = 1e5

        for epoch in range(args.epochs):
            train_one_epoch(train_loader, model, optimizer, epoch)
            scheduler.step()

            if (epoch + 1) % 20 == 0 or (epoch + 1) == args.epochs or args.save_best:
                val_res, val_center_rmse = validate(val_loader, model, train_orig_y_min=train_orig_y_min, prefix=f'Val Epoch {epoch+1}')
                cur_raw_acc = val_res[args.plot_head]['bal']['all']['raw_acc']
                if args.save_best:
                    if (cur_raw_acc > best_val_raw_acc + 1e-4) or (abs(cur_raw_acc - best_val_raw_acc) <= 1e-4 and val_center_rmse < best_val_rmse):
                        best_val_raw_acc = cur_raw_acc
                        best_val_rmse = val_center_rmse
                        best_epoch = epoch + 1
                        torch.save({'epoch': best_epoch, 'state_dict': model.state_dict()}, best_ckpt_path)

        if not args.save_best:
            best_epoch = args.epochs
            torch.save({'epoch': best_epoch, 'state_dict': model.state_dict()}, best_ckpt_path)
        else:
            checkpoint = torch.load(best_ckpt_path)
            best_epoch = checkpoint.get('epoch', best_epoch)
            model.load_state_dict(checkpoint['state_dict'], strict=True)

    print("Now testing on the test set")
    test_res, _ = validate(test_loader, model, train_orig_y_min=train_orig_y_min, prefix='Test')

    for head_choice in ['singlecls', 'adjcls', 'combadd', 'combmul']:
        res_tr = collect_predictions_df(train_eval_loader, df_train, model, head_type=head_choice)
        res_va = collect_predictions_df(val_loader, df_val, model, head_type=head_choice)
        res_te = collect_predictions_df(test_loader, df_test, model, head_type=head_choice)
        res_all = pd.concat([res_tr, res_va, res_te], ignore_index=True).sort_values('Center_GT').reset_index(drop=True)

        tr_r = np.sqrt(mean_squared_error(res_tr['Center_GT'], res_tr['Predicted_Alpha']))
        va_r = np.sqrt(mean_squared_error(res_va['Center_GT'], res_va['Predicted_Alpha']))
        te_r = np.sqrt(mean_squared_error(res_te['Center_GT'], res_te['Predicted_Alpha']))
        all_r = np.sqrt(mean_squared_error(res_all['Center_GT'], res_all['Predicted_Alpha']))
        te_bal = test_res[head_choice]['bal']['all']['acc']

        print(f"\n==========================================")
        print(f"   HCA [{exp_tag.upper()}] ({head_choice}) Final Alignment Results (Epoch {best_epoch})")
        print(f"==========================================")
        print(f"[Training Set]   Interval Acc: {res_tr['Is_In_Range'].mean():.2%} | Center RMSE: {tr_r:.4f}")
        print(f"[Validation Set] Interval Acc: {res_va['Is_In_Range'].mean():.2%} | Center RMSE: {va_r:.4f}")
        print(
            f"[Test Set]       Interval Acc: {res_te['Is_In_Range'].mean():.2%} | Bal Acc: {te_bal:.2f}% | Center RMSE: {te_r:.4f}")
        print(f"[ALL Combined]   Interval Acc: {res_all['Is_In_Range'].mean():.2%} | Center RMSE: {all_r:.4f}")
        print(f"==========================================")

        if head_choice == args.plot_head:
            for prefix in ['hca', f'hca_{exp_tag}']:
                plot_regression_results(res_tr, f'HCA [{exp_tag.upper()}] ({head_choice}): Training Set',
                                        f'./test/{prefix}_train_results.png')
                plot_regression_results(res_va, f'HCA [{exp_tag.upper()}] ({head_choice}): Validation Set',
                                        f'./test/{prefix}_val_results.png')
                plot_regression_results(res_te, f'HCA [{exp_tag.upper()}] ({head_choice}): Test Set',
                                        f'./test/{prefix}_test_results.png')
                plot_regression_results(res_all, f'HCA [{exp_tag.upper()}] ({head_choice}): ALL Set',
                                        f'./test/{prefix}_all_results.png')
                plot_prediction_distribution(res_tr, f"Training Set [{exp_tag.upper()}]",
                                             f"./test/{prefix}_dist_train.png")
                plot_prediction_distribution(res_va, f"Validation Set [{exp_tag.upper()}]",
                                             f"./test/{prefix}_dist_val.png")
                plot_prediction_distribution(res_te, f"Test Set [{exp_tag.upper()}]", f"./test/{prefix}_dist_test.png")
                plot_prediction_distribution(res_all, f"ALL Set [{exp_tag.upper()}]", f"./test/{prefix}_dist_all.png")
                res_tr.to_csv(f'./test/{prefix}_train_results.csv', index=False)
                res_va.to_csv(f'./test/{prefix}_validation_results.csv', index=False)
                res_te.to_csv(f'./test/{prefix}_test_results.csv', index=False)
                res_all.to_csv(f'./test/{prefix}_all_results.csv', index=False)


def train_one_epoch(train_loader, model, optimizer, epoch):
    model.train()
    head_num = model.module.head_num

    if args.fds:
        model.module.FDS.update_last_epoch_stats(epoch)

    fds_encodings, fds_labels = [], []

    for tab_feats, targets, weights in train_loader:
        tab_feats = tab_feats.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        weights = weights.cuda(non_blocking=True).view(-1)

        y_center = targets[:, 0:1]
        orig_min = targets[:, 1:2]
        orig_max = targets[:, 2:3]
        tol_min = targets[:, 3:4]
        tol_max = targets[:, 4:5]

        c_center = torch.clamp(y_center * 10.0 - 1.0, 0.0, 19.0)
        fds_targets = c_center.round()

        optimizer.zero_grad()
        allpre = model(tab_feats, fds_targets, epoch)

        if args.fds and epoch >= args.start_update:
            fds_encodings.append(allpre['encoding'].detach())
            fds_labels.append(fds_targets.squeeze(1).detach())

        cls_grid = torch.arange(20, device=tab_feats.device, dtype=torch.float32).unsqueeze(0)
        soft_gt = torch.exp(-0.5 * ((cls_grid - c_center) / 0.65) ** 2)
        soft_gt = soft_gt / (soft_gt.sum(dim=1, keepdim=True) + 1e-7)

        loss_ce_x = (-(soft_gt * F.log_softmax(allpre['x'], dim=1)).sum(dim=1) * weights).mean()
        loss_ce_adj = (-(soft_gt * F.log_softmax(allpre['adjust_x'], dim=1)).sum(dim=1) * weights).mean()

        loss_mh = 0.0
        for hi in range(head_num):
            tmp_cindice = torch.tensor(model.module.head_class_indice[hi], dtype=torch.float32, device=tab_feats.device)
            gt_c = Count2Class(y_center.view(-1), tmp_cindice).long()
            ce_hi = F.cross_entropy(allpre['mhpre_dict'][f'cls_{hi}'], gt_c, reduction='none')
            loss_mh = loss_mh + (ce_hi * weights).mean()

        reg_weights = 0.7 + 0.3 * weights if (args.lds or args.reweight != 'none') else weights
        preds_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)
        loss_reg = 0.0
        for k in ['singlecls', 'adjcls', 'combadd', 'combmul']:
            p_val = preds_dict[k]
            l1_center = (F.smooth_l1_loss(p_val, y_center, beta=0.08, reduction='none').view(-1) * reg_weights).mean()
            tube_core = ((F.relu(orig_min - p_val) + F.relu(p_val - orig_max)).view(-1) * reg_weights).mean()
            tube_tol = ((F.relu(tol_min - p_val) + F.relu(p_val - tol_max)).view(-1) * reg_weights).mean()
            loss_reg = loss_reg + (2.0 * l1_center + 3.0 * tube_core + 5.0 * tube_tol)

        loss = loss_ce_x + loss_ce_adj + 0.5 * loss_mh + loss_reg

        # 若啟用 RankSim，計算特徵空間與標籤空間的排序相似度正規化損失
        if args.regularization_weight > 0:
            loss_ranksim = batchwise_ranking_regularizer(
                allpre['encoding'], y_center, lambda_val=args.interpolation_lambda
            )
            loss = loss + args.regularization_weight * loss_ranksim

        loss.backward()
        optimizer.step()

    if args.fds and len(fds_encodings) > 0:
        model.module.FDS.update_running_stats(
            torch.cat(fds_encodings, dim=0),
            torch.cat(fds_labels, dim=0),
            epoch
        )


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
        return_str += '%12s&\t%12.4f&\t%10.4f&\t%10.4f&\t%10.4f&\t%10.4f&\n' % (
            'Tube_L1', shot_dict['all']['raw_tube_l1'], shot_dict['all']['tube_l1'],
            shot_dict['many']['tube_l1'], shot_dict['median']['tube_l1'], shot_dict['low']['tube_l1']
        )
        return return_str


def validate(val_loader, model, train_orig_y_min=None, prefix='Val'):
    losses_mse = AverageMeter('Center_MSE', ':.4f')
    model.eval()
    eval_dict = {k: interval_evaluator() for k in ['singlecls', 'adjcls', 'combadd', 'combmul']}

    with torch.no_grad():
        for tab_feats, targets, _ in val_loader:
            tab_feats = tab_feats.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            y_center = targets[:, 0:1]
            orig_min = targets[:, 1:2]
            tol_min = targets[:, 3:4]
            tol_max = targets[:, 4:5]

            allpre = model(tab_feats)
            pred_dict = model.module.hier_combine_continuous(allpre, temp=args.temp)

            for k in ['singlecls', 'adjcls', 'combadd', 'combmul']:
                # 👇 先四捨五入對齊到小數第一位 (0.1 ~ 2.0)，再送入評估器
                pred_1dec = torch.clamp(torch.round(pred_dict[k] * 10.0) / 10.0, 0.1, 2.0)
                eval_dict[k].update(pred_1dec.cpu(), tol_min.cpu(), tol_max.cpu(), orig_min.cpu())

            mse_err = F.mse_loss(pred_dict[args.plot_head], y_center)
            losses_mse.update(mse_err.item(), tab_feats.size(0))

        result_dict = dict()
        center_rmse = np.sqrt(losses_mse.avg)
        print_str = f'[{prefix}] Interval Accuracy (+-{args.tol:.2f}) & Tube_L1 (Center RMSE: {center_rmse:.4f})\n'
        for tkey in list(eval_dict.keys()):
            result_dict[tkey] = eval_dict[tkey].evaluate(train_orig_y_min)
            print_str += f'<{tkey}> \n' + eval_dict[tkey].print_result(bstr='bal') + '-*' * 25 + '\n'
        print_str += '==' * 25 + '\n'
        logging.info(print_str)
    return result_dict, center_rmse


if __name__ == '__main__':
    main()