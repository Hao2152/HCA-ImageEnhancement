# eval_metrics.py
import numpy as np
import torch
from collections import defaultdict


def shot_interval_metrics(preds, tol_min, tol_max, orig_y_min, train_orig_y_min, many_shot_thr=100, low_shot_thr=25):
    """
    計算擴展區間 [acceptable_alpha_min - 0.3, acceptable_alpha_max + 0.3] 的：
    1. Int_Acc (%): 預測值落在擴展區間內的命中率
    2. Tube_L1    : 超出擴展區間外的距離誤差 (落在擴展區間內為 0)
    3. Core_L1    : 距離原始區間中心的差距 (供挑選最佳模型時防止全猜常數坍縮)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(tol_min, torch.Tensor):
        tol_min = tol_min.detach().cpu().numpy()
    if isinstance(tol_max, torch.Tensor):
        tol_max = tol_max.detach().cpu().numpy()
    if isinstance(orig_y_min, torch.Tensor):
        orig_y_min = orig_y_min.detach().cpu().numpy()

    preds = np.round(np.array(preds, dtype=np.float64).reshape(-1), 2)
    tol_min = np.round(np.array(tol_min, dtype=np.float64).reshape(-1), 2)
    tol_max = np.round(np.array(tol_max, dtype=np.float64).reshape(-1), 2)
    orig_y_min = np.round(np.array(orig_y_min, dtype=np.float64).reshape(-1), 2)
    train_orig_y_min = np.round(np.array(train_orig_y_min, dtype=np.float64).reshape(-1), 2)

    # 1. 判斷是否落在 [acceptable_alpha_min - 0.3, acceptable_alpha_max + 0.3] 內
    in_interval = ((preds >= tol_min - 1e-4) & (preds <= tol_max + 1e-4)).astype(np.float64) * 100.0
    # 2. 計算超出擴展區間外的 Tube L1 Error
    tube_l1 = np.maximum(0.0, tol_min - preds) + np.maximum(0.0, preds - tol_max)

    many_acc, med_acc, low_acc = [], [], []
    many_tube, med_tube, low_tube = [], [], []
    many_cnt, med_cnt, low_cnt = [], [], []

    # 依據原始 20 個類別 (orig_y_min: 0.1 ~ 2.0) 計算 Balanced 指標
    for l in np.unique(orig_y_min):
        tr_cnt = len(train_orig_y_min[train_orig_y_min == l])
        mask = (orig_y_min == l)
        te_cnt = np.sum(mask)
        cls_acc_mean = np.mean(in_interval[mask])
        cls_tube_mean = np.mean(tube_l1[mask])

        if tr_cnt > many_shot_thr:
            many_acc.append(cls_acc_mean); many_tube.append(cls_tube_mean); many_cnt.append(te_cnt)
        elif tr_cnt < low_shot_thr:
            low_acc.append(cls_acc_mean); low_tube.append(cls_tube_mean); low_cnt.append(te_cnt)
        else:
            med_acc.append(cls_acc_mean); med_tube.append(cls_tube_mean); med_cnt.append(te_cnt)

    shot_dict = defaultdict(dict)
    for name, acc_l, tube_l, cnt_l in [
        ('many', many_acc, many_tube, many_cnt),
        ('median', med_acc, med_tube, med_cnt),
        ('low', low_acc, low_tube, low_cnt)
    ]:
        shot_dict[name]['acc'] = np.mean(acc_l) if len(cnt_l) > 0 else -1.0
        shot_dict[name]['tube_l1'] = np.mean(tube_l) if len(cnt_l) > 0 else -1.0

    all_acc = many_acc + med_acc + low_acc
    all_tube = many_tube + med_tube + low_tube
    shot_dict['all']['acc'] = np.mean(all_acc) if len(all_acc) > 0 else -1.0
    shot_dict['all']['tube_l1'] = np.mean(all_tube) if len(all_tube) > 0 else -1.0
    shot_dict['all']['raw_acc'] = np.mean(in_interval)
    shot_dict['all']['raw_tube_l1'] = np.mean(tube_l1)
    return shot_dict