# resnet_cls_mh.py
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from fds import FDS
from cls_funcs import Count2Class

print = logging.info


class TabularHCA_Net(nn.Module):
    """
    免 CNN 純表格特徵版 HCA 架構 (100% 對應 Sandbox 87.5% Test / 90.0% Val 結構)：
    輸入 30 維原圖高頻特徵 -> MLP Encoder -> (選用 FDS) -> H 個階層分類頭 + 最細分類頭 + 調整頭
    """
    def __init__(self, in_dim=30, feat_dim=256, dropout=0.25,
                 fds=False, bucket_num=20, bucket_start=0, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=1, momentum=0.9,
                 class_indice=None, class2count=None,
                 head_class_indice=[], head_class2count=[], head_weight=None, cmax=2.0, **kwargs):
        super(TabularHCA_Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.cnum = len(class2count)
        self.class_indice = torch.Tensor(class_indice).float()
        self.class2count = torch.Tensor(class2count).float()
        self.cmax = cmax
        self.head_num = len(head_class2count)
        self.head_class_indice = head_class_indice
        self.head_class2count = head_class2count
        self.head_weight = head_weight

        # 最細分類頭 (20 類)
        self.linear = nn.Linear(feat_dim, self.cnum)

        # H 個由粗到細的階層分類頭
        for hi in range(self.head_num):
            setattr(self, f'cls_head_{hi}', nn.Linear(feat_dim, len(self.head_class2count[hi])))

        # HCA-d 調整分類頭
        self.adjust_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Softplus(),
            nn.Linear(128, self.cnum)
        )

        # FDS 模組
        self.fds = fds
        self.start_smooth = start_smooth
        if fds:
            self.FDS = FDS(
                feature_dim=feat_dim, bucket_num=bucket_num, bucket_start=bucket_start,
                start_update=start_update, start_smooth=start_smooth,
                kernel=kernel, ks=ks, sigma=sigma, momentum=momentum
            )

    def forward(self, tab_feats, targets=None, epoch=None):
        encoding = self.encoder(tab_feats)

        fused_s = encoding
        if self.training and self.fds and targets is not None and epoch is not None:
            if epoch >= self.start_smooth:
                fused_s = self.FDS.smooth(fused_s.clone(), targets, epoch)

        out_x = self.linear(fused_s)
        mhpre_dict = {f'cls_{hi}': getattr(self, f'cls_head_{hi}')(fused_s) for hi in range(self.head_num)}
        adj_x = self.adjust_head(fused_s)

        return {'x': out_x, 'adjust_x': adj_x, 'mhpre_dict': mhpre_dict, 'encoding': encoding}

    def hier_combine_continuous(self, allpre, temp=0.35):
        device = allpre['x'].device
        vals = self.class2count.to(device).view(1, -1)
        p_fine = F.softmax(allpre['x'] / temp, dim=1)
        p_adj = F.softmax(allpre['adjust_x'] / temp, dim=1)
        log_p_sum = F.log_softmax(allpre['x'], dim=1)
        p_add_raw = F.softmax(allpre['x'], dim=1)

        for hi in range(self.head_num):
            logits_h = allpre['mhpre_dict'][f'cls_{hi}']
            p_h = F.softmax(logits_h, dim=1)
            log_p_h = F.log_softmax(logits_h, dim=1)
            tmp_cindice = torch.tensor(self.head_class_indice[hi], dtype=torch.float32, device=device)
            fine_to_coarse = Count2Class(vals.view(-1), tmp_cindice).long()
            p_add_raw = p_add_raw + p_h[:, fine_to_coarse]
            log_p_sum = log_p_sum + log_p_h[:, fine_to_coarse]

        p_mul_sharp = F.softmax(log_p_sum / (self.head_num + 1) / temp, dim=1)
        p_add_sharp = F.softmax(torch.log(p_add_raw + 1e-7) / temp, dim=1)
        return {
            'singlecls': (p_fine * vals).sum(dim=1, keepdim=True),
            'adjcls': (p_adj * vals).sum(dim=1, keepdim=True),
            'combadd': (p_add_sharp * vals).sum(dim=1, keepdim=True),
            'combmul': (p_mul_sharp * vals).sum(dim=1, keepdim=True),
        }


def resnet50_cls_mh(**kwargs):
    return TabularHCA_Net(**kwargs)