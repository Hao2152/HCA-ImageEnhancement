# ranksim.py
import random
import torch
import torch.nn.functional as F


def rank(seq):
    """計算序列由小到大的排名 (0-indexed)"""
    return torch.argsort(torch.argsort(seq).flip(1))


def rank_normalised(seq):
    """正規化至 [1/N, 1] 的排名向量"""
    return (rank(seq) + 1).float() / seq.size()[1]


class TrueRanker(torch.autograd.Function):
    """
    ICLR 2020 Blackbox Differentiation of Combinatorial Solvers:
    透過插值強度 lambda 計算離散排序運算的連續近似梯度
    """
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None


def batchwise_ranking_regularizer(features, targets, lambda_val=2.0):
    """
    計算單一 Batch 的 RankSim 損失：
    features: [B, D] 特徵矩陣 (allpre['encoding'])
    targets : [B, 1] 目標值 (y_center)
    """
    loss = torch.tensor(0.0, device=features.device)
    targets = targets.view(-1, 1)

    # 針對每個獨特的標籤值只抽樣一個樣本，避免頭部類別 (如 2.0) 主導排序矩陣並解決平手 (Ties) 問題
    batch_unique_targets = torch.unique(targets)
    if len(batch_unique_targets) < 2:
        return loss

    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            idx_pool = (targets == target).nonzero(as_tuple=False)[:, 0]
            sampled_indices.append(random.choice(idx_pool).item())
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        x = features
        y = targets

    # 計算特徵空間的兩兩 Cosine Similarity 矩陣 [M, M]
    x_norm = F.normalize(x.view(x.size(0), -1), dim=1)
    xxt = torch.matmul(x_norm, x_norm.permute(1, 0))

    # 計算特徵相似度排名與真實標籤距離排名的 MSE
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0, 1)).detach()
        feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), lambda_val)
        loss = loss + F.mse_loss(feature_ranks, label_ranks)

    return loss