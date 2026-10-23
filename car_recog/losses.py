"""度量学习损失: ArcFace (核心) / CosFace / 纯 Softmax (对照).

ArcFace 原理: 特征与分类权重均归一化到超球面后, 对真实类别施加加性角度间隔 m,
即 cos(theta_y + m) < cos(theta_y), 迫使同品牌车辆特征在角度空间内聚、
不同品牌分离 —— 这与下游 Faiss 余弦检索的度量空间完全一致。

分类权重 W 由本模块持有并随训练更新, 推理阶段不参与计算。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss.

    参数:
        s: 特征缩放因子 (超球面半径), 典型值 32~64;
        m: 角度间隔 (弧度), 典型值 0.3~0.5;
        easy_margin: True 时使用 cos(theta)-m 的简化形式 (训练前期更稳定);
        label_smoothing: 标签平滑系数, 缓解过拟合与错误标注。
    """

    def __init__(self, embedding_dim: int, num_classes: int, s: float = 32.0,
                 m: float = 0.5, easy_margin: bool = False, label_smoothing: float = 0.0):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.ls = label_smoothing
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # cos(theta+m) 越过 cos(pi-m) 后转为线性形式, 保持单调 (Deng et al. 2019)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self._eps = 1e-7

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """归一化余弦相似度 logits (cos theta), 供验证集分类精度统计使用."""
        x = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(x, w).clamp(-1.0 + self._eps, 1.0 - self._eps)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos_theta = self.logits(embeddings)
        if self.easy_margin:
            cos_m = torch.where(cos_theta > 0, cos_theta - self.m, cos_theta)
        else:
            sin_theta = torch.sqrt(1.0 - cos_theta ** 2).clamp(min=self._eps)
            cos_m = cos_theta * self.cos_m - sin_theta * self.sin_m
            cos_m = torch.where(cos_theta > self.th, cos_m, cos_theta - self.mm)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cos_theta.dtype)
        logits = self.s * (cos_theta * (1 - one_hot) + cos_m * one_hot)
        return F.cross_entropy(logits, labels, label_smoothing=self.ls)


class CosFaceLoss(nn.Module):
    """Additive Cosine Margin Loss (对照实验用)."""

    def __init__(self, embedding_dim: int, num_classes: int, s: float = 32.0,
                 m: float = 0.35, label_smoothing: float = 0.0):
        super().__init__()
        self.s = s
        self.m = m
        self.ls = label_smoothing
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings, dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(x, w).clamp(-1.0, 1.0)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos_theta = self.logits(embeddings)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cos_theta.dtype)
        logits = self.s * (cos_theta * (1 - one_hot) + (cos_theta - self.m) * one_hot)
        return F.cross_entropy(logits, labels, label_smoothing=self.ls)


class SoftmaxHead(nn.Module):
    """纯 Softmax 分类头 (基线对照实验用)."""

    def __init__(self, embedding_dim: int, num_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.ls = label_smoothing

    def logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.fc(embeddings), labels, label_smoothing=self.ls)


def build_loss(name: str, embedding_dim: int, num_classes: int, loss_cfg) -> nn.Module:
    name = (name or "arcface").lower()
    if name == "arcface":
        return ArcFaceLoss(embedding_dim, num_classes,
                           s=float(loss_cfg.get("s", 32.0)),
                           m=float(loss_cfg.get("m", 0.5)),
                           easy_margin=bool(loss_cfg.get("easy_margin", False)),
                           label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    if name == "cosface":
        return CosFaceLoss(embedding_dim, num_classes,
                           s=float(loss_cfg.get("s", 32.0)),
                           m=float(loss_cfg.get("m", 0.35)),
                           label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    if name == "softmax":
        return SoftmaxHead(embedding_dim, num_classes,
                           label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    raise ValueError(f"未知损失类型: {name}, 可选: arcface / cosface / softmax")
