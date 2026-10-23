"""轻量级度量学习 Backbone: MobileNetV3 + 512 维 Embedding 头.

设计说明:
- 移除 MobileNetV3 原始分类层, 替换为 Linear + BatchNorm1d 的 Embedding 头;
- 分类权重 W 不放在模型里, 而是由 losses.ArcFaceLoss 持有 —— 训练完成后
  导出 Embedding 网络即可直接推理, 无需拆层 (见 train_export.py 的 export 命令);
- BN 置于 Embedding 头末尾可稳定度量学习收敛, 导出时以 eval 模式跟踪 (running stats)。
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("car-recog.model")

_BACKBONE_WEIGHTS = {
    "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
}


class MobileNetV3Embedding(nn.Module):
    """输入 (N,3,H,W) -> 输出 (N,embedding_dim) 特征向量 (未归一化)."""

    def __init__(self, embedding_dim: int = 512, backbone: str = "mobilenet_v3_large",
                 pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        if backbone not in _BACKBONE_WEIGHTS:
            raise ValueError(f"不支持的 backbone: {backbone}, 可选: {list(_BACKBONE_WEIGHTS)}")
        weights = _BACKBONE_WEIGHTS[backbone] if pretrained else None
        try:
            m = getattr(models, backbone)(weights=weights)
        except Exception as e:  # 离线/下载失败等场景: 降级为随机初始化并告警
            logger.warning("预训练权重加载失败 (%s), 使用随机初始化继续", e)
            m = getattr(models, backbone)(weights=None)

        # 去掉 classifier, 保留 features + avgpool
        self.backbone = nn.Sequential(*list(m.children())[:-1])

        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, 3, 224, 224)).flatten(1).shape[1]

        head = [nn.Linear(feat_dim, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim)]
        if dropout > 0:
            head.insert(0, nn.Dropout(dropout))
        self.head = nn.Sequential(*head)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.head(feat)


class NormalizedEmbedding(nn.Module):
    """导出用包装: 输出 L2 归一化特征, 使 Faiss 内积检索等价于余弦相似度."""

    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.embedding(x), dim=1)
