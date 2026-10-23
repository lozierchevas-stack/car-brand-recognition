"""训练/验证数据增强: 含夜间模拟 (RandomNightAug) 与局部遮挡 (RandomOcclusion).

夜间增强先验 (夜间图像: 低亮度、低对比、偏冷色温、高噪声):
  1) 亮度整体压暗; 2) Gamma 校正进一步压缩动态范围;
  3) 蓝色通道增益模拟路灯/冷色温光源; 4) 高斯噪声模拟传感器噪声。

遮挡增强: 随机 1~3 个矩形区域, 以灰块/黑块/噪声填充,
模拟行人、立柱、相邻车辆等真实场景遮挡。
"""
from __future__ import annotations

import math
import random

import torch
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # 与 car_recog.utils 保持一致
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomNightAug:
    """夜间/暗光条件模拟, 作用于 (C,H,W) float32 [0,1] 张量."""

    def __init__(self, p: float = 0.3, brightness: tuple = (0.15, 0.55),
                 gamma: tuple = (1.2, 2.4), noise_sigma: tuple = (0.01, 0.06),
                 blue_gain: tuple = (1.0, 1.35)):
        self.p = p
        self.brightness = brightness
        self.gamma = gamma
        self.noise_sigma = noise_sigma
        self.blue_gain = blue_gain

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        img = img * random.uniform(*self.brightness)
        img = torch.pow(img.clamp(min=0.0), random.uniform(*self.gamma))
        img[2] = img[2] * random.uniform(*self.blue_gain)  # RGB 的 B 通道
        noise = torch.randn_like(img) * random.uniform(*self.noise_sigma)
        return (img + noise).clamp(0.0, 1.0)


class RandomOcclusion:
    """局部遮挡模拟: 随机矩形区域填充, 作用于 (C,H,W) float32 [0,1] 张量."""

    def __init__(self, p: float = 0.35, num_patches: tuple = (1, 3),
                 scale: tuple = (0.04, 0.18), ratio: tuple = (0.3, 3.3),
                 fills: tuple = ("black", "gray", "noise")):
        self.p = p
        self.num_patches = num_patches
        self.scale = scale
        self.ratio = ratio
        self.fills = fills

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return img
        c, h, w = img.shape
        for _ in range(random.randint(*self.num_patches)):
            area = h * w * random.uniform(*self.scale)
            aspect = random.uniform(*self.ratio)
            ph = min(h - 1, int(round(math.sqrt(area * aspect))))
            pw = min(w - 1, int(round(math.sqrt(area / aspect))))
            if ph < 1 or pw < 1:
                continue
            top = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            fill = random.choice(self.fills)
            if fill == "black":
                img[:, top:top + ph, left:left + pw] = 0.0
            elif fill == "gray":
                img[:, top:top + ph, left:left + pw] = random.uniform(0.2, 0.6)
            else:
                img[:, top:top + ph, left:left + pw] = torch.rand(
                    c, ph, pw, dtype=img.dtype, device=img.device)
        return img


def get_train_transforms(size: int = 224, aug_cfg=None) -> T.Compose:
    aug = aug_cfg or {}
    ops = [
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(*aug.get("color_jitter", [0.25, 0.25, 0.2, 0.05])),
        T.ToTensor(),
        RandomNightAug(p=float(aug.get("night_p", 0.3))),
        RandomOcclusion(p=float(aug.get("occlusion_p", 0.35))),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    erasing_p = float(aug.get("random_erasing_p", 0.0))
    if erasing_p > 0:
        ops.append(T.RandomErasing(p=erasing_p, scale=(0.05, 0.2)))
    return T.Compose(ops)


def get_val_transforms(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
