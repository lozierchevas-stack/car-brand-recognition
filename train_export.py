#!/usr/bin/env python3
"""ArcFace 度量学习训练与 ONNX 自动化导出 (核心交付 2).

训练管线要点:
- Backbone: MobileNetV3 (移除分类层) -> 512 维 Embedding;
- 损失: ArcFace (加性角度间隔), 可选 CosFace / Softmax 对照;
- 采样: P-K Sampler (每 Batch 抽 P 个品牌, 每品牌 K 张图);
- 增强: 夜间模拟 + 局部遮挡 + 颜色抖动 (训练集);
- 加速: AMP 混合精度 + 预热余弦退火学习率;
- 评测: 验证集类中心召回率 recall@1 + ArcFace 分类精度;
- 检查点: 保存最佳模型 + 类中心 gallery (可直接构建 Faiss 索引)。

导出管线要点:
- eval 模式静态化导出 ONNX (含 L2 归一化输出, 内积检索=余弦);
- 动态 Batch 支持; onnx.checker 校验 + onnxruntime 数值对齐比对。

用法:
  python train_export.py train --epochs 60 --val-limit 800
  python train_export.py export --checkpoint artifacts/checkpoints/best.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from car_recog import utils
from car_recog.config import load_config
from car_recog.dataset import CarBrandDataset, PKSampler, make_loaders
from car_recog.losses import build_loss
from car_recog.model import MobileNetV3Embedding, NormalizedEmbedding
from car_recog.transforms import get_train_transforms, get_val_transforms

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TB_AVAILABLE = False

logger = utils.setup_logger("car-recog.train")


# ======================================================================
# 优化器 / 学习率策略
# ======================================================================

def _optimizer_param_groups(*modules, lr: float, momentum: float, weight_decay: float):
    """分离权重衰减组: BN/bias 不做 L2 衰减, 权重做衰减 (工业惯例)."""
    decay, no_decay = [], []
    for m in modules:
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim <= 1 or "bias" in n.lower()) else decay).append(p)
    return [
        {"params": decay, "lr": lr, "momentum": momentum, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "momentum": momentum, "weight_decay": 0.0},
    ]


def _make_scheduler(optimizer, epochs: int, warmup_epochs: int, steps_per_epoch: int):
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    total_steps = max(warmup_steps + 1, int(epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))  # 余弦退火

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# 训练器
# ======================================================================

class Trainer:
    def __init__(self, cfg, args):
        utils.seed_everything(cfg.project.seed)
        self.cfg = cfg
        self.args = args
        self.device = utils.get_device("auto")  # 有 CUDA 用 CUDA, 否则自动降级 CPU
        self.amp = bool(cfg.train.amp) and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        logger.info("训练设备: %s (AMP=%s)", self.device, self.amp)

        self.train_loader, self.val_loader, self.brands = make_loaders(cfg)
        self.num_classes = len(self.brands)
        logger.info("训练集: %d 张 / %d 品牌 | 验证集: %d 张",
                    len(self.train_loader.dataset), self.num_classes,
                    len(self.val_loader.dataset))

        self.model = MobileNetV3Embedding(cfg.model.embedding_dim, cfg.model.backbone,
                                          cfg.model.pretrained, cfg.model.dropout)
        self.criterion = build_loss(cfg.train.loss.type, cfg.model.embedding_dim,
                                    self.num_classes, cfg.train.loss)
        self.optimizer = torch.optim.SGD(
            _optimizer_param_groups(self.model, self.criterion,
                                    lr=cfg.train.lr, momentum=cfg.train.momentum,
                                    weight_decay=cfg.train.weight_decay))
        self.scheduler = _make_scheduler(self.optimizer, cfg.train.epochs,
                                         cfg.train.warmup_epochs, len(self.train_loader))
        self.scaler = GradScaler("cuda", enabled=self.amp)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        self.epoch0 = 0
        self.best_metric = 0.0
        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            self._load_checkpoint(args.resume)

        self.writer = None
        if TB_AVAILABLE and args.tensorboard:
            self.writer = SummaryWriter(self.ckpt_dir / "tb")

    # ---------- 检查点 ----------

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.criterion.load_state_dict(ckpt["criterion_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch0 = int(ckpt["epoch"]) + 1
        self.best_metric = float(ckpt.get("best_metric", 0.0))
        logger.info("已从 %s 恢复训练 (epoch=%d, best recall@1=%.4f)",
                    path, self.epoch0, self.best_metric)

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict, gallery) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "criterion_state": self.criterion.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": self.best_metric,
            "class_names": self.brands,
            "embedding_dim": self.cfg.model.embedding_dim,
            "backbone": self.cfg.model.backbone,
            "gallery": gallery,  # 类中心向量, 可直接构建 Faiss 索引
            "metrics": metrics,
        }, path)
        logger.info("检查点已保存: %s", path)

    # ---------- 训练 / 评测 ----------

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        self.criterion.train()
        losses, seen = 0.0, 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg.train.epochs}")
        for batch_idx, (img, label) in enumerate(pbar):
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.amp):
                emb = self.model(img)
                loss = self.criterion(emb, label)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            losses += loss.item()
            seen += 1
            if batch_idx % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
        avg = losses / max(1, seen)
        logger.info("Epoch %d 完成: avg_loss=%.4f, lr=%.2e",
                    epoch + 1, avg, self.optimizer.param_groups[0]["lr"])
        return avg

    def recalibrate_bn(self, max_batches: int = 50, batch_size: int = 64) -> bool:
        """BN running stats 重校准: 用训练集 train-mode 前向 (无梯度) 重估统计量.

        动机: 小 batch / 短训练后, Backbone 各 BN 层的 running stats 被噪声污染,
        eval 模式 (导出/评测/推理) 会把所有输入映射到近似相同的特征 (塌缩)。
        重校准后 running stats 恢复为数据集真实统计, 导出的 ONNX 与
        检查点中的 gallery 才能在推理侧与训练侧保持一致。
        返回是否成功 (数据目录缺失时跳过并告警)。
        """
        try:
            ds = CarBrandDataset(self.cfg.paths.cropped_root, "train",
                                 get_val_transforms(self.cfg.dataset.image_size))
        except FileNotFoundError:
            logger.warning("BN 重校准跳过: 裁剪数据目录不存在 (%s), "
                           "将沿用检查点中的 running stats", self.cfg.paths.cropped_root)
            return False
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=False)
        self.model.train()  # train 模式下前向才会用 batch 统计更新 running stats
        with torch.no_grad():
            for i, (img, _) in enumerate(loader):
                if i >= max_batches:
                    break
                self.model(img.to(self.device))
        logger.info("BN running stats 重校准完成 (%d 批, %d 张图像)",
                    min(max_batches, len(loader)), min(max_batches, len(loader)) * batch_size)
        return True

    @torch.inference_mode()
    def evaluate(self, limit: int | None = None,
                 train_mode: bool = False) -> tuple[dict, dict | None]:
        """验证集评测: 类中心 gallery + 余弦检索 recall@1, 附带 ArcFace 分类精度.

        train_mode=True 时用 batch 统计 (训练早期 running stats 未收敛时的
        监控指标); 否则用 eval 模式 (running stats, 与导出/推理一致)。
        """
        self.model.train() if train_mode else self.model.eval()
        self.criterion.eval()
        embs, labels = [], []
        correct, total = 0, 0
        for i, (img, label) in enumerate(self.val_loader):
            if limit and i * self.val_loader.batch_size >= limit:
                break
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.amp):
                emb = self.model(img)
                logits = self.criterion.logits(emb)
            correct += (logits.argmax(1) == label).sum().item()
            total += label.numel()
            embs.append(F.normalize(emb.float(), dim=1).cpu().numpy())
            labels.append(label.cpu().numpy())
        if not embs:
            return {"recall@1": 0.0, "arcface_acc": 0.0}, None
        E = np.vstack(embs)
        L = np.concatenate(labels)
        classes = np.unique(L)
        centers = np.vstack([utils.l2_normalize(E[L == c].mean(0)) for c in classes])
        sim = E @ centers.T
        pred = classes[sim.argmax(1)]
        metrics = {
            "recall@1": float((pred == L).mean()),
            "arcface_acc": correct / max(1, total),
            "val_samples": int(len(L)),
            "num_classes": int(len(classes)),
        }
        gallery = {"vectors": centers.astype(np.float32),
                   "brands": [self.brands[int(c)] for c in classes]}
        return metrics, gallery

    def run(self) -> None:
        epochs = self.cfg.train.epochs
        if self.epoch0 >= epochs:
            # 断点续训语义: 恢复点已不早于目标轮数时, --epochs 按"追加轮数"处理,
            # 否则 range(epoch0, epochs) 为空集, 一个 epoch 都不会跑
            epochs = self.epoch0 + epochs
            self.cfg.train.epochs = epochs
            self.scheduler = _make_scheduler(self.optimizer, epochs,
                                             self.cfg.train.warmup_epochs,
                                             len(self.train_loader))
            logger.info("恢复点 epoch=%d >= 目标轮数, 按追加 %d 轮处理 (总轮数 %d)",
                        self.epoch0, epochs - self.epoch0, epochs)
        logger.info("开始训练: %d epochs, P=%d K=%d, batch=%d, loss=%s",
                    epochs, self.cfg.train.p, self.cfg.train.k,
                    self.cfg.train.p * self.cfg.train.k, self.cfg.train.loss.type)
        for epoch in range(self.epoch0, epochs):
            t0 = time.time()
            avg_loss = self.train_one_epoch(epoch)
            if self.writer:
                self.writer.add_scalar("train/loss", avg_loss, epoch)
                self.writer.add_scalar("train/lr",
                                       self.optimizer.param_groups[0]["lr"], epoch)
            if (epoch + 1) % self.cfg.train.eval_every == 0 or \
                    epoch == self.cfg.train.epochs - 1:
                # 训练中监控: 用 batch 统计 (running stats 尚未收敛时 eval 模式不可信)
                metrics, gallery = self.evaluate(limit=self.cfg.train.val_limit, train_mode=True)
                logger.info("验证(监控) | recall@1=%.4f arcface_acc=%.4f (samples=%d, %.1fs)",
                            metrics["recall@1"], metrics["arcface_acc"],
                            metrics["val_samples"], time.time() - t0)
                if self.writer:
                    self.writer.add_scalar("val/recall@1", metrics["recall@1"], epoch)
                    self.writer.add_scalar("val/arcface_acc", metrics["arcface_acc"], epoch)
                if metrics["recall@1"] >= self.best_metric:
                    self.best_metric = metrics["recall@1"]
                    self.save_checkpoint(self.ckpt_dir / "best.pth", epoch, metrics, gallery)
                # 每轮落盘 last, 意外关机后可从最近一轮续训
                self.save_checkpoint(self.ckpt_dir / "last.pth", epoch, metrics, gallery)

        # 训练结束: 重校准 BN running stats, 以与推理侧一致的 eval 模式复评并落盘
        self.recalibrate_bn(max_batches=50)
        final_limit = None if self.args.final_eval else self.cfg.train.val_limit
        metrics, gallery = self.evaluate(limit=final_limit, train_mode=False)
        logger.info("最终评测(eval模式) | recall@1=%.4f arcface_acc=%.4f (samples=%d)",
                    metrics["recall@1"], metrics["arcface_acc"], metrics["val_samples"])
        final_epoch = self.cfg.train.epochs - 1
        self.save_checkpoint(self.ckpt_dir / "best.pth", final_epoch, metrics, gallery)
        self.save_checkpoint(self.ckpt_dir / "last.pth", final_epoch, metrics, gallery)
        with open(self.ckpt_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("训练结束, 检查点已保存: %s (best 监控 recall@1=%.4f)",
                    self.ckpt_dir, self.best_metric)


# ======================================================================
# ONNX 导出
# ======================================================================

def export_onnx(args) -> None:
    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"检查点不存在: {ckpt_path}, 请先运行: python train_export.py train")
    # 检查点为本项目自产的可信文件, 内含 numpy gallery 数组, 需关闭 weights_only
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dim = int(ckpt.get("embedding_dim", cfg.model.embedding_dim))
    backbone = ckpt.get("backbone", cfg.model.backbone)

    model = MobileNetV3Embedding(embedding_dim=dim, backbone=backbone,
                                 pretrained=False, dropout=cfg.model.dropout)
    model.load_state_dict(ckpt["model_state"])
    # BN 重校准: 确保导出模型的 running stats 与训练数据分布一致
    # (小 batch/短训练后不重校准, 导出的模型在推理侧会输出塌缩特征)
    try:
        from car_recog.transforms import get_val_transforms
        ds = CarBrandDataset(cfg.paths.cropped_root, "train",
                             get_val_transforms(cfg.dataset.image_size))
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
        model.train()
        with torch.no_grad():
            for i, (img, _) in enumerate(loader):
                if i >= 50:
                    break
                model(img)
        logger.info("BN running stats 重校准完成 (导出前)")
    except FileNotFoundError:
        logger.warning("裁剪数据目录不存在, 跳过 BN 重校准, 沿用检查点 running stats")
    model.eval()
    export_model = NormalizedEmbedding(model) if cfg.export.normalize_output else model

    out_path = Path(args.output or cfg.export.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opset = int(args.opset or cfg.export.opset)
    dynamic_axes = ({"input": {0: "batch"}, "embedding": {0: "batch"}}
                    if cfg.export.dynamic_batch else None)

    logger.info("导出 ONNX -> %s (opset=%d, dynamic_batch=%s, normalize=%s)",
                out_path, opset, bool(dynamic_axes), cfg.export.normalize_output)
    torch.onnx.export(
        export_model,
        torch.zeros(1, 3, cfg.dataset.image_size, cfg.dataset.image_size),
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # 结构校验
    try:
        import onnx
        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        logger.info("onnx.checker 校验通过")
    except ImportError:
        logger.warning("未安装 onnx 包, 跳过结构校验 (pip install onnx)")

    # 数值对齐: onnxruntime 输出 vs PyTorch 输出
    try:
        import onnxruntime as ort
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 3, cfg.dataset.image_size, cfg.dataset.image_size)
                       ).astype(np.float32)
        expected = export_model(torch.from_numpy(x)).detach().numpy()
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"input": x})[0]
        err = float(np.abs(expected - got).max())
        if err < 1e-4:
            logger.info("onnxruntime 数值对齐校验通过 (max_err=%.2e)", err)
        else:
            logger.warning("onnxruntime 输出与 PyTorch 不一致 (max_err=%.2e)", err)
    except ImportError:
        logger.warning("未安装 onnxruntime, 跳过数值校验")

    # 侧车文件: 类别名映射
    names_path = out_path.with_suffix(".class_names.json")
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": ckpt["class_names"]}, f, ensure_ascii=False, indent=2)

    logger.info("导出完成: %s (%.1f MB)", out_path, out_path.stat().st_size / (1 << 20))
    logger.info("下一步: python export_trt.py --onnx %s   # 可选 TensorRT 加速", out_path)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ArcFace 训练与 ONNX 导出")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="训练 ArcFace 度量学习模型")
    p_train.add_argument("--config", default=None, help="配置文件路径 (默认 configs/default.yaml)")
    p_train.add_argument("--cropped-root", default=None, help="覆盖裁剪数据根目录")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--p", type=int, default=None, help="P-K 采样的品牌数 P")
    p_train.add_argument("--k", type=int, default=None, help="P-K 采样每品牌样本数 K")
    p_train.add_argument("--lr", type=float, default=None)
    p_train.add_argument("--val-limit", type=int, default=None, help="每次验证最多评估图像数")
    p_train.add_argument("--eval-every", type=int, default=None)
    p_train.add_argument("--no-amp", action="store_true", help="禁用 AMP 混合精度")
    p_train.add_argument("--final-eval", action="store_true", help="训练结束后全量验证")
    p_train.add_argument("--tensorboard", action="store_true", help="启用 TensorBoard 日志")
    p_train.add_argument("--resume", default="", help="从检查点恢复训练")
    p_train.set_defaults(func=lambda a: run_train(a))

    p_export = sub.add_parser("export", help="导出 ONNX 模型")
    p_export.add_argument("--config", default=None)
    p_export.add_argument("--checkpoint", default="artifacts/checkpoints/best.pth")
    p_export.add_argument("--output", default=None, help="ONNX 输出路径")
    p_export.add_argument("--opset", type=int, default=None)
    p_export.set_defaults(func=lambda a: export_onnx(a))

    args = parser.parse_args()
    args.func(args)


def run_train(args) -> None:
    overrides = {
        "train.epochs": args.epochs, "train.p": args.p, "train.k": args.k,
        "train.lr": args.lr, "train.val_limit": args.val_limit,
        "train.eval_every": args.eval_every,
    }
    if args.cropped_root:
        overrides["paths.cropped_root"] = args.cropped_root
    if args.no_amp:
        overrides["train.amp"] = False
    cfg = load_config(args.config, overrides=overrides)
    trainer = Trainer(cfg, args)
    trainer.run()


if __name__ == "__main__":
    main()
