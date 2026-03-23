"""
Phase 1: Full training script — Swin-Tiny + ArcFace
Phase A (epoch 1-10): Freeze backbone stages 0,1
Phase B (epoch 11-50): Unfreeze all, reduce LR, CosineAnnealing
Early stopping: val EER không cải thiện 8 epoch
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

# Chạy từ project root hoặc phase1: python models/phase1/train.py
_phase1_dir = Path(__file__).resolve().parent
if str(_phase1_dir) not in sys.path:
    sys.path.insert(0, str(_phase1_dir))

import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset import build_datasets
from metrics import compute_auc, compute_eer, compute_similarity_matrix, compute_tar_at_far, compute_template_similarity_scores
from model import (
    BACKBONE_CHOICES as MODEL_BACKBONE_CHOICES,
    get_embedding_output_dim,
    freeze_backbone_stages,
    unfreeze_all,
    build_model,
)
from tta_utils import build_tta_plan, forward_tta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Config ───────────────────────────────────────────────────────────────────

BATCH_SIZE = 32
EPOCHS_TOTAL = 50
EPOCHS_FREEZE = 10
LR_PHASE_A = 3e-4
LR_PHASE_B = 3e-5
WEIGHT_DECAY = 0.05
EARLY_STOP_PATIENCE = 8
NUM_WORKERS = 4
ARC_S_START = 16.0
ARC_S_END = 30.0
ARC_M_START = 0.0
ARC_M_END = 0.35
ARCFACE_WARMUP_EPOCHS = 8
TRIPLET_WEIGHT = 0.35
TRIPLET_MARGIN = 0.20
PK_CLASSES = 8
PK_SAMPLES = 4
TRIPLET_WARMUP_EPOCHS = 8
NEGATIVE_QUEUE_SIZE = 4096
LABEL_SMOOTHING = 0.0
TRIPLET_MINING = "semi-hard"
DROP_PATH_RATE = 0.2
ARC_SUBCENTERS = 1
CA_REDUCTION = 32
PREFETCH_FACTOR = 4
ACCUM_STEPS = 1
EMA_DECAY = 0.0
BACKBONE_NAME = "swin_tiny_patch4_window7_224"
BACKBONE_CHOICES = MODEL_BACKBONE_CHOICES


class PKBatchSampler(Sampler[list[int]]):
    """
    Generate batches with P subjects x K samples each.
    This guarantees positive pairs in-batch for metric learning.
    """

    def __init__(
        self,
        labels: Sequence[int],
        p_classes: int,
        k_samples: int,
        steps_per_epoch: int = 0,
        seed: int = 42,
    ) -> None:
        if p_classes <= 0 or k_samples <= 0:
            raise ValueError("p_classes và k_samples phải > 0")
        self.p_classes = int(p_classes)
        self.k_samples = int(k_samples)
        self.batch_size = self.p_classes * self.k_samples
        self.seed = int(seed)
        self.epoch = 0

        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(labels):
            label_to_indices[int(lab)].append(idx)
        self.label_to_indices = dict(label_to_indices)
        self.unique_labels = np.array(sorted(self.label_to_indices.keys()), dtype=np.int64)
        if len(self.unique_labels) == 0:
            raise RuntimeError("PK sampler không tìm thấy nhãn nào trong train set")

        auto_steps = max(1, len(labels) // self.batch_size)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch > 0 else auto_steps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        replace_labels = len(self.unique_labels) < self.p_classes

        for _ in range(self.steps_per_epoch):
            chosen_labels = rng.choice(self.unique_labels, size=self.p_classes, replace=replace_labels)
            batch: list[int] = []
            for lab in chosen_labels:
                indices = self.label_to_indices[int(lab)]
                replace_samples = len(indices) < self.k_samples
                picks = rng.choice(indices, size=self.k_samples, replace=replace_samples)
                batch.extend(int(x) for x in picks)
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch


class HardNegativeQueue:
    """
    FIFO memory bank for cross-batch negatives.
    Stores normalized embeddings and labels from previous batches.
    """

    def __init__(self, embedding_dim: int, capacity: int, device: torch.device) -> None:
        self.capacity = max(0, int(capacity))
        self.device = device
        self.embedding_dim = embedding_dim
        self.embeddings = torch.empty((0, embedding_dim), device=device, dtype=torch.float32)
        self.labels = torch.empty((0,), device=device, dtype=torch.long)

    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if self.capacity <= 0 or embeddings.numel() == 0:
            return
        with torch.no_grad():
            emb = embeddings.detach()
            lab = labels.detach()
            if emb.device != self.device:
                emb = emb.to(self.device)
            if lab.device != self.device:
                lab = lab.to(self.device)
            self.embeddings = torch.cat([self.embeddings, emb], dim=0)
            self.labels = torch.cat([self.labels, lab], dim=0)
            if self.embeddings.size(0) > self.capacity:
                keep = self.capacity
                self.embeddings = self.embeddings[-keep:]
                self.labels = self.labels[-keep:]

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings, self.labels

    def reset(self) -> None:
        self.embeddings = torch.empty((0, self.embedding_dim), device=self.device, dtype=torch.float32)
        self.labels = torch.empty((0,), device=self.device, dtype=torch.long)


class ModelEMA:
    """
    Exponential moving average of model parameters/buffers.
    Improves generalization and stabilizes low-FAR metrics.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k].detach()
            if not torch.is_floating_point(v):
                v.copy_(src)
            else:
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
    negative_queue: HardNegativeQueue | None = None,
    mining: str = "semi-hard",
) -> tuple[torch.Tensor, int]:
    """
    Batch-hard triplet on L2-normalized embeddings.
    Returns (loss, num_valid_anchors).
    """
    distances = torch.cdist(embeddings, embeddings, p=2)
    same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
    pos_mask = same & ~eye
    neg_mask = ~same

    hardest_pos = distances.masked_fill(~pos_mask, -1e9).max(dim=1).values
    hard_neg = distances.masked_fill(~neg_mask, 1e9).min(dim=1).values
    if mining == "semi-hard":
        semi_neg_mask = neg_mask & (distances > hardest_pos.unsqueeze(1))
        semi_neg = distances.masked_fill(~semi_neg_mask, 1e9).min(dim=1).values
        hardest_neg = torch.where(torch.isfinite(semi_neg), semi_neg, hard_neg)
    else:
        hardest_neg = hard_neg
    has_neg = neg_mask.any(dim=1)

    if negative_queue is not None:
        queue_emb, queue_labels = negative_queue.get()
        if queue_emb.numel() > 0:
            cross_dist = torch.cdist(embeddings, queue_emb, p=2)
            cross_neg_mask = labels.unsqueeze(1).ne(queue_labels.unsqueeze(0))
            cross_hard = cross_dist.masked_fill(~cross_neg_mask, 1e9).min(dim=1).values
            if mining == "semi-hard":
                cross_semi_mask = cross_neg_mask & (cross_dist > hardest_pos.unsqueeze(1))
                cross_semi = cross_dist.masked_fill(~cross_semi_mask, 1e9).min(dim=1).values
                cross_pick = torch.where(torch.isfinite(cross_semi), cross_semi, cross_hard)
            else:
                cross_pick = cross_hard
            hardest_neg = torch.minimum(hardest_neg, cross_pick)
            has_neg = has_neg | cross_neg_mask.any(dim=1)

    valid = pos_mask.any(dim=1) & has_neg

    if valid.any():
        losses = torch.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
        return losses.mean(), int(valid.sum().item())
    return embeddings.sum() * 0.0, 0


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al.).
    features: [N, V, D] where V is number of views (>=2)
    labels:   [N]
    """
    if features.dim() != 3:
        raise ValueError("features must have shape [N, V, D]")
    n, v, d = features.shape
    if v < 2:
        return features.sum() * 0.0

    features = F.normalize(features, p=2, dim=2)
    feats = features.view(n * v, d)
    labels = labels.view(n, 1).repeat(1, v).view(-1)

    device = feats.device
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)
    logits = torch.matmul(feats, feats.T) / max(1e-6, float(temperature))
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(mask) - torch.eye(n * v, device=device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    denom = mask.sum(dim=1).clamp(min=1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / denom
    loss = -mean_log_prob_pos.mean()
    return loss


def run_epoch(
    model: nn.Module,
    arcface: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    phase: str,
    triplet_weight: float = 0.0,
    triplet_margin: float = 0.2,
    negative_queue: HardNegativeQueue | None = None,
    triplet_mining: str = "semi-hard",
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    channels_last: bool = False,
    accum_steps: int = 1,
    ema_model: ModelEMA | None = None,
    center_loss_module: nn.Module | None = None,
    center_loss_weight: float = 0.0,
    center_optimizer: torch.optim.Optimizer | None = None,
    supcon_weight: float = 0.0,
    supcon_temp: float = 0.07,
) -> tuple[float, float, float, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_supcon_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    accum_steps = max(1, int(accum_steps))
    num_batches = len(loader)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"  {phase}", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images_b = None
        if isinstance(images, (tuple, list)) and len(images) == 2:
            images, images_b = images

        images = images.to(DEVICE, non_blocking=True)
        if channels_last and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        if images_b is not None:
            images_b = images_b.to(DEVICE, non_blocking=True)
            if channels_last and DEVICE.type == "cuda":
                images_b = images_b.contiguous(memory_format=torch.channels_last)
        labels = labels.to(DEVICE, non_blocking=True)

        amp_enabled = use_amp and DEVICE.type == "cuda"
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
            emb = model(images)
            emb_b = model(images_b) if images_b is not None else None
        emb_fp32 = emb.float()
        logits = arcface(emb_fp32, labels)
        ce_loss = criterion(logits, labels)
        triplet_loss = emb_fp32.sum() * 0.0
        if triplet_weight > 0:
            triplet_loss, _ = batch_hard_triplet_loss(
                emb_fp32,  # cdist in fp32 is more stable
                labels,
                margin=triplet_margin,
                negative_queue=negative_queue,
                mining=triplet_mining,
            )
        supcon_loss = emb_fp32.sum() * 0.0
        if supcon_weight > 0 and emb_b is not None:
            feats = torch.stack([emb_fp32, emb_b.float()], dim=1)
            supcon_loss = supervised_contrastive_loss(feats, labels, temperature=supcon_temp)
        loss = ce_loss + triplet_weight * triplet_loss + supcon_weight * supcon_loss
        if center_loss_module is not None and center_loss_weight > 0:
            c_loss = center_loss_module(emb_fp32, labels)
            loss = loss + center_loss_weight * c_loss

        if optimizer is not None:
            loss_to_backprop = loss / accum_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)
            if do_step:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    if center_optimizer is not None:
                        scaler.step(center_optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    if center_optimizer is not None:
                        center_optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if center_optimizer is not None:
                    center_optimizer.zero_grad(set_to_none=True)
                if ema_model is not None:
                    ema_model.update(model)

        total_loss += loss.item() * images.size(0)
        total_triplet_loss += triplet_loss.item() * images.size(0)
        total_supcon_loss += supcon_loss.item() * images.size(0)
        with torch.no_grad():
            # Accuracy monitoring should use non-margin logits.
            pred_logits = arcface.forward_no_margin(emb_fp32)
        _, pred = pred_logits.max(1)
        topk = min(5, pred_logits.size(1))
        pred_top5 = pred_logits.topk(topk, dim=1).indices
        correct += pred.eq(labels).sum().item()
        correct_top5 += pred_top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
        total += images.size(0)
        postfix = {
            "loss": f"{loss.item():.4f}",
            "ce": f"{ce_loss.item():.4f}",
            "tri": f"{triplet_loss.item():.4f}",
            "acc": f"{100*correct/total:.2f}%",
            "top5": f"{100*correct_top5/total:.2f}%",
        }
        if supcon_weight > 0:
            postfix["sup"] = f"{supcon_loss.item():.4f}"
        pbar.set_postfix(postfix)
        if negative_queue is not None:
            negative_queue.enqueue(emb_fp32, labels)

    return total_loss / total, correct / total, correct_top5 / total, total_triplet_loss / total


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    use_amp: bool = False,
    channels_last: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embs, labs = [], []
    for images, labels in tqdm(loader, desc="  Extract emb"):
        images = images.to(DEVICE, non_blocking=True)
        if channels_last and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        amp_enabled = use_amp and DEVICE.type == "cuda"
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
            emb = model(images)
        embs.append(emb.float().cpu().numpy())
        labs.append(labels.numpy())
    return np.vstack(embs), np.concatenate(labs)


@torch.no_grad()
def extract_embeddings_tta(
    model: nn.Module,
    loader: DataLoader,
    use_amp: bool = False,
    channels_last: bool = False,
    tta_flips: bool = True,
    tta_variants: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Test-time augmentation: average embeddings from original + horizontally flipped.
    This substantially improves EER and TAR at low FAR.
    """
    model.eval()
    embs, labs = [], []
    amp_enabled = use_amp and DEVICE.type == "cuda"
    if tta_variants is None:
        variants = ["hflip"] if tta_flips else []
    else:
        variants = list(tta_variants)
    tta_plan = build_tta_plan(variants)
    for images, labels in tqdm(loader, desc="  Extract emb (TTA)"):
        images = images.to(DEVICE, non_blocking=True)
        if channels_last and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        emb = forward_tta(model, images, tta_plan, amp_enabled=amp_enabled).float()
        embs.append(emb.cpu().numpy())
        labs.append(labels.numpy())
    return np.vstack(embs), np.concatenate(labs)


def compute_val_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    use_amp: bool = False,
    channels_last: bool = False,
    use_tta: bool = False,
    tta_variants: Sequence[str] | None = None,
    gallery_loader: DataLoader | None = None,
    gallery_score_mode: str = "mean_template",
    gallery_topk: int = 2,
    gallery_probe_znorm: bool = False,
) -> tuple[float, float, float, float, float]:
    if gallery_loader is not None:
        if use_tta:
            gallery_emb, gallery_lab = extract_embeddings_tta(
                model,
                gallery_loader,
                use_amp=use_amp,
                channels_last=channels_last,
                tta_variants=tta_variants,
            )
            probe_emb, probe_lab = extract_embeddings_tta(
                model,
                val_loader,
                use_amp=use_amp,
                channels_last=channels_last,
                tta_variants=tta_variants,
            )
        else:
            gallery_emb, gallery_lab = extract_embeddings(
                model, gallery_loader, use_amp=use_amp, channels_last=channels_last
            )
            probe_emb, probe_lab = extract_embeddings(
                model, val_loader, use_amp=use_amp, channels_last=channels_last
            )
        scores, is_genuine = compute_template_similarity_scores(
            gallery_embeddings=gallery_emb,
            gallery_labels=gallery_lab,
            probe_embeddings=probe_emb,
            probe_labels=probe_lab,
            score_mode=gallery_score_mode,
            topk=gallery_topk,
            probe_znorm=gallery_probe_znorm,
        )
    elif use_tta:
        emb, lab = extract_embeddings_tta(
            model,
            val_loader,
            use_amp=use_amp,
            channels_last=channels_last,
            tta_variants=tta_variants,
        )
        scores, is_genuine = compute_similarity_matrix(emb, lab)
    else:
        emb, lab = extract_embeddings(model, val_loader, use_amp=use_amp, channels_last=channels_last)
        scores, is_genuine = compute_similarity_matrix(emb, lab)
    eer, _ = compute_eer(scores, is_genuine)
    auc = compute_auc(scores, is_genuine)
    tar_1e3 = compute_tar_at_far(scores, is_genuine, 1e-3)
    tar_1e4 = compute_tar_at_far(scores, is_genuine, 1e-4)
    tar_1e5 = compute_tar_at_far(scores, is_genuine, 1e-5)
    return eer, auc, tar_1e3, tar_1e4, tar_1e5


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Hybrid Swin-RS + CA + ArcFace training")
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\AI_PROJECT\PALM_PRINT\data\after\raw_224x224px",
        help="Path to after/raw_224x224px",
    )
    parser.add_argument("--output", type=str, default=None, help="Output dir (default: runs/phase1_<timestamp>)")
    parser.add_argument(
        "--backbone-name",
        type=str,
        default=BACKBONE_NAME,
        choices=BACKBONE_CHOICES,
        help="Backbone model name (Swin family)",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=EPOCHS_TOTAL)
    parser.add_argument("--epochs-freeze", type=int, default=EPOCHS_FREEZE)
    parser.add_argument("--lr-a", type=float, default=LR_PHASE_A)
    parser.add_argument("--lr-b", type=float, default=LR_PHASE_B)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--arc-s-start", type=float, default=ARC_S_START)
    parser.add_argument("--arc-s-end", type=float, default=ARC_S_END)
    parser.add_argument("--arc-m-start", type=float, default=ARC_M_START)
    parser.add_argument("--arc-m-end", type=float, default=ARC_M_END)
    parser.add_argument("--arc-warmup-epochs", type=int, default=ARCFACE_WARMUP_EPOCHS)
    parser.add_argument("--triplet-weight", type=float, default=TRIPLET_WEIGHT)
    parser.add_argument("--triplet-weight-start", type=float, default=0.0)
    parser.add_argument("--triplet-warmup-epochs", type=int, default=TRIPLET_WARMUP_EPOCHS)
    parser.add_argument("--triplet-margin", type=float, default=TRIPLET_MARGIN)
    parser.add_argument("--triplet-mining", type=str, choices=["hard", "semi-hard"], default=TRIPLET_MINING)
    parser.add_argument("--negative-queue-size", type=int, default=NEGATIVE_QUEUE_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--drop-path-rate", type=float, default=DROP_PATH_RATE)
    parser.add_argument("--arc-subcenters", type=int, default=ARC_SUBCENTERS)
    parser.add_argument("--ca-reduction", type=int, default=CA_REDUCTION)
    parser.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY, help="0 disables EMA")
    parser.add_argument(
        "--use-rs-patch-embed",
        action="store_true",
        help="Enable RS-style patch embedding (hybrid mode)",
    )
    parser.add_argument(
        "--use-coordinate-attn",
        action="store_true",
        help="Enable Coordinate Attention after Swin stages (hybrid mode)",
    )
    # Backward-compatible aliases from previous revision.
    parser.add_argument("--disable-rs-patch-embed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-coordinate-attn", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision (fp16, CUDA only)")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--pk-classes", type=int, default=PK_CLASSES, help="P subjects per batch for PK sampler")
    parser.add_argument("--pk-samples", type=int, default=PK_SAMPLES, help="K samples per subject for PK sampler")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 = auto for PK sampler")
    parser.add_argument("--disable-pk-sampler", action="store_true")
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Bắt buộc dùng CUDA, nếu không có sẽ báo lỗi",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(amp=False)
    args = parser.parse_args()

    out_dir = Path(args.output or f"runs/phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_dir}")

    # Chọn thiết bị, cho phép bắt buộc GPU
    global DEVICE
    if args.force_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("force-gpu được bật nhưng torch.cuda.is_available() = False. Kiểm tra driver / CUDA.")
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_amp = bool(args.amp and DEVICE.type == "cuda")
    use_rs_patch_embed = bool(args.use_rs_patch_embed and (not args.disable_rs_patch_embed))
    use_coordinate_attention = bool(args.use_coordinate_attn and (not args.disable_coordinate_attn))

    # ─── Dataset ───────────────────────────────────────────────────────────
    log.info(
        "Building datasets (subject-independent %.2f/%.2f/%.2f)...",
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )
    train_ds, val_ds, test_ds, label_map = build_datasets(
        args.data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_labels = sorted(set(lab for _, lab in train_ds.samples))
    train_label_to_idx = {lb: i for i, lb in enumerate(train_labels)}
    num_classes_arc = len(train_label_to_idx)

    # Remap train labels to 0..num_classes_arc-1
    train_ds.samples = [(p, train_label_to_idx[lab]) for p, lab in train_ds.samples]

    log.info(f"  Train: {len(train_ds)} samples, {num_classes_arc} subjects")
    log.info(f"  Val:   {len(val_ds)} samples")
    log.info(f"  Test:  {len(test_ds)} samples")

    train_batch_sampler = None
    loader_prefetch = args.prefetch_factor if args.workers > 0 else None
    if args.disable_pk_sampler:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(args.workers > 0),
            prefetch_factor=loader_prefetch,
            drop_last=True,
        )
    else:
        train_labels_for_sampler = [lab for _, lab in train_ds.samples]
        train_batch_sampler = PKBatchSampler(
            train_labels_for_sampler,
            p_classes=args.pk_classes,
            k_samples=args.pk_samples,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
        )
        if args.batch_size != train_batch_sampler.batch_size:
            log.info(
                "PK sampler dùng batch_size hiệu dụng = %d (P=%d, K=%d), bỏ qua --batch-size=%d",
                train_batch_sampler.batch_size,
                args.pk_classes,
                args.pk_samples,
                args.batch_size,
            )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=args.workers,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(args.workers > 0),
            prefetch_factor=loader_prefetch,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=loader_prefetch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=loader_prefetch,
    )

    # ─── Model ───────────────────────────────────────────────────────────────
    model, arcface = build_model(
        num_classes=num_classes_arc,
        backbone_name=args.backbone_name,
        embedding_dim=args.embedding_dim,
        drop_path_rate=args.drop_path_rate,
        arc_s=args.arc_s_end,
        arc_m=args.arc_m_end,
        arc_subcenters=args.arc_subcenters,
        use_rs_patch_embed=use_rs_patch_embed,
        use_coordinate_attention=use_coordinate_attention,
        ca_reduction=args.ca_reduction,
        pretrained=(not args.no_pretrained),
    )
    model_config = {
        "backbone_name": args.backbone_name,
        "embedding_dim": args.embedding_dim,
        "drop_path_rate": args.drop_path_rate,
        "arc_s": args.arc_s_end,
        "arc_m": args.arc_m_end,
        "arc_subcenters": args.arc_subcenters,
        "use_rs_patch_embed": use_rs_patch_embed,
        "use_coordinate_attention": use_coordinate_attention,
        "ca_reduction": args.ca_reduction,
    }
    model = model.to(DEVICE)
    if args.channels_last and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    arcface = arcface.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    ema_model = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    embedding_dim = get_embedding_output_dim(model)
    negative_queue = HardNegativeQueue(
        embedding_dim=embedding_dim,
        capacity=args.negative_queue_size,
        device=DEVICE,
    )

    # Phase A: freeze
    freeze_backbone_stages(model.backbone, num_stages=2)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=args.lr_a,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_eer = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    scheduler = None

    log.info("=" * 60)
    log.info("Phase A (epoch 1-%d): Freeze backbone stages 0,1", args.epochs_freeze)
    log.info(
        "Loss config: CE(ls=%.3f) + triplet_w(epoch)[%.3f->%.3f,warmup=%d], margin=%.3f, mining=%s, queue=%d, emb_dim=%d, PK sampler=%s",
        args.label_smoothing,
        args.triplet_weight_start,
        args.triplet_weight,
        args.triplet_warmup_epochs,
        args.triplet_margin,
        args.triplet_mining,
        args.negative_queue_size,
        args.embedding_dim,
        "off" if args.disable_pk_sampler else f"on (P={args.pk_classes}, K={args.pk_samples})",
    )
    log.info(
        "Backbone config: %s, rs_patch=%s, coord_attn=%s (reduction=%d), drop_path=%.3f, arc_subcenters=%d, pretrained=%s",
        args.backbone_name,
        use_rs_patch_embed,
        use_coordinate_attention,
        args.ca_reduction,
        args.drop_path_rate,
        args.arc_subcenters,
        (not args.no_pretrained),
    )
    log.info(
        "Speed config: amp=%s, channels_last=%s, prefetch_factor=%s, accum_steps=%d, ema_decay=%.5f, cudnn_benchmark=%s, tf32=%s",
        use_amp,
        (args.channels_last and DEVICE.type == "cuda"),
        loader_prefetch,
        args.accum_steps,
        args.ema_decay,
        (DEVICE.type == "cuda"),
        (DEVICE.type == "cuda"),
    )
    log.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        negative_queue.reset()

        if args.arc_warmup_epochs <= 1:
            arc_s = args.arc_s_end
            arc_m = args.arc_m_end
        else:
            warm_ratio = min(1.0, (epoch - 1) / (args.arc_warmup_epochs - 1))
            arc_s = args.arc_s_start + warm_ratio * (args.arc_s_end - args.arc_s_start)
            arc_m = args.arc_m_start + warm_ratio * (args.arc_m_end - args.arc_m_start)
        arcface.set_margin_scale(arc_s, arc_m)
        if args.triplet_warmup_epochs <= 1:
            triplet_weight_epoch = args.triplet_weight
        else:
            triplet_ratio = min(1.0, (epoch - 1) / (args.triplet_warmup_epochs - 1))
            triplet_weight_epoch = args.triplet_weight_start + triplet_ratio * (
                args.triplet_weight - args.triplet_weight_start
            )

        # Phase B từ epoch 11
        if epoch == args.epochs_freeze + 1:
            log.info("=" * 60)
            log.info("Phase B: Unfreeze all, LR=%s", args.lr_b)
            log.info("=" * 60)
            unfreeze_all(model.backbone)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_b
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.epochs_freeze
            )

        train_loss, train_acc, train_top5, train_triplet = run_epoch(
            model,
            arcface,
            train_loader,
            optimizer,
            criterion,
            f"E{epoch}",
            triplet_weight=triplet_weight_epoch,
            triplet_margin=args.triplet_margin,
            negative_queue=negative_queue,
            triplet_mining=args.triplet_mining,
            scaler=scaler,
            use_amp=use_amp,
            channels_last=args.channels_last,
            accum_steps=args.accum_steps,
            ema_model=ema_model,
        )

        model_for_eval = ema_model.module if ema_model is not None else model
        val_eer, val_auc, val_tar1e3, val_tar1e4, val_tar1e5 = compute_val_metrics(
            model_for_eval,
            val_loader,
            use_amp=use_amp,
            channels_last=args.channels_last,
        )
        lr = optimizer.param_groups[0]["lr"]

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_top5_acc": train_top5,
            "train_triplet_loss": train_triplet,
            "val_eer": val_eer,
            "val_auc": val_auc,
            "val_tar_1e3": val_tar1e3,
            "val_tar_1e4": val_tar1e4,
            "val_tar_1e5": val_tar1e5,
            "lr": lr,
            "arc_s": arc_s,
            "arc_m": arc_m,
            "triplet_weight": triplet_weight_epoch,
        }
        history.append(rec)
        log.info(
            f"Epoch {epoch:3d} | loss={train_loss:.4f} tri={train_triplet:.4f} "
            f"acc={train_acc*100:.2f}% top5={train_top5*100:.2f}% | "
            f"val EER={val_eer*100:.2f}% AUC={val_auc:.4f} "
            f"TAR@1e-3={val_tar1e3*100:.2f}% TAR@1e-4={val_tar1e4*100:.2f}% "
            f"TAR@1e-5={val_tar1e5*100:.2f}% | "
            f"lr={lr:.2e} arc(s={arc_s:.2f},m={arc_m:.3f}) tri_w={triplet_weight_epoch:.3f}"
        )

        if epoch > args.epochs_freeze and scheduler is not None:
            scheduler.step()

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model_for_eval.state_dict(),
                    "arcface": arcface.state_dict(),
                    "val_eer": val_eer,
                    "val_auc": val_auc,
                    "val_tar_1e3": val_tar1e3,
                    "val_tar_1e4": val_tar1e4,
                    "val_tar_1e5": val_tar1e5,
                    "num_classes_arc": num_classes_arc,
                    "model_config": model_config,
                    "split_config": {
                        "seed": args.seed,
                        "train_ratio": args.train_ratio,
                        "val_ratio": args.val_ratio,
                        "test_ratio": args.test_ratio,
                    },
                    "train_config": {
                        "embedding_dim": args.embedding_dim,
                        "label_smoothing": args.label_smoothing,
                        "triplet_weight": args.triplet_weight,
                        "triplet_weight_start": args.triplet_weight_start,
                        "triplet_warmup_epochs": args.triplet_warmup_epochs,
                        "triplet_margin": args.triplet_margin,
                        "triplet_mining": args.triplet_mining,
                        "negative_queue_size": args.negative_queue_size,
                        "backbone_name": args.backbone_name,
                        "accum_steps": args.accum_steps,
                        "ema_decay": args.ema_decay,
                        "amp": use_amp,
                        "channels_last": (args.channels_last and DEVICE.type == "cuda"),
                        "prefetch_factor": loader_prefetch,
                        "pk_enabled": (not args.disable_pk_sampler),
                        "pk_classes": args.pk_classes,
                        "pk_samples": args.pk_samples,
                        "steps_per_epoch": args.steps_per_epoch,
                    },
                },
                out_dir / "best.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            log.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    log.info(f"Best val EER: {best_val_eer*100:.2f}% at epoch {best_epoch}")

    # ─── Load best & final test ──────────────────────────────────────────────
    ckpt = torch.load(out_dir / "best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    emb_test, lab_test = extract_embeddings(
        model,
        test_loader,
        use_amp=use_amp,
        channels_last=args.channels_last,
    )
    scores_test, is_genuine_test = compute_similarity_matrix(emb_test, lab_test)
    test_eer, thresh = compute_eer(scores_test, is_genuine_test)
    test_auc = compute_auc(scores_test, is_genuine_test)
    test_tar1e3 = compute_tar_at_far(scores_test, is_genuine_test, 1e-3)
    test_tar1e4 = compute_tar_at_far(scores_test, is_genuine_test, 1e-4)
    test_tar1e5 = compute_tar_at_far(scores_test, is_genuine_test, 1e-5)

    log.info("=" * 60)
    log.info("FINAL TEST SET METRICS")
    log.info("=" * 60)
    log.info(f"  EER:          {test_eer*100:.2f}%")
    log.info(f"  AUC:          {test_auc:.4f}")
    log.info(f"  TAR@FAR=1e-3: {test_tar1e3*100:.2f}%")
    log.info(f"  TAR@FAR=1e-4: {test_tar1e4*100:.2f}%")
    log.info(f"  TAR@FAR=1e-5: {test_tar1e5*100:.2f}%")
    log.info(f"  Threshold@EER: {thresh:.4f}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(
            {
                "eer": test_eer,
                "auc": test_auc,
                "tar_at_far_1e3": test_tar1e3,
                "tar_at_far_1e4": test_tar1e4,
                "tar_at_far_1e5": test_tar1e5,
                "threshold_at_eer": thresh,
                "best_epoch": best_epoch,
                "num_classes_arc": num_classes_arc,
                "model_config": model_config,
                "split_config": {
                    "seed": args.seed,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "test_ratio": args.test_ratio,
                },
                "train_config": {
                    "embedding_dim": args.embedding_dim,
                    "label_smoothing": args.label_smoothing,
                    "triplet_weight": args.triplet_weight,
                    "triplet_weight_start": args.triplet_weight_start,
                    "triplet_warmup_epochs": args.triplet_warmup_epochs,
                    "triplet_margin": args.triplet_margin,
                    "triplet_mining": args.triplet_mining,
                    "negative_queue_size": args.negative_queue_size,
                    "backbone_name": args.backbone_name,
                    "accum_steps": args.accum_steps,
                    "ema_decay": args.ema_decay,
                    "amp": use_amp,
                    "channels_last": (args.channels_last and DEVICE.type == "cuda"),
                    "prefetch_factor": loader_prefetch,
                    "pk_enabled": (not args.disable_pk_sampler),
                    "pk_classes": args.pk_classes,
                    "pk_samples": args.pk_samples,
                    "steps_per_epoch": args.steps_per_epoch,
                },
            },
            f,
            indent=2,
        )

    log.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
