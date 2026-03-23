"""
PVTree pipeline for phase1:
1) Generate synthetic palm vein set (CCO-like growth).
2) Pre-train Swin + ArcFace on synthetic data.
3) Fine-tune on real TongJi ROI with ArcFace (+ triplet optional).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_phase1_dir = Path(__file__).resolve().parent
if str(_phase1_dir) not in sys.path:
    sys.path.insert(0, str(_phase1_dir))

import train as base_train
from metrics import (
    GALLERY_SCORE_MODES,
    compute_auc,
    compute_eer,
    compute_fnmr_at_fmr,
    compute_rank1_accuracy,
    compute_similarity_matrix,
    compute_tar_at_far,
    compute_template_rank1_accuracy,
    compute_template_similarity_scores,
)
from model import (
    BACKBONE_CHOICES,
    CenterLoss,
    build_model,
    freeze_backbone_stages,
    get_embedding_output_dim,
    unfreeze_all,
)
from tta_utils import parse_tta_variants_arg
from pvtree_data import (
    DEFAULT_EVAL_TRANSFORM,
    DEFAULT_TRAIN_TRANSFORM,
    build_eval_transform,
    build_real_finetune_datasets,
    build_synthetic_pretrain_datasets,
    build_train_transform,
    collect_subject_images,
)
from pvtree_synthetic import PVTreeSynthConfig, generate_synthetic_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("phase1_pvtree")


def _cleanup_cuda() -> None:
    """Aggressively free GPU memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PathLabelDataset(Dataset):
    """
    Minimal dataset wrapper using explicit (path, label) tuples.
    Keeps `.samples` for PK sampler compatibility.
    """

    def __init__(self, samples: list[tuple[str, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        image = self.transform(image)
        return image, int(label)


class ContrastivePathLabelDataset(PathLabelDataset):
    """
    Return two augmented views per image for supervised contrastive learning.
    """

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        view1 = self.transform(image)
        view2 = self.transform(image)
        return (view1, view2), int(label)


def set_device(force_gpu: bool) -> torch.device:
    if force_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--force-gpu was enabled but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def build_loader(dataset, batch_size: int, workers: int, shuffle: bool, drop_last: bool, prefetch_factor: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def count_genuine_pairs(dataset) -> int:
    counts: dict[int, int] = {}
    for _, label in getattr(dataset, "samples", []):
        label = int(label)
        counts[label] = counts.get(label, 0) + 1
    return int(sum(n * (n - 1) // 2 for n in counts.values() if n >= 2))


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    arcface: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    correct_top5 = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            emb = model(images)
        emb = emb.float()

        logits_margin = arcface(emb, labels)
        loss = criterion(logits_margin, labels)

        logits = arcface.forward_no_margin(emb)
        pred = logits.argmax(dim=1)
        topk = min(5, logits.size(1))
        top5 = logits.topk(topk, dim=1).indices

        bsz = images.size(0)
        total_loss += float(loss.item()) * bsz
        total += bsz
        correct += int(pred.eq(labels).sum().item())
        correct_top5 += int(top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item())

    return total_loss / max(total, 1), correct / max(total, 1), correct_top5 / max(total, 1)


def load_pretrained_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model_state = model.state_dict()

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    # Backward compatibility: older checkpoints used
    # embedding = [Linear, BatchNorm1d] without the leading Dropout layer.
    if "embedding.0.weight" in cleaned and "embedding.1.weight" in model_state:
        legacy_linear = cleaned.get("embedding.0.weight")
        current_linear = model_state.get("embedding.1.weight")
        if (
            legacy_linear is not None
            and current_linear is not None
            and tuple(legacy_linear.shape) == tuple(current_linear.shape)
        ):
            remapped = {}
            for key, value in cleaned.items():
                if key.startswith("embedding.0."):
                    remapped[key.replace("embedding.0.", "embedding.1.", 1)] = value
                elif key.startswith("embedding.1."):
                    remapped[key.replace("embedding.1.", "embedding.2.", 1)] = value
                else:
                    remapped[key] = value
            cleaned = remapped
            log.info("Detected legacy embedding head format in %s; remapped embedding keys", checkpoint_path)

    filtered = {}
    skipped_shape = []
    for key, value in cleaned.items():
        if key not in model_state:
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered[key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    log.info(
        "Loaded pretrain weights from %s | loaded=%d missing=%d unexpected=%d skipped_shape=%d",
        checkpoint_path,
        len(filtered),
        len(missing),
        len(unexpected),
        len(skipped_shape),
    )
    if skipped_shape:
        preview = ", ".join(
            f"{name}: ckpt={src} model={dst}" for name, src, dst in skipped_shape[:5]
        )
        log.info("Skipped incompatible tensors: %s", preview)


def select_best(
    best_eer: float,
    best_tar1e4: float,
    best_tar1e5: float,
    cur_eer: float,
    cur_tar1e4: float,
    cur_tar1e5: float,
    best_by: str = "eer",
) -> bool:
    eps_eer = 1e-5
    eps_tar = 1e-4

    if best_by == "eer":
        if cur_eer < best_eer - eps_eer:
            return True
        if abs(cur_eer - best_eer) <= eps_eer and cur_tar1e5 > best_tar1e5 + eps_tar:
            return True
        if abs(cur_eer - best_eer) <= eps_eer and abs(cur_tar1e5 - best_tar1e5) <= eps_tar and cur_tar1e4 > best_tar1e4 + eps_tar:
            return True
        return False

    if best_by == "tar1e5":
        if cur_tar1e5 > best_tar1e5 + eps_tar:
            return True
        if abs(cur_tar1e5 - best_tar1e5) <= eps_tar and cur_eer < best_eer - eps_eer:
            return True
        if abs(cur_tar1e5 - best_tar1e5) <= eps_tar and abs(cur_eer - best_eer) <= eps_eer and cur_tar1e4 > best_tar1e4 + eps_tar:
            return True
        return False

    if cur_tar1e4 > best_tar1e4 + eps_tar:
        return True
    if abs(cur_tar1e4 - best_tar1e4) <= eps_tar and cur_tar1e5 > best_tar1e5 + eps_tar:
        return True
    if abs(cur_tar1e4 - best_tar1e4) <= eps_tar and abs(cur_tar1e5 - best_tar1e5) <= eps_tar and cur_eer < best_eer - eps_eer:
        return True
    return False


def build_hybrid_train_dataset(
    real_train_ds,
    synthetic_root: str | Path,
    seed: int,
    max_synth_subjects: int = 0,
    max_synth_samples_per_subject: int = 0,
    transform=None,
) -> tuple[PathLabelDataset, int, dict]:
    """
    Build hybrid train set = real-train samples + synthetic samples.

    Synthetic subjects are mapped to new label IDs appended after real labels.
    """
    if transform is None:
        transform = DEFAULT_TRAIN_TRANSFORM

    real_samples = list(real_train_ds.samples)
    if not real_samples:
        raise RuntimeError("Real train dataset is empty; cannot build hybrid dataset")

    synth_subject_images = collect_subject_images(synthetic_root)
    subjects = sorted(synth_subject_images.keys())
    rng = random.Random(seed + 13)
    rng.shuffle(subjects)

    if max_synth_subjects > 0:
        subjects = subjects[: max_synth_subjects]
    if not subjects:
        raise RuntimeError("No synthetic subjects selected for hybrid training")

    # Real train labels are 0..num_real_classes-1 by construction.
    real_num_classes = len({int(lab) for _, lab in real_samples})
    label_offset = real_num_classes

    hybrid_samples = list(real_samples)
    synth_image_count = 0

    for sid, subject in enumerate(subjects):
        label = label_offset + sid
        paths = list(synth_subject_images[subject])
        if max_synth_samples_per_subject > 0 and len(paths) > max_synth_samples_per_subject:
            sub_rng = random.Random(seed + sid * 9973 + 101)
            paths = sub_rng.sample(paths, max_synth_samples_per_subject)
        for path in sorted(paths):
            hybrid_samples.append((path, label))
            synth_image_count += 1

    hybrid_ds = PathLabelDataset(hybrid_samples, transform=transform)
    info = {
        "enabled": True,
        "synthetic_root": str(Path(synthetic_root)),
        "synthetic_subjects_selected": len(subjects),
        "synthetic_images_selected": synth_image_count,
        "real_train_images": len(real_samples),
        "total_train_images": len(hybrid_samples),
        "real_num_classes": real_num_classes,
        "synthetic_num_classes": len(subjects),
        "total_num_classes": real_num_classes + len(subjects),
        "max_synth_subjects": int(max_synth_subjects),
        "max_synth_samples_per_subject": int(max_synth_samples_per_subject),
    }
    return hybrid_ds, len(subjects), info


def main() -> None:
    parser = argparse.ArgumentParser(description="PVTree synthetic pretrain + real finetune")
    parser.add_argument("--real-data", type=str, default=r"C:\AI_PROJECT\PALM_PRINT\data\after\TongJi_ROI_224x224")
    parser.add_argument("--synthetic-root", type=str, default=r"C:\AI_PROJECT\PALM_PRINT\data\after\TongJi_PVTreeSynth_Paper_224x224")
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument("--channels-last", action="store_true")

    parser.add_argument("--backbone-name", type=str, default="swin_tiny_patch4_window7_224", choices=BACKBONE_CHOICES)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--arc-subcenters", type=int, default=3)
    parser.add_argument("--use-rs-patch-embed", action="store_true")
    parser.add_argument("--use-coordinate-attn", action="store_true")
    parser.add_argument("--ca-reduction", type=int, default=32)
    parser.add_argument("--no-imagenet-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-depth", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["gray3", "raw_clahe_gabor"],
        default="gray3",
        help="gray3 = grayscale repeated to 3 channels; raw_clahe_gabor = raw + CLAHE + Gabor response",
    )

    parser.add_argument("--skip-synth-generation", action="store_true")
    parser.add_argument("--overwrite-synth-root", action="store_true")
    parser.add_argument("--overwrite-synth-subject", action="store_true")
    parser.add_argument("--synth-subjects", type=int, default=8000)
    parser.add_argument("--synth-samples-per-subject", type=int, default=8)
    parser.add_argument("--synth-workers", type=int, default=8)
    parser.add_argument("--synth-mode", type=str, choices=["pattern", "nir_fallback"], default="nir_fallback")
    parser.add_argument("--synth-branch-points", type=int, default=70)
    parser.add_argument("--synth-no-bezier", action="store_true")
    parser.add_argument("--hybrid-phase1", action="store_true")
    parser.add_argument("--hybrid-synth-subjects", type=int, default=0)
    parser.add_argument("--hybrid-synth-samples-per-subject", type=int, default=0)

    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--pretrain-checkpoint", type=str, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-weight-decay", type=float, default=0.05)
    parser.add_argument("--pretrain-val-ratio", type=float, default=0.1)
    parser.add_argument("--pretrain-arc-s", type=float, default=26.0)
    parser.add_argument("--pretrain-arc-m", type=float, default=0.2)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["subject_independent", "within_subject"],
        default="subject_independent",
        help="subject_independent = unseen identities in val/test; within_subject = split images inside each subject",
    )
    parser.add_argument(
        "--verification-mode",
        type=str,
        choices=["pairwise", "train_gallery"],
        default="pairwise",
        help="pairwise = compare samples within val/test; train_gallery = compare val/test probes against templates built from train split",
    )
    parser.add_argument(
        "--gallery-score-mode",
        type=str,
        choices=GALLERY_SCORE_MODES,
        default="mean_template",
        help="How to aggregate multiple gallery images per subject in train_gallery mode",
    )
    parser.add_argument(
        "--gallery-topk",
        type=int,
        default=2,
        help="Top-k used when gallery-score-mode=topk_mean",
    )
    parser.add_argument(
        "--gallery-probe-znorm",
        action="store_true",
        help="Apply per-probe z-normalization across subject scores in train_gallery mode",
    )
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-freeze-epochs", type=int, default=10)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--finetune-weight-decay", type=float, default=0.05)
    parser.add_argument("--finetune-lr-freeze", type=float, default=3e-4)
    parser.add_argument("--finetune-lr-backbone", type=float, default=3e-5)
    parser.add_argument("--finetune-lr-head", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1,
                        help="Run validation every N epochs (always evaluates last epoch)")
    parser.add_argument("--lr-min-factor", type=float, default=0.01,
                        help="Minimum LR as fraction of initial (prevents LR=0)")

    parser.add_argument("--arc-s-start", type=float, default=16.0)
    parser.add_argument("--arc-s-end", type=float, default=30.0)
    parser.add_argument("--arc-m-start", type=float, default=0.0)
    parser.add_argument("--arc-m-end", type=float, default=0.35)
    parser.add_argument("--arc-warmup-epochs", type=int, default=8)

    parser.add_argument("--triplet-weight", type=float, default=0.40)
    parser.add_argument("--triplet-weight-start", type=float, default=0.05)
    parser.add_argument("--triplet-warmup-epochs", type=int, default=12)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--triplet-mining", type=str, choices=["hard", "semi-hard"], default="semi-hard")
    parser.add_argument("--negative-queue-size", type=int, default=4096)

    parser.add_argument("--disable-pk-sampler", action="store_true")
    parser.add_argument("--pk-classes", type=int, default=12)
    parser.add_argument("--pk-samples", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.1,
                        help="Dropout before embedding projection")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation for val/test evaluation")
    parser.add_argument(
        "--tta-variants",
        type=str,
        default=None,
        help="Comma-separated TTA variants (e.g., hflip,vflip,rot10,rot-10). Original view is always included.",
    )
    parser.add_argument("--cosine-restarts", type=int, default=0,
                        help="Number of warm restarts for CosineAnnealingWarmRestarts (0 = standard cosine)")
    parser.add_argument("--best-by", type=str, choices=["eer", "tar1e4", "tar1e5"], default="eer")
    parser.add_argument("--save-topk-eer", type=int, default=10)
    parser.add_argument("--use-clahe", action="store_true",
                        help="Apply CLAHE vein enhancement before augmentation")
    parser.add_argument("--strong-aug", action="store_true",
                        help="Enable heavy spatial augmentations (elastic/grid/optical distortion)")
    parser.add_argument("--minimal-aug", action="store_true",
                        help="Use minimal NIR augmentation: resize + affine(+/-10deg, translate) + brightness/contrast")
    parser.add_argument("--veintr-aug", action="store_true",
                        help="Use VeinTr-style augmentation: rotation/translation/Gaussian noise/Random erasing")
    parser.add_argument("--train-rotate-deg", type=float, default=None,
                        help="Override train-time rotation range in degrees")
    parser.add_argument("--min-images-per-subject", type=int, default=0,
                        help="Filter out subjects with fewer than N images (set >= pk_samples)")
    parser.add_argument("--min-eval-images-per-subject", type=int, default=2,
                        help="Minimum images reserved for each val/test split in within_subject mode")
    parser.add_argument("--center-loss-weight", type=float, default=0.0,
                        help="Weight for center loss (0 = disabled, try 0.003-0.01)")
    parser.add_argument("--center-loss-lr", type=float, default=0.5,
                        help="Learning rate for center loss centers")
    parser.add_argument("--supcon", action="store_true",
                        help="Enable supervised contrastive loss (2 augmented views per image)")
    parser.add_argument("--supcon-weight", type=float, default=0.15,
                        help="Weight for supervised contrastive loss")
    parser.add_argument("--supcon-temp", type=float, default=0.07,
                        help="Temperature for supervised contrastive loss")
    parser.add_argument("--supcon-stop-epoch", type=int, default=0,
                        help="Disable SupCon after this epoch (0 = keep for all epochs)")

    args = parser.parse_args()

    tta_variants = parse_tta_variants_arg(args.tta_variants)
    if tta_variants and not args.tta:
        args.tta = True

    supcon_weight = args.supcon_weight if args.supcon else 0.0
    if args.supcon and supcon_weight <= 0:
        log.warning("supcon enabled but supcon_weight <= 0. Disabling SupCon.")
        args.supcon = False
        supcon_weight = 0.0
    args.eval_every = max(1, int(args.eval_every))

    # CRITICAL: Validate warmup epochs don't exceed finetune epochs
    # This is a common mistake that prevents the model from reaching target margins
    if args.arc_warmup_epochs > args.finetune_epochs:
        log.warning(
            "arc_warmup_epochs (%d) > finetune_epochs (%d). Auto-adjusting to %d",
            args.arc_warmup_epochs,
            args.finetune_epochs,
            max(1, args.finetune_epochs - 2),
        )
        args.arc_warmup_epochs = max(1, args.finetune_epochs - 2)

    if args.triplet_warmup_epochs > args.finetune_epochs:
        log.warning(
            "triplet_warmup_epochs (%d) > finetune_epochs (%d). Auto-adjusting to %d",
            args.triplet_warmup_epochs,
            args.finetune_epochs,
            max(1, args.finetune_epochs - 2),
        )
        args.triplet_warmup_epochs = max(1, args.finetune_epochs - 2)

    if args.finetune_epochs < 15:
        log.warning(
            "finetune_epochs=%d is very low. For best EER, use 30-70 epochs.",
            args.finetune_epochs,
        )

    if args.veintr_aug and args.strong_aug:
        log.warning("veintr-aug overrides strong-aug. Disabling strong-aug.")
        args.strong_aug = False

    if args.veintr_aug and args.minimal_aug:
        log.warning("veintr-aug overrides minimal-aug. Disabling minimal-aug.")
        args.minimal_aug = False

    if args.minimal_aug and args.strong_aug:
        log.warning("minimal-aug and strong-aug were both enabled. Disabling strong-aug and keeping minimal-aug.")
        args.strong_aug = False
    if args.verification_mode == "train_gallery" and args.split_mode != "within_subject":
        raise ValueError("verification-mode=train_gallery requires split-mode=within_subject")
    args.gallery_topk = max(1, int(args.gallery_topk))

    out_dir = Path(args.output or f"runs/phase1_pvtree_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = set_device(args.force_gpu)
    base_train.DEVICE = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_amp = bool(args.amp and device.type == "cuda")

    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    synth_root = Path(args.synthetic_root)
    synth_meta = None
    if not args.skip_synth_generation:
        synth_cfg = PVTreeSynthConfig(
            num_subjects=args.synth_subjects,
            samples_per_subject=args.synth_samples_per_subject,
            synth_mode=args.synth_mode,
            branch_points=args.synth_branch_points,
            use_bezier_crease=(not args.synth_no_bezier),
        )
        synth_meta = generate_synthetic_dataset(
            output_root=synth_root,
            cfg=synth_cfg,
            seed=args.seed,
            workers=args.synth_workers,
            overwrite_root=args.overwrite_synth_root,
            overwrite_subject=args.overwrite_synth_subject,
        )
        with (out_dir / "synthetic_generation.json").open("w", encoding="utf-8") as f:
            json.dump(synth_meta, f, indent=2)

    pretrain_ckpt: Path | None = Path(args.pretrain_checkpoint).resolve() if args.pretrain_checkpoint else None

    if pretrain_ckpt is None and not args.skip_pretrain:
        pretrain_train_transform = build_train_transform(
            use_clahe=False,
            use_albu=False,
            image_size=args.image_size,
            minimal=False,
            input_mode=args.input_mode,
            veintr_aug=False,
            rotate_deg=args.train_rotate_deg,
        )
        pretrain_eval_transform = build_eval_transform(
            use_clahe=False,
            image_size=args.image_size,
            input_mode=args.input_mode,
        )
        synth_train_ds, synth_val_ds, synth_classes = build_synthetic_pretrain_datasets(
            root=synth_root,
            val_ratio=args.pretrain_val_ratio,
            seed=args.seed,
            train_transform=pretrain_train_transform,
            eval_transform=pretrain_eval_transform,
        )
        synth_train_loader = build_loader(
            synth_train_ds,
            batch_size=args.pretrain_batch_size,
            workers=args.workers,
            shuffle=True,
            drop_last=True,
            prefetch_factor=args.prefetch_factor,
        )
        synth_val_loader = build_loader(
            synth_val_ds,
            batch_size=args.pretrain_batch_size,
            workers=args.workers,
            shuffle=False,
            drop_last=False,
            prefetch_factor=args.prefetch_factor,
        )

        # Pretrain uses SIMPLE config: 1 subcenter, no label smoothing, no emb
        # dropout. These regularisations are finetune-only — applying them here
        # prevents the model from learning ANY synthetic features (v3 lesson).
        pre_model, pre_arc = build_model(
            num_classes=synth_classes,
            backbone_name=args.backbone_name,
            embedding_dim=args.embedding_dim,
            drop_path_rate=args.drop_path_rate,
            arc_s=args.pretrain_arc_s,
            arc_m=args.pretrain_arc_m,
            arc_subcenters=1,
            use_rs_patch_embed=args.use_rs_patch_embed,
            use_coordinate_attention=args.use_coordinate_attn,
            ca_reduction=args.ca_reduction,
            pretrained=(not args.no_imagenet_pretrained),
            emb_dropout=0.0,
            image_size=args.image_size,
            transformer_dim=args.transformer_dim,
            transformer_depth=args.transformer_depth,
            transformer_heads=args.transformer_heads,
            transformer_mlp_ratio=args.transformer_mlp_ratio,
            transformer_dropout=args.transformer_dropout,
        )
        pre_model = pre_model.to(device)
        pre_arc = pre_arc.to(device)
        if args.channels_last and device.type == "cuda":
            pre_model = pre_model.to(memory_format=torch.channels_last)

        pre_criterion = nn.CrossEntropyLoss()  # no label smoothing for pretrain
        pre_optimizer = torch.optim.AdamW(
            list(pre_model.parameters()) + list(pre_arc.parameters()),
            lr=args.pretrain_lr,
            weight_decay=args.pretrain_weight_decay,
        )
        pre_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pre_optimizer, T_max=max(1, args.pretrain_epochs))
        pre_scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

        pre_hist = []
        best_acc = -1.0
        for epoch in range(1, args.pretrain_epochs + 1):
            tr_loss, tr_acc, tr_top5, _ = base_train.run_epoch(
                pre_model,
                pre_arc,
                synth_train_loader,
                pre_optimizer,
                pre_criterion,
                f"PreE{epoch}",
                triplet_weight=0.0,
                scaler=pre_scaler,
                use_amp=use_amp,
                channels_last=args.channels_last,
                accum_steps=args.accum_steps,
                supcon_weight=0.0,
                supcon_temp=args.supcon_temp,
            )
            vl_loss, vl_acc, vl_top5 = evaluate_classification(
                pre_model,
                pre_arc,
                synth_val_loader,
                pre_criterion,
                device,
                use_amp,
                args.channels_last,
            )
            pre_hist.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "train_top5": tr_top5,
                "val_loss": vl_loss,
                "val_acc": vl_acc,
                "val_top5": vl_top5,
                "lr": pre_optimizer.param_groups[0]["lr"],
            })
            log.info(
                "[Pretrain] E%03d train_acc=%.2f%% val_acc=%.2f%% val_top5=%.2f%%",
                epoch,
                tr_acc * 100,
                vl_acc * 100,
                vl_top5 * 100,
            )
            if vl_acc > best_acc:
                best_acc = vl_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": "pretrain",
                        "model": pre_model.state_dict(),
                        "arcface": pre_arc.state_dict(),
                        "val_acc": vl_acc,
                    },
                    out_dir / "pretrain_best.pt",
                )
            pre_scheduler.step()

        with (out_dir / "pretrain_history.json").open("w", encoding="utf-8") as f:
            json.dump(pre_hist, f, indent=2)
        pretrain_ckpt = out_dir / "pretrain_best.pt"

        # ---- Free pretrain objects to reclaim GPU memory ----
        del pre_model, pre_arc, pre_optimizer, pre_scheduler, pre_scaler, pre_criterion
        del synth_train_loader, synth_val_loader, synth_train_ds, synth_val_ds
        _cleanup_cuda()
        log.info("[Memory] Pretrain objects freed. VRAM available: %.0f MB",
                 torch.cuda.mem_get_info()[0] / 1024**2 if torch.cuda.is_available() else 0)

    # Build transforms (with or without CLAHE)
    finetune_train_transform = build_train_transform(
        use_clahe=args.use_clahe,
        use_albu=args.strong_aug,
        image_size=args.image_size,
        minimal=args.minimal_aug,
        input_mode=args.input_mode,
        veintr_aug=args.veintr_aug,
        rotate_deg=args.train_rotate_deg,
    )
    finetune_eval_transform = build_eval_transform(
        use_clahe=args.use_clahe,
        image_size=args.image_size,
        input_mode=args.input_mode,
    )
    if args.input_mode == "raw_clahe_gabor":
        log.info("[Input] Multi-channel vein input enabled: raw + CLAHE + Gabor")
    if args.use_clahe:
        log.info("[CLAHE] Vein enhancement enabled for train & eval transforms")
    if args.strong_aug:
        log.info("[Aug] Strong albumentations enabled (elastic/grid/optical)")
    if args.minimal_aug:
        log.info("[Aug] Minimal NIR augmentation enabled (resize + affine +/-10deg + translate + brightness/contrast)")
    if args.veintr_aug:
        log.info("[Aug] VeinTr-style augmentation enabled (+/-15deg, 10%% translate, Gaussian noise, random erasing)")
    if args.train_rotate_deg is not None:
        log.info("[Aug] Train rotation override enabled: +/-%.1f deg", args.train_rotate_deg)

    train_ds, val_ds, test_ds, num_real_classes, split_info = build_real_finetune_datasets(
        root=args.real_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_transform=finetune_train_transform,
        eval_transform=finetune_eval_transform,
        min_images_per_subject=args.min_images_per_subject,
        split_mode=args.split_mode,
        min_eval_images_per_subject=args.min_eval_images_per_subject,
    )
    num_train_classes = num_real_classes
    hybrid_info = {"enabled": False}
    gallery_ds = PathLabelDataset(list(train_ds.samples), finetune_eval_transform)
    log.info(
        "[Split] mode=%s train=%d val=%d test=%d classes=%d",
        split_info.mode,
        len(train_ds),
        len(val_ds),
        len(test_ds),
        num_real_classes,
    )
    if args.verification_mode == "pairwise":
        val_pairs = count_genuine_pairs(val_ds)
        test_pairs = count_genuine_pairs(test_ds)
        log.info("[Split] genuine pairs: val=%d test=%d", val_pairs, test_pairs)
        if val_pairs <= 0 or test_pairs <= 0:
            raise RuntimeError(
                "Validation/test split produced no genuine pairs. Increase images per subject or adjust split ratios."
            )
    else:
        log.info(
            "[Verify] mode=train_gallery gallery_samples=%d val_probes=%d test_probes=%d score_mode=%s topk=%d probe_znorm=%s",
            len(gallery_ds),
            len(val_ds),
            len(test_ds),
            args.gallery_score_mode,
            args.gallery_topk,
            args.gallery_probe_znorm,
        )

    supcon_transform = finetune_train_transform

    if args.hybrid_phase1:
        hybrid_transform = build_train_transform(
            use_clahe=False,
            use_albu=False,
            image_size=args.image_size,
            minimal=False,
            input_mode=args.input_mode,
            veintr_aug=False,
            rotate_deg=args.train_rotate_deg,
        )
        train_ds, num_synth_classes, hybrid_info = build_hybrid_train_dataset(
            real_train_ds=train_ds,
            synthetic_root=args.synthetic_root,
            seed=args.seed,
            max_synth_subjects=args.hybrid_synth_subjects,
            max_synth_samples_per_subject=args.hybrid_synth_samples_per_subject,
            transform=hybrid_transform,
        )
        supcon_transform = hybrid_transform
        num_train_classes = num_real_classes + num_synth_classes
        log.info(
            "[Hybrid] real_train_images=%d synth_images=%d total_train_images=%d classes(real=%d synth=%d total=%d)",
            hybrid_info["real_train_images"],
            hybrid_info["synthetic_images_selected"],
            hybrid_info["total_train_images"],
            hybrid_info["real_num_classes"],
            hybrid_info["synthetic_num_classes"],
            hybrid_info["total_num_classes"],
        )

    if args.supcon:
        train_ds = ContrastivePathLabelDataset(list(train_ds.samples), supcon_transform)
        if args.supcon_stop_epoch > 0:
            log.info(
                "[SupCon] enabled | weight=%.3f temp=%.3f stop_epoch=%d (2 views per image)",
                supcon_weight, args.supcon_temp, args.supcon_stop_epoch,
            )
        else:
            log.info("[SupCon] enabled | weight=%.3f temp=%.3f (2 views per image)", supcon_weight, args.supcon_temp)

    if args.disable_pk_sampler:
        train_loader = build_loader(
            train_ds,
            batch_size=args.finetune_batch_size,
            workers=args.workers,
            shuffle=True,
            drop_last=True,
            prefetch_factor=args.prefetch_factor,
        )
        train_sampler = None
    else:
        train_labels = [lab for _, lab in train_ds.samples]
        train_sampler = base_train.PKBatchSampler(
            train_labels,
            p_classes=args.pk_classes,
            k_samples=args.pk_samples,
            steps_per_epoch=args.steps_per_epoch,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.workers > 0),
            prefetch_factor=(args.prefetch_factor if args.workers > 0 else None),
        )

    gallery_loader = build_loader(gallery_ds, args.finetune_batch_size, args.workers, False, False, args.prefetch_factor)
    val_loader = build_loader(val_ds, args.finetune_batch_size, args.workers, False, False, args.prefetch_factor)
    test_loader = build_loader(test_ds, args.finetune_batch_size, args.workers, False, False, args.prefetch_factor)

    model, arcface = build_model(
        num_classes=num_train_classes,
        backbone_name=args.backbone_name,
        embedding_dim=args.embedding_dim,
        drop_path_rate=args.drop_path_rate,
        arc_s=args.arc_s_end,
        arc_m=args.arc_m_end,
        arc_subcenters=args.arc_subcenters,
        use_rs_patch_embed=args.use_rs_patch_embed,
        use_coordinate_attention=args.use_coordinate_attn,
        ca_reduction=args.ca_reduction,
        pretrained=(not args.no_imagenet_pretrained),
        emb_dropout=args.emb_dropout,
        image_size=args.image_size,
        transformer_dim=args.transformer_dim,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_dropout=args.transformer_dropout,
    )
    model = model.to(device)
    arcface = arcface.to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if pretrain_ckpt is not None and pretrain_ckpt.is_file():
        load_pretrained_model_weights(model, pretrain_ckpt, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    ema_model = base_train.ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # Center Loss setup
    center_loss_mod = None
    center_optimizer = None
    if args.center_loss_weight > 0:
        center_loss_mod = CenterLoss(num_train_classes, args.embedding_dim).to(device)
        center_optimizer = torch.optim.SGD(center_loss_mod.parameters(), lr=args.center_loss_lr)
        log.info("[CenterLoss] weight=%.4f lr=%.3f classes=%d dim=%d",
                 args.center_loss_weight, args.center_loss_lr, num_train_classes, args.embedding_dim)

    freeze_backbone_stages(model.backbone, num_stages=2)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(arcface.parameters()), lr=args.finetune_lr_freeze, weight_decay=args.finetune_weight_decay)
    scheduler = None
    approx_updates_per_epoch = 0
    if len(train_loader) > 0:
        approx_updates_per_epoch = math.ceil(len(train_loader) / max(1, args.accum_steps))
    if ema_model is not None:
        log.info(
            "[EMA] decay=%.4f validation uses EMA weights | train_batches=%d accum=%d approx_updates/epoch=%d",
            args.ema_decay,
            len(train_loader),
            args.accum_steps,
            approx_updates_per_epoch,
        )
        if args.ema_decay >= 0.9995 and approx_updates_per_epoch > 0 and approx_updates_per_epoch < 100:
            warmup_old_weight = args.ema_decay ** approx_updates_per_epoch
            log.info(
                "[EMA] First epoch still retains ~%.2f%% of warm-start weights in validation EMA; early val metrics may look flat.",
                warmup_old_weight * 100.0,
            )

    queue_dim = get_embedding_output_dim(model)
    negative_queue = base_train.HardNegativeQueue(queue_dim, args.negative_queue_size, device)

    history = []
    eer_improvements = []
    best_eer = float("inf")
    best_tar1e4 = -1.0
    best_tar1e5 = -1.0
    best_epoch = -1
    patience = 0

    for epoch in range(1, args.finetune_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        negative_queue.reset()

        if epoch == args.finetune_freeze_epochs + 1:
            unfreeze_all(model.backbone)
            # Use single LR for all parameters (matching successful baseline approach)
            # Split LR (different backbone vs head) can cause instability
            base_lr = args.finetune_lr_backbone
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": base_lr},
                    {"params": model.embedding.parameters(), "lr": args.finetune_lr_head},
                    {"params": arcface.parameters(), "lr": args.finetune_lr_head},
                ],
                weight_decay=args.finetune_weight_decay,
            )
            remaining_epochs = max(1, args.finetune_epochs - args.finetune_freeze_epochs)
            # Use CosineAnnealingLR with eta_min to prevent LR collapse to 0
            eta_min = base_lr * args.lr_min_factor
            if args.cosine_restarts > 0:
                T_0 = max(1, remaining_epochs // (args.cosine_restarts + 1))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=1, eta_min=eta_min
                )
                log.info(
                    "[Finetune] Unfreezing all. LR backbone=%.2e head=%.2e eta_min=%.2e "
                    "CosineWarmRestarts T_0=%d restarts=%d",
                    base_lr, args.finetune_lr_head, eta_min, T_0, args.cosine_restarts,
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining_epochs, eta_min=eta_min
                )
                log.info(
                    "[Finetune] Unfreezing all. LR backbone=%.2e head=%.2e eta_min=%.2e T_max=%d",
                    base_lr, args.finetune_lr_head, eta_min, remaining_epochs,
                )

        if args.arc_warmup_epochs <= 1:
            arc_s, arc_m = args.arc_s_end, args.arc_m_end
        else:
            ratio = min(1.0, (epoch - 1) / float(args.arc_warmup_epochs - 1))
            arc_s = args.arc_s_start + ratio * (args.arc_s_end - args.arc_s_start)
            arc_m = args.arc_m_start + ratio * (args.arc_m_end - args.arc_m_start)
        arcface.set_margin_scale(arc_s, arc_m)

        if args.triplet_warmup_epochs <= 1:
            tri_w = args.triplet_weight
        else:
            tri_ratio = min(1.0, (epoch - 1) / float(args.triplet_warmup_epochs - 1))
            tri_w = args.triplet_weight_start + tri_ratio * (args.triplet_weight - args.triplet_weight_start)

        # Switch to hard mining after triplet warmup for stronger gradients
        current_mining = args.triplet_mining
        if args.triplet_mining == "semi-hard" and epoch > args.triplet_warmup_epochs:
            current_mining = "hard"

        epoch_supcon_weight = supcon_weight
        if args.supcon_stop_epoch > 0 and epoch > args.supcon_stop_epoch:
            epoch_supcon_weight = 0.0

        tr_loss, tr_acc, tr_top5, tr_tri = base_train.run_epoch(
            model,
            arcface,
            train_loader,
            optimizer,
            criterion,
            f"FtE{epoch}",
            triplet_weight=tri_w,
            triplet_margin=args.triplet_margin,
            negative_queue=negative_queue,
            triplet_mining=current_mining,
            scaler=scaler,
            use_amp=use_amp,
            channels_last=args.channels_last,
            accum_steps=args.accum_steps,
            ema_model=ema_model,
            center_loss_module=center_loss_mod,
            center_loss_weight=args.center_loss_weight,
            center_optimizer=center_optimizer,
            supcon_weight=epoch_supcon_weight,
            supcon_temp=args.supcon_temp,
        )

        should_eval = (epoch % args.eval_every == 0) or (epoch == args.finetune_epochs)
        val_eer = None
        val_auc = None
        val_tar1e3 = None
        val_tar1e4 = None
        val_tar1e5 = None
        if should_eval:
            # ---- Free training caches before evaluation to avoid OOM ----
            _cleanup_cuda()

            model_eval = ema_model.module if ema_model is not None else model
            val_eer, val_auc, val_tar1e3, val_tar1e4, val_tar1e5 = base_train.compute_val_metrics(
                model_eval,
                val_loader,
                use_amp=use_amp,
                channels_last=args.channels_last,
                use_tta=args.tta,
                tta_variants=tta_variants,
                gallery_loader=(gallery_loader if args.verification_mode == "train_gallery" else None),
                gallery_score_mode=args.gallery_score_mode,
                gallery_topk=args.gallery_topk,
                gallery_probe_znorm=args.gallery_probe_znorm,
            )
        else:
            model_eval = ema_model.module if ema_model is not None else model

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_top5": tr_top5,
            "train_triplet_loss": tr_tri,
            "train_supcon_weight": epoch_supcon_weight,
            "val_eer": val_eer,
            "val_auc": val_auc,
            "val_tar_1e3": val_tar1e3,
            "val_tar_1e4": val_tar1e4,
            "val_tar_1e5": val_tar1e5,
            "lr": optimizer.param_groups[0]["lr"],
            "arc_s": arc_s,
            "arc_m": arc_m,
            "triplet_weight": tri_w,
        })

        if should_eval:
            log.info(
                "[Finetune] E%03d acc=%.2f%% top5=%.2f%% val_eer=%.2f%% tar1e4=%.2f%% tar1e5=%.2f%% supcon_w=%.3f",
                epoch,
                tr_acc * 100,
                tr_top5 * 100,
                val_eer * 100,
                val_tar1e4 * 100,
                val_tar1e5 * 100,
                epoch_supcon_weight,
            )

            if val_eer < best_eer - 1e-5:
                eer_improvements.append(
                    {
                        "epoch": epoch,
                        "val_eer": val_eer,
                        "val_tar_1e4": val_tar1e4,
                        "val_tar_1e5": val_tar1e5,
                        "val_auc": val_auc,
                    }
                )

            if select_best(
                best_eer=best_eer,
                best_tar1e4=best_tar1e4,
                best_tar1e5=best_tar1e5,
                cur_eer=val_eer,
                cur_tar1e4=val_tar1e4,
                cur_tar1e5=val_tar1e5,
                best_by=args.best_by,
            ):
                best_eer = val_eer
                best_tar1e4 = val_tar1e4
                best_tar1e5 = val_tar1e5
                best_epoch = epoch
                patience = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model_eval.state_dict(),
                        "arcface": arcface.state_dict(),
                        "val_eer": val_eer,
                        "val_auc": val_auc,
                        "val_tar_1e3": val_tar1e3,
                        "val_tar_1e4": val_tar1e4,
                        "val_tar_1e5": val_tar1e5,
                        "num_classes_arc": num_train_classes,
                        "split_config": {
                            "seed": args.seed,
                            "split_mode": split_info.mode,
                            "train_ratio": args.train_ratio,
                            "val_ratio": args.val_ratio,
                            "test_ratio": args.test_ratio,
                            "train_subjects": split_info.train_subjects,
                            "val_subjects": split_info.val_subjects,
                            "test_subjects": split_info.test_subjects,
                        },
                        "model_config": {
                            "backbone_name": args.backbone_name,
                            "embedding_dim": args.embedding_dim,
                            "drop_path_rate": args.drop_path_rate,
                            "arc_s": args.arc_s_end,
                            "arc_m": args.arc_m_end,
                            "arc_subcenters": args.arc_subcenters,
                            "use_rs_patch_embed": args.use_rs_patch_embed,
                            "use_coordinate_attention": args.use_coordinate_attn,
                            "ca_reduction": args.ca_reduction,
                            "image_size": args.image_size,
                            "transformer_dim": args.transformer_dim,
                            "transformer_depth": args.transformer_depth,
                            "transformer_heads": args.transformer_heads,
                            "transformer_mlp_ratio": args.transformer_mlp_ratio,
                            "transformer_dropout": args.transformer_dropout,
                        },
                        "data_config": {
                            "verification_mode": args.verification_mode,
                            "gallery_score_mode": args.gallery_score_mode,
                            "gallery_topk": args.gallery_topk,
                            "gallery_probe_znorm": args.gallery_probe_znorm,
                            "use_clahe": args.use_clahe,
                            "input_mode": args.input_mode,
                            "strong_aug": args.strong_aug,
                            "minimal_aug": args.minimal_aug,
                            "veintr_aug": args.veintr_aug,
                            "train_rotate_deg": args.train_rotate_deg,
                            "min_images_per_subject": args.min_images_per_subject,
                            "min_eval_images_per_subject": args.min_eval_images_per_subject,
                            "tta": args.tta,
                            "tta_variants": tta_variants,
                            "supcon": args.supcon,
                            "supcon_weight": supcon_weight,
                            "supcon_temp": args.supcon_temp,
                            "supcon_stop_epoch": args.supcon_stop_epoch,
                        },
                    },
                    out_dir / "best.pt",
                )
            else:
                patience += 1
        else:
            log.info(
                "[Finetune] E%03d acc=%.2f%% top5=%.2f%% supcon_w=%.3f | val skipped (eval_every=%d)",
                epoch,
                tr_acc * 100,
                tr_top5 * 100,
                epoch_supcon_weight,
                args.eval_every,
            )

        if scheduler is not None:
            scheduler.step()

        if patience >= args.patience:
            break

    if best_epoch < 0:
        raise RuntimeError("No best checkpoint saved during fine-tuning")

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    if args.verification_mode == "train_gallery":
        if args.tta:
            emb_gallery, lab_gallery = base_train.extract_embeddings_tta(
                model,
                gallery_loader,
                use_amp=use_amp,
                channels_last=args.channels_last,
                tta_variants=tta_variants,
            )
            emb_test, lab_test = base_train.extract_embeddings_tta(
                model,
                test_loader,
                use_amp=use_amp,
                channels_last=args.channels_last,
                tta_variants=tta_variants,
            )
        else:
            emb_gallery, lab_gallery = base_train.extract_embeddings(
                model, gallery_loader, use_amp=use_amp, channels_last=args.channels_last
            )
            emb_test, lab_test = base_train.extract_embeddings(
                model, test_loader, use_amp=use_amp, channels_last=args.channels_last
            )
        scores, is_genuine = compute_template_similarity_scores(
            gallery_embeddings=emb_gallery,
            gallery_labels=lab_gallery,
            probe_embeddings=emb_test,
            probe_labels=lab_test,
            score_mode=args.gallery_score_mode,
            topk=args.gallery_topk,
            probe_znorm=args.gallery_probe_znorm,
        )
        test_rank1 = compute_template_rank1_accuracy(
            gallery_embeddings=emb_gallery,
            gallery_labels=lab_gallery,
            probe_embeddings=emb_test,
            probe_labels=lab_test,
            score_mode=args.gallery_score_mode,
            topk=args.gallery_topk,
        )
    elif args.tta:
        emb_test, lab_test = base_train.extract_embeddings_tta(
            model,
            test_loader,
            use_amp=use_amp,
            channels_last=args.channels_last,
            tta_variants=tta_variants,
        )
        scores, is_genuine = compute_similarity_matrix(emb_test, lab_test)
        test_rank1 = compute_rank1_accuracy(emb_test, lab_test)
    else:
        emb_test, lab_test = base_train.extract_embeddings(model, test_loader, use_amp=use_amp, channels_last=args.channels_last)
        scores, is_genuine = compute_similarity_matrix(emb_test, lab_test)
        test_rank1 = compute_rank1_accuracy(emb_test, lab_test)
    test_eer, test_thr = compute_eer(scores, is_genuine)
    test_auc = compute_auc(scores, is_genuine)
    test_tar1e2 = compute_tar_at_far(scores, is_genuine, 1e-2)
    test_tar1e3 = compute_tar_at_far(scores, is_genuine, 1e-3)
    test_tar1e4 = compute_tar_at_far(scores, is_genuine, 1e-4)
    test_tar1e5 = compute_tar_at_far(scores, is_genuine, 1e-5)
    test_fnmr_fmr100 = compute_fnmr_at_fmr(scores, is_genuine, 1e-2)
    test_fnmr_fmr1000 = compute_fnmr_at_fmr(scores, is_genuine, 1e-3)

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "eer_improvements.json").open("w", encoding="utf-8") as f:
        json.dump(eer_improvements, f, indent=2)

    topk = max(1, int(args.save_topk_eer))
    best_epochs_by_eer = sorted(
        (
            {
                "epoch": item["epoch"],
                "val_eer": item["val_eer"],
                "val_tar_1e4": item["val_tar_1e4"],
                "val_tar_1e5": item.get("val_tar_1e5"),
                "val_auc": item["val_auc"],
            }
            for item in history
            if item["val_eer"] is not None
        ),
        key=lambda x: x["val_eer"],
    )[:topk]
    with (out_dir / "best_epochs_by_eer.json").open("w", encoding="utf-8") as f:
        json.dump(best_epochs_by_eer, f, indent=2)

    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_eer": best_eer,
                "best_val_tar_1e4": best_tar1e4,
                "best_val_tar_1e5": best_tar1e5,
                "test": {
                    "eer": test_eer,
                    "auc": test_auc,
                    "rank1": test_rank1,
                    "tar_at_far_1e2": test_tar1e2,
                    "tar_at_far_1e3": test_tar1e3,
                    "tar_at_far_1e4": test_tar1e4,
                    "tar_at_far_1e5": test_tar1e5,
                    "fnmr_at_fmr100": test_fnmr_fmr100,
                    "fnmr_at_fmr1000": test_fnmr_fmr1000,
                    "threshold_at_eer": test_thr,
                },
                "pretrain_checkpoint": str(pretrain_ckpt) if pretrain_ckpt else None,
                "synthetic_meta": synth_meta,
                "hybrid_info": hybrid_info,
                "args": vars(args),
                "best_by": args.best_by,
                "best_epochs_by_eer": best_epochs_by_eer,
            },
            f,
            indent=2,
        )

    log.info(
        "Best epoch (%s): %d | val EER=%.2f%% val TAR@1e-4=%.2f%%",
        args.best_by,
        best_epoch,
        best_eer * 100,
        best_tar1e4 * 100,
    )
    log.info(
        "Best val TAR@1e-5=%.2f%% | Test EER=%.2f%% Rank1=%.2f%% AUC=%.4f TAR@1e-4=%.2f%% TAR@1e-5=%.2f%%",
        best_tar1e5 * 100,
        test_eer * 100,
        test_rank1 * 100,
        test_auc,
        test_tar1e4 * 100,
        test_tar1e5 * 100,
    )
    log.info("Saved to %s", out_dir)


if __name__ == "__main__":
    main()
