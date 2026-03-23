"""
Phase 1: Post-training evaluation
- Extract embeddings từ test set
- ROC curve
- EER, TAR@FAR=1e-3, 1e-4
- t-SNE visualization
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

_phase1_dir = Path(__file__).resolve().parent
if str(_phase1_dir) not in sys.path:
    sys.path.insert(0, str(_phase1_dir))

import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from metrics import (
    GALLERY_SCORE_MODES,
    compute_auc,
    compute_eer,
    compute_fnmr_at_fmr,
    compute_rank1_accuracy,
    compute_roc,
    compute_similarity_matrix,
    compute_tar_at_far,
    compute_template_rank1_accuracy,
    compute_template_similarity_scores,
)
from model import build_model
from pvtree_data import build_eval_transform, build_real_finetune_datasets
from tta_utils import build_tta_plan, forward_tta, parse_tta_variants_arg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    use_tta: bool = False,
    tta_plan=None,
    use_amp: bool = False,
    channels_last: bool = False,
):
    model.eval()
    embs, labs = [], []
    amp_enabled = use_amp and DEVICE.type == "cuda"
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        if channels_last and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        if use_tta:
            emb = forward_tta(model, images, tta_plan or [], amp_enabled=amp_enabled)
        else:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_enabled):
                emb = model(images)
        embs.append(emb.float().cpu().numpy())
        labs.append(labels.numpy())
    return np.vstack(embs), np.concatenate(labs)


def _parse_float_list(spec: str | None) -> list[float]:
    if spec is None:
        return []
    parts = [p.strip() for p in spec.replace(";", ",").split(",")]
    parts = [p for p in parts if p]
    return [float(p) for p in parts]


def _prepare_latency_batches(
    loader,
    max_batches: int,
    channels_last: bool,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(DEVICE, non_blocking=True)
        if channels_last and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        batches.append(images)
        if len(batches) >= max_batches:
            break
    return batches


@torch.no_grad()
def _benchmark_latency(
    model,
    batches: Iterable[torch.Tensor],
    use_amp: bool,
    tta_plan,
    warmup: int,
    iters: int,
) -> dict:
    amp_enabled = use_amp and DEVICE.type == "cuda"
    device_type = DEVICE.type
    times = []
    total_images = 0
    for idx, images in enumerate(batches):
        if idx < warmup:
            if tta_plan:
                _ = forward_tta(model, images, tta_plan, amp_enabled=amp_enabled)
            else:
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                    _ = model(images)
            continue

        if device_type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if tta_plan:
            _ = forward_tta(model, images, tta_plan, amp_enabled=amp_enabled)
        else:
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
                _ = model(images)
        if device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        batch_time = t1 - t0
        batch_size = int(images.shape[0])
        times.append((batch_time, batch_size))
        total_images += batch_size
        if len(times) >= iters:
            break

    if not times:
        return {
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "images_per_sec": None,
            "batches": 0,
            "images": 0,
        }

    per_image_ms = [(t / b) * 1000.0 for t, b in times]
    mean_ms = float(np.mean(per_image_ms))
    p50_ms = float(np.percentile(per_image_ms, 50))
    p90_ms = float(np.percentile(per_image_ms, 90))
    p95_ms = float(np.percentile(per_image_ms, 95))
    total_time = sum(t for t, _ in times)
    images_per_sec = float(total_images / total_time) if total_time > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p95_ms": p95_ms,
        "images_per_sec": images_per_sec,
        "batches": len(times),
        "images": total_images,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\AI_PROJECT\PALM_PRINT\data\after\raw_224x224px",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Save ROC + t-SNE here")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None, help="Override split seed (default: from checkpoint or 42)")
    parser.add_argument("--train-ratio", type=float, default=None, help="Override train split ratio")
    parser.add_argument("--val-ratio", type=float, default=None, help="Override val split ratio")
    parser.add_argument("--test-ratio", type=float, default=None, help="Override test split ratio")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["subject_independent", "within_subject"],
        default=None,
        help="Override split mode saved in checkpoint",
    )
    parser.add_argument("--min-images-per-subject", type=int, default=None, help="Override minimum images per subject filter")
    parser.add_argument(
        "--min-eval-images-per-subject",
        type=int,
        default=None,
        help="Override minimum images reserved for each val/test split in within_subject mode",
    )
    parser.add_argument(
        "--verification-mode",
        type=str,
        choices=["pairwise", "train_gallery"],
        default=None,
        help="Override verification protocol saved in checkpoint",
    )
    parser.add_argument(
        "--gallery-score-mode",
        type=str,
        choices=GALLERY_SCORE_MODES,
        default=None,
        help="Override gallery subject aggregation mode saved in checkpoint",
    )
    parser.add_argument(
        "--gallery-topk",
        type=int,
        default=None,
        help="Override top-k when gallery-score-mode=topk_mean",
    )
    parser.add_argument("--gallery-probe-znorm", dest="gallery_probe_znorm", action="store_true")
    parser.add_argument("--no-gallery-probe-znorm", dest="gallery_probe_znorm", action="store_false")
    parser.add_argument("--tta", dest="tta", action="store_true", help="Override checkpoint to enable TTA")
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Override checkpoint to disable TTA")
    parser.add_argument(
        "--tta-variants",
        type=str,
        default=None,
        help="Comma-separated TTA variants (e.g., hflip,vflip,rot10,rot-10). Original view is always included.",
    )
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--tsne-max-samples", type=int, default=2000, help="Subsample for t-SNE if large")
    parser.add_argument("--no-plots", action="store_true", help="Skip ROC/t-SNE plots")
    parser.add_argument("--eval-rotate-deg", type=str, default=None, help="Comma-separated fixed rotations (deg) for probe/test images")
    parser.add_argument("--amp", action="store_true", help="Use AMP for inference")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last tensor format on CUDA")
    parser.add_argument("--measure-latency", action="store_true", help="Benchmark inference latency on test set")
    parser.add_argument("--latency-warmup", type=int, default=10, help="Warmup batches before timing")
    parser.add_argument("--latency-iters", type=int, default=50, help="Timed batches for latency")
    parser.add_argument("--latency-batch-size", type=int, default=None, help="Override batch size for latency benchmark")
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Bắt buộc dùng CUDA, nếu không có sẽ báo lỗi",
    )
    parser.set_defaults(gallery_probe_znorm=None)
    parser.set_defaults(tta=None)
    args = parser.parse_args()

    global DEVICE
    if args.force_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("force-gpu được bật nhưng torch.cuda.is_available() = False. Kiểm tra driver / CUDA.")
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    arcface_state = ckpt["arcface"]
    model_cfg = ckpt.get("model_config", {})

    arc_subcenters = int(model_cfg.get("arc_subcenters", 1))
    if "num_classes_arc" in ckpt:
        num_classes = int(ckpt["num_classes_arc"])
    else:
        arc_rows = int(arcface_state["weight"].shape[0])
        num_classes = arc_rows // max(1, arc_subcenters)

    model, _ = build_model(
        num_classes=num_classes,
        backbone_name=str(model_cfg.get("backbone_name", "swin_tiny_patch4_window7_224")),
        embedding_dim=int(model_cfg.get("embedding_dim", 512)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.2)),
        arc_s=float(model_cfg.get("arc_s", 30.0)),
        arc_m=float(model_cfg.get("arc_m", 0.35)),
        arc_subcenters=arc_subcenters,
        use_rs_patch_embed=bool(model_cfg.get("use_rs_patch_embed", False)),
        use_coordinate_attention=bool(model_cfg.get("use_coordinate_attention", False)),
        ca_reduction=int(model_cfg.get("ca_reduction", 32)),
        pretrained=False,
        image_size=int(model_cfg.get("image_size", 224)),
        transformer_dim=int(model_cfg.get("transformer_dim", 256)),
        transformer_depth=int(model_cfg.get("transformer_depth", 3)),
        transformer_heads=int(model_cfg.get("transformer_heads", 4)),
        transformer_mlp_ratio=float(model_cfg.get("transformer_mlp_ratio", 4.0)),
        transformer_dropout=float(model_cfg.get("transformer_dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)

    split_cfg = ckpt.get("split_config", {})
    split_seed = args.seed if args.seed is not None else split_cfg.get("seed", 42)
    split_mode = args.split_mode or str(split_cfg.get("split_mode", "subject_independent"))
    train_ratio = args.train_ratio if args.train_ratio is not None else split_cfg.get("train_ratio", 0.70)
    val_ratio = args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.15)
    test_ratio = args.test_ratio if args.test_ratio is not None else split_cfg.get("test_ratio", 0.15)
    data_cfg = ckpt.get("data_config", {})
    verification_mode = args.verification_mode or str(data_cfg.get("verification_mode", "pairwise"))
    gallery_score_mode = args.gallery_score_mode or str(data_cfg.get("gallery_score_mode", "mean_template"))
    gallery_topk = int(
        args.gallery_topk
        if args.gallery_topk is not None
        else data_cfg.get("gallery_topk", 2)
    )
    gallery_probe_znorm = (
        bool(args.gallery_probe_znorm)
        if args.gallery_probe_znorm is not None
        else bool(data_cfg.get("gallery_probe_znorm", False))
    )
    use_tta = bool(args.tta) if args.tta is not None else bool(data_cfg.get("tta", False))
    tta_variants = parse_tta_variants_arg(args.tta_variants)
    if tta_variants is None:
        cfg_variants = data_cfg.get("tta_variants")
        if isinstance(cfg_variants, list):
            tta_variants = [str(v) for v in cfg_variants]
        elif isinstance(cfg_variants, str):
            tta_variants = parse_tta_variants_arg(cfg_variants)
    if tta_variants and args.tta is None:
        use_tta = True
    if use_tta and not tta_variants:
        tta_variants = ["hflip"]
    tta_plan = build_tta_plan(tta_variants)

    angles = _parse_float_list(args.eval_rotate_deg)
    if not angles:
        angles = [None]
    save_plots = (len(angles) == 1) and (not args.no_plots)

    out_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.data import DataLoader

    results = []
    for angle in angles:
        gallery_transform = build_eval_transform(
            use_clahe=bool(data_cfg.get("use_clahe", False)),
            image_size=int(model_cfg.get("image_size", 224)),
            input_mode=str(data_cfg.get("input_mode", "gray3")),
        )
        eval_transform = build_eval_transform(
            use_clahe=bool(data_cfg.get("use_clahe", False)),
            image_size=int(model_cfg.get("image_size", 224)),
            input_mode=str(data_cfg.get("input_mode", "gray3")),
            rotate_deg=angle,
        )

        train_ds, _, test_ds, _, _ = build_real_finetune_datasets(
            root=args.data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
            train_transform=gallery_transform,
            eval_transform=eval_transform,
            min_images_per_subject=int(
                args.min_images_per_subject
                if args.min_images_per_subject is not None
                else data_cfg.get("min_images_per_subject", 0)
            ),
            split_mode=split_mode,
            min_eval_images_per_subject=int(
                args.min_eval_images_per_subject
                if args.min_eval_images_per_subject is not None
                else data_cfg.get("min_eval_images_per_subject", 2)
            ),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),
        )
        gallery_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),
        )

        if verification_mode == "train_gallery":
            gallery_emb, gallery_lab = extract_embeddings(
                model,
                gallery_loader,
                use_tta=use_tta,
                tta_plan=tta_plan,
                use_amp=args.amp,
                channels_last=args.channels_last,
            )
            emb, lab = extract_embeddings(
                model,
                test_loader,
                use_tta=use_tta,
                tta_plan=tta_plan,
                use_amp=args.amp,
                channels_last=args.channels_last,
            )
            scores, is_genuine = compute_template_similarity_scores(
                gallery_embeddings=gallery_emb,
                gallery_labels=gallery_lab,
                probe_embeddings=emb,
                probe_labels=lab,
                score_mode=gallery_score_mode,
                topk=gallery_topk,
                probe_znorm=gallery_probe_znorm,
            )
            rank1 = compute_template_rank1_accuracy(
                gallery_embeddings=gallery_emb,
                gallery_labels=gallery_lab,
                probe_embeddings=emb,
                probe_labels=lab,
                score_mode=gallery_score_mode,
                topk=gallery_topk,
            )
        else:
            emb, lab = extract_embeddings(
                model,
                test_loader,
                use_tta=use_tta,
                tta_plan=tta_plan,
                use_amp=args.amp,
                channels_last=args.channels_last,
            )
            scores, is_genuine = compute_similarity_matrix(emb, lab)
            rank1 = compute_rank1_accuracy(emb, lab)

        eer, thresh = compute_eer(scores, is_genuine)
        auc = compute_auc(scores, is_genuine)
        tar1e2 = compute_tar_at_far(scores, is_genuine, 1e-2)
        tar1e3 = compute_tar_at_far(scores, is_genuine, 1e-3)
        tar1e4 = compute_tar_at_far(scores, is_genuine, 1e-4)
        fnmr100 = compute_fnmr_at_fmr(scores, is_genuine, 1e-2)
        fnmr1000 = compute_fnmr_at_fmr(scores, is_genuine, 1e-3)
        fpr, tpr, _ = compute_roc(scores, is_genuine)

        angle_label = "none" if angle is None else f"{angle:g}"
        print("=" * 60)
        print("TEST SET METRICS")
        print("=" * 60)
        print(f"  Rotation(deg): {angle_label}")
        print(f"  Verification:  {verification_mode}")
        print(f"  TTA:           {use_tta} | variants={tta_variants or []}")
        if verification_mode == "train_gallery":
            print(f"  Gallery mode:  {gallery_score_mode} (topk={gallery_topk}, probe_znorm={gallery_probe_znorm})")
        print(f"  EER:           {eer*100:.2f}%")
        print(f"  Rank-1:        {rank1*100:.2f}%")
        print(f"  AUC:           {auc:.4f}")
        print(f"  TAR@FAR=1e-2:  {tar1e2*100:.2f}%")
        print(f"  TAR@FAR=1e-3:  {tar1e3*100:.2f}%")
        print(f"  TAR@FAR=1e-4:  {tar1e4*100:.2f}%")
        print(f"  FNMR@FMR100:   {fnmr100*100:.2f}%")
        print(f"  FNMR@FMR1000:  {fnmr1000*100:.2f}%")
        print(f"  Threshold@EER: {thresh:.4f}")

        results.append(
            {
                "rotation_deg": angle,
                "eer": eer,
                "rank1": rank1,
                "auc": auc,
                "tar_at_far_1e2": tar1e2,
                "tar_at_far_1e3": tar1e3,
                "tar_at_far_1e4": tar1e4,
                "fnmr_at_fmr100": fnmr100,
                "fnmr_at_fmr1000": fnmr1000,
                "threshold_at_eer": thresh,
            }
        )

        if save_plots:
            # ROC
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, "b-", linewidth=2, label="ROC")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("FAR (False Accept Rate)")
            plt.ylabel("TAR (True Accept Rate)")
            plt.title(f"ROC — EER={eer*100:.2f}% AUC={auc:.4f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            roc_path = out_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            print(f"  ROC saved: {roc_path}")

            # t-SNE
            n = len(emb)
            if n > args.tsne_max_samples:
                idx = np.random.choice(n, args.tsne_max_samples, replace=False)
                emb_tsne = emb[idx]
                lab_tsne = lab[idx]
            else:
                emb_tsne, lab_tsne = emb, lab

            perplexity = min(args.tsne_perplexity, max(5, len(emb_tsne) // 4))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            emb_2d = tsne.fit_transform(emb_tsne)

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=lab_tsne, cmap="tab20", alpha=0.6, s=10)
            plt.colorbar(scatter, label="Subject ID")
            plt.title("Embedding t-SNE (test set)")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.tight_layout()
            tsne_path = out_dir / "tsne_embeddings.png"
            plt.savefig(tsne_path, dpi=150)
            plt.close()
            print(f"  t-SNE saved: {tsne_path}")

    if len(results) > 1:
        sweep_path = out_dir / "rotation_sweep.json"
        with sweep_path.open("w", encoding="utf-8") as f:
            json.dump({"rotations": results}, f, indent=2)
        print(f"Rotation sweep saved: {sweep_path}")

    metrics_path = out_dir / "evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(Path(args.checkpoint)),
                "data": args.data,
                "verification_mode": verification_mode,
                "gallery_score_mode": gallery_score_mode,
                "gallery_topk": gallery_topk,
                "gallery_probe_znorm": gallery_probe_znorm,
                "tta": use_tta,
                "tta_variants": tta_variants,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Evaluation metrics saved: {metrics_path}")

    if args.measure_latency:
        latency_batch = args.latency_batch_size or args.batch_size
        latency_transform = build_eval_transform(
            use_clahe=bool(data_cfg.get("use_clahe", False)),
            image_size=int(model_cfg.get("image_size", 224)),
            input_mode=str(data_cfg.get("input_mode", "gray3")),
        )
        latency_train, _, latency_test, _, _ = build_real_finetune_datasets(
            root=args.data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=split_seed,
            train_transform=latency_transform,
            eval_transform=latency_transform,
            min_images_per_subject=int(
                args.min_images_per_subject
                if args.min_images_per_subject is not None
                else data_cfg.get("min_images_per_subject", 0)
            ),
            split_mode=split_mode,
            min_eval_images_per_subject=int(
                args.min_eval_images_per_subject
                if args.min_eval_images_per_subject is not None
                else data_cfg.get("min_eval_images_per_subject", 2)
            ),
        )
        latency_loader = DataLoader(
            latency_test,
            batch_size=latency_batch,
            shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),
        )

        max_batches = max(1, args.latency_warmup + args.latency_iters)
        batches = _prepare_latency_batches(latency_loader, max_batches=max_batches, channels_last=args.channels_last)

        base_stats = _benchmark_latency(
            model,
            batches,
            use_amp=args.amp,
            tta_plan=[],
            warmup=args.latency_warmup,
            iters=args.latency_iters,
        )
        tta_stats = None
        if use_tta and tta_plan:
            tta_stats = _benchmark_latency(
                model,
                batches,
                use_amp=args.amp,
                tta_plan=tta_plan,
                warmup=args.latency_warmup,
                iters=args.latency_iters,
            )

        print("=" * 60)
        print("LATENCY BENCHMARK (model forward only)")
        print("=" * 60)
        print(f"  Batch size:    {latency_batch}")
        print(f"  AMP:           {args.amp}")
        print(f"  ChannelsLast:  {args.channels_last}")
        print(f"  Base mean ms:  {base_stats['mean_ms']:.3f} | p95={base_stats['p95_ms']:.3f} | ips={base_stats['images_per_sec']:.1f}")
        if tta_stats is not None:
            print(f"  TTA variants:  {tta_variants or []} (views={1 + len(tta_plan)})")
            print(f"  TTA mean ms:   {tta_stats['mean_ms']:.3f} | p95={tta_stats['p95_ms']:.3f} | ips={tta_stats['images_per_sec']:.1f}")

        latency_path = out_dir / "latency_metrics.json"
        with latency_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "batch_size": latency_batch,
                    "amp": args.amp,
                    "channels_last": args.channels_last,
                    "base": base_stats,
                    "tta": tta_stats,
                    "tta_variants": tta_variants,
                },
                f,
                indent=2,
            )
        print(f"Latency metrics saved: {latency_path}")


if __name__ == "__main__":
    main()
