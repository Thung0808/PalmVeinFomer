"""
Biometric verification metrics: EER, TAR@FAR, ROC
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

GALLERY_SCORE_MODES = (
    "mean_template",
    "max",
    "topk_mean",
    "mean_max_fusion",
)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _build_template_matrix(
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    g_norm = _normalize_rows(gallery_embeddings)
    g_labels = np.asarray(gallery_labels, dtype=np.int32)
    template_labels = np.array(sorted({int(x) for x in g_labels}), dtype=np.int32)
    templates = []
    for label in template_labels:
        template = g_norm[g_labels == label].mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-8)
        templates.append(template.astype(np.float32))
    return np.stack(templates, axis=0), template_labels


def _aggregate_subject_scores(
    sample_scores: np.ndarray,
    gallery_labels: np.ndarray,
    template_labels: np.ndarray,
    score_mode: str,
    topk: int,
) -> np.ndarray:
    gallery_labels = np.asarray(gallery_labels, dtype=np.int32)
    subject_scores = np.empty((sample_scores.shape[0], len(template_labels)), dtype=np.float32)

    for idx, label in enumerate(template_labels):
        subject_view = sample_scores[:, gallery_labels == label]
        if subject_view.shape[1] == 0:
            raise ValueError(f"No gallery samples found for label={label}")
        if score_mode == "max":
            subject_scores[:, idx] = subject_view.max(axis=1)
            continue
        if score_mode == "topk_mean":
            k = min(max(1, int(topk)), subject_view.shape[1])
            if k == subject_view.shape[1]:
                topk_scores = subject_view
            else:
                kth = subject_view.shape[1] - k
                topk_scores = np.partition(subject_view, kth=kth, axis=1)[:, -k:]
            subject_scores[:, idx] = topk_scores.mean(axis=1)
            continue
        raise ValueError(f"Unsupported score_mode for sample aggregation: {score_mode}")

    return subject_scores


def compute_gallery_subject_score_matrix(
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    probe_embeddings: np.ndarray,
    score_mode: str = "mean_template",
    topk: int = 2,
    probe_znorm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a probe-vs-subject similarity matrix for gallery verification.
    score_mode:
      - mean_template: cosine to one normalized mean template per subject
      - max: best gallery image score per subject
      - topk_mean: mean of top-k gallery image scores per subject
      - mean_max_fusion: 0.5 * mean_template + 0.5 * max
    probe_znorm:
      - normalize each probe row by its subject-score mean/std to stabilize thresholds
    """
    if len(gallery_embeddings) == 0 or len(probe_embeddings) == 0:
        raise ValueError("gallery_embeddings and probe_embeddings must be non-empty")
    if score_mode not in GALLERY_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    p_norm = _normalize_rows(probe_embeddings)
    template_matrix, template_labels = _build_template_matrix(gallery_embeddings, gallery_labels)

    if score_mode == "mean_template":
        subject_scores = np.dot(p_norm, template_matrix.T)
    elif score_mode == "mean_max_fusion":
        template_scores = np.dot(p_norm, template_matrix.T)
        g_norm = _normalize_rows(gallery_embeddings)
        sample_scores = np.dot(p_norm, g_norm.T)
        max_scores = _aggregate_subject_scores(
            sample_scores,
            gallery_labels,
            template_labels,
            score_mode="max",
            topk=topk,
        )
        subject_scores = 0.5 * (template_scores + max_scores)
    else:
        g_norm = _normalize_rows(gallery_embeddings)
        sample_scores = np.dot(p_norm, g_norm.T)
        subject_scores = _aggregate_subject_scores(
            sample_scores,
            gallery_labels,
            template_labels,
            score_mode=score_mode,
            topk=topk,
        )

    if probe_znorm:
        row_mean = subject_scores.mean(axis=1, keepdims=True)
        row_std = subject_scores.std(axis=1, keepdims=True)
        subject_scores = (subject_scores - row_mean) / (row_std + 1e-6)

    return subject_scores.astype(np.float32), template_labels.astype(np.int32)


def compute_similarity_matrix(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    embeddings: (N, D), normalized
    labels: (N,) subject ids
    Returns: scores, is_genuine (1 = same subject, 0 = different)
    """
    n = len(embeddings)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(emb_norm, emb_norm.T)

    scores = []
    is_genuine = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(sim[i, j])
            is_genuine.append(1 if labels[i] == labels[j] else 0)

    return np.array(scores, dtype=np.float32), np.array(is_genuine, dtype=np.int32)


def compute_template_similarity_scores(
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    probe_embeddings: np.ndarray,
    probe_labels: np.ndarray,
    score_mode: str = "mean_template",
    topk: int = 2,
    probe_znorm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Template/gallery-based verification scores flattened as pair labels.
    """
    subject_scores, template_labels = compute_gallery_subject_score_matrix(
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        probe_embeddings=probe_embeddings,
        score_mode=score_mode,
        topk=topk,
        probe_znorm=probe_znorm,
    )
    probe_labels = np.asarray(probe_labels, dtype=np.int32)
    scores = subject_scores.reshape(-1)
    is_genuine = (probe_labels[:, None] == template_labels[None, :]).astype(np.int32).reshape(-1)
    return scores.astype(np.float32), is_genuine.astype(np.int32)


def compute_eer(scores: np.ndarray, is_genuine: np.ndarray) -> tuple[float, float]:
    """
    Equal Error Rate: FAR = FRR.
    Returns: (EER, threshold_at_EER)
    Uses linear interpolation around the FAR/FRR crossing for higher precision.
    """
    fpr, tpr, thresholds = roc_curve(is_genuine, scores, pos_label=1)
    fnr = 1 - tpr
    diff = fpr - fnr

    crossing_idx = np.where(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
    if crossing_idx.size > 0:
        idx = int(crossing_idx[0])
        x1 = float(diff[idx])
        x2 = float(diff[idx + 1])
        if abs(x2 - x1) < 1e-12:
            weight = 0.5
        else:
            weight = float(np.clip(-x1 / (x2 - x1), 0.0, 1.0))
        eer = float(fpr[idx] + weight * (fpr[idx + 1] - fpr[idx]))
        threshold = float(thresholds[idx] + weight * (thresholds[idx + 1] - thresholds[idx]))
        return eer, threshold

    eer_idx = int(np.nanargmin(np.absolute(diff)))
    eer = (float(fpr[eer_idx]) + float(fnr[eer_idx])) / 2.0
    return eer, float(thresholds[eer_idx])


def compute_tar_at_far(
    scores: np.ndarray,
    is_genuine: np.ndarray,
    far_target: float,
) -> float:
    """
    TAR (True Accept Rate) at given FAR (False Accept Rate).
    Threshold = (1-far_target) quantile of impostor scores (high scores = accept).
    """
    n_impostor = int(np.sum(is_genuine == 0))
    n_genuine = int(np.sum(is_genuine == 1))
    if n_impostor == 0 or n_genuine == 0:
        return 1.0

    imp_scores = np.sort(scores[is_genuine == 0])
    gen_scores = scores[is_genuine == 1]
    idx = min(int((1 - far_target) * n_impostor), n_impostor - 1)
    threshold = imp_scores[idx]
    tar = np.mean(gen_scores >= threshold)
    return float(tar)


def compute_fnmr_at_fmr(
    scores: np.ndarray,
    is_genuine: np.ndarray,
    fmr_target: float,
) -> float:
    """FNMR at a target FMR threshold."""
    return float(1.0 - compute_tar_at_far(scores, is_genuine, fmr_target))


def compute_rank1_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Rank-1 identification accuracy via leave-one-out nearest-neighbor on embeddings.
    """
    if len(embeddings) <= 1:
        return 1.0

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(emb_norm, emb_norm.T)
    np.fill_diagonal(sim, -np.inf)
    pred_idx = np.argmax(sim, axis=1)
    pred_labels = labels[pred_idx]
    return float(np.mean(pred_labels == labels))


def compute_template_rank1_accuracy(
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    probe_embeddings: np.ndarray,
    probe_labels: np.ndarray,
    score_mode: str = "mean_template",
    topk: int = 2,
) -> float:
    """
    Rank-1 using the same gallery scoring mode as verification.
    """
    if len(gallery_embeddings) == 0 or len(probe_embeddings) == 0:
        return 1.0

    sim, template_labels = compute_gallery_subject_score_matrix(
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        probe_embeddings=probe_embeddings,
        score_mode=score_mode,
        topk=topk,
        probe_znorm=False,
    )
    probe_labels = np.asarray(probe_labels, dtype=np.int32)
    pred_labels = template_labels[np.argmax(sim, axis=1)]
    return float(np.mean(pred_labels == probe_labels))


def compute_roc(scores: np.ndarray, is_genuine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns fpr, tpr, thresholds."""
    return roc_curve(is_genuine, scores, pos_label=1)


def compute_auc(scores: np.ndarray, is_genuine: np.ndarray) -> float:
    """ROC AUC (higher = better)."""
    return float(roc_auc_score(is_genuine, scores))
