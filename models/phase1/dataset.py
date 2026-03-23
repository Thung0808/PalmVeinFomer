"""
Phase 1: ROI-free Global Texture Stream — Dataset
Subject-independent split: 70% train / 15% val / 15% test
Cấu trúc: {root}/{subject_id}/{image.png}
"""

from __future__ import annotations

import os
import random
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def collect_subject_images(root: str | Path) -> dict[str, list[str]]:
    """
    Thu thập ảnh theo subject.
    Returns: {subject_id: [path1, path2, ...]}
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root không tồn tại: {root}")

    subject_images: dict[str, list[str]] = {}
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        subject_id = subdir.name
        paths = []
        for f in subdir.iterdir():
            if f.suffix.lower() in exts:
                paths.append(str(f))
        paths.sort()
        if paths:
            subject_images[subject_id] = paths

    if not subject_images:
        raise RuntimeError(f"Không tìm thấy ảnh trong: {root}")
    return subject_images


def subject_independent_split(
    subject_images: dict[str, list[str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """
    Chia theo subject — không để cùng subject ở cả train và test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    subjects = sorted(subject_images.keys())
    random.seed(seed)
    random.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_subs = subjects[:n_train]
    val_subs = subjects[n_train : n_train + n_val]
    test_subs = subjects[n_train + n_val :]

    train_data = {s: subject_images[s] for s in train_subs}
    val_data = {s: subject_images[s] for s in val_subs}
    test_data = {s: subject_images[s] for s in test_subs}

    return train_data, val_data, test_data


# ─── Transforms (NIR, grayscale → 3ch for pretrained) ───────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224, scale=(0.92, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomRotation(12),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PalmVeinDataset(Dataset):
    """
    Dataset NIR palm với subject-independent split.
    Flatten: [(path, label), ...] với label là index trong label_map.
    """

    def __init__(
        self,
        subject_paths: dict[str, list[str]],
        label_map: dict[str, int],
        transform: transforms.Compose | None = None,
    ):
        self.transform = transform or VAL_TRANSFORM
        self.samples: list[tuple[str, int]] = []
        self.label_map = label_map

        for subject_id, paths in subject_paths.items():
            lab = label_map.get(subject_id)
            if lab is None:
                continue
            for p in paths:
                self.samples.append((p, lab))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


def build_datasets(
    root: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[PalmVeinDataset, PalmVeinDataset, PalmVeinDataset, dict[str, int]]:
    """
    Xây dựng train/val/test từ root, trả về (train_ds, val_ds, test_ds, label_map).
    label_map: subject_id → int (chỉ cho train subjects, val/test dùng map chung).
    """
    subject_images = collect_subject_images(root)
    train_data, val_data, test_data = subject_independent_split(
        subject_images, train_ratio, val_ratio, test_ratio, seed
    )

    all_subjects = sorted(subject_images.keys())
    label_map = {s: i for i, s in enumerate(all_subjects)}

    train_ds = PalmVeinDataset(train_data, label_map, transform=TRAIN_TRANSFORM)
    val_ds = PalmVeinDataset(val_data, label_map, transform=VAL_TRANSFORM)
    test_ds = PalmVeinDataset(test_data, label_map, transform=VAL_TRANSFORM)

    return train_ds, val_ds, test_ds, label_map
