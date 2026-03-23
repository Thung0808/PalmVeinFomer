from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    import albumentations as A
    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _ensure_grayscale_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _apply_clahe_array(
    arr: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(arr)


class CLAHETransform:
    """Apply CLAHE on grayscale PIL image to enhance vein contrast."""
    def __init__(self, clip_limit: float = 3.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = _ensure_grayscale_array(img)
        arr = _apply_clahe_array(arr, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        return Image.fromarray(arr, mode="L")


class GaussianNoiseTransform:
    """Add Gaussian noise on tensor images to improve robustness on raw NIR capture."""

    def __init__(self, std: float = 0.03, p: float = 0.5) -> None:
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0 or torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RawClaheGaborRGBTransform:
    """
    Build a 3-channel vein-enhanced image:
    channel 0 = raw grayscale,
    channel 1 = CLAHE-enhanced grayscale,
    channel 2 = max Gabor response on CLAHE image.
    """

    def __init__(
        self,
        clip_limit: float = 3.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        gabor_kernel_size: int = 15,
        gabor_sigma: float = 4.0,
        gabor_lambda: float = 8.0,
        gabor_gamma: float = 0.5,
    ) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self.kernels = [
            cv2.getGaborKernel(
                (gabor_kernel_size, gabor_kernel_size),
                gabor_sigma,
                theta,
                gabor_lambda,
                gabor_gamma,
                0,
                ktype=cv2.CV_32F,
            )
            for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        raw = _ensure_grayscale_array(img)
        clahe = _apply_clahe_array(raw, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
        clahe_f = clahe.astype(np.float32) / 255.0
        responses = [np.abs(cv2.filter2D(clahe_f, cv2.CV_32F, kernel)) for kernel in self.kernels]
        gabor = np.max(np.stack(responses, axis=0), axis=0)
        gabor = cv2.normalize(gabor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        stacked = np.stack([raw, clahe, gabor], axis=2)
        return Image.fromarray(stacked, mode="RGB")


class AlbumentationsWrapper:
    """Wrap albumentations spatial transforms for PIL images."""
    def __init__(self, albu_transform):
        self.albu = albu_transform

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        result = self.albu(image=arr)["image"]
        mode = img.mode
        return Image.fromarray(result, mode=mode)


class FixedRotation:
    """Deterministic rotation for evaluation robustness tests."""
    def __init__(self, degrees: float) -> None:
        self.degrees = float(degrees)

    def __call__(self, img: Image.Image) -> Image.Image:
        return TF.rotate(img, self.degrees, interpolation=InterpolationMode.BILINEAR, fill=0)


def _build_albu_spatial() -> AlbumentationsWrapper | None:
    if not _HAS_ALBU:
        return None
    return AlbumentationsWrapper(
        A.Compose([
            A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.2),
        ])
    )


def build_train_transform(
    use_clahe: bool = False,
    use_albu: bool = False,
    image_size: int = 224,
    minimal: bool = False,
    input_mode: str = "gray3",
    veintr_aug: bool = False,
    gaussian_noise_std: float = 0.03,
    random_erasing_p: float = 0.15,
    rotate_deg: float | None = None,
) -> transforms.Compose:
    pre: list = []
    if input_mode == "raw_clahe_gabor":
        pre.append(RawClaheGaborRGBTransform(clip_limit=3.0, tile_grid_size=(8, 8)))
    else:
        if use_clahe:
            pre.append(CLAHETransform(clip_limit=3.0, tile_grid_size=(8, 8)))
        pre.append(transforms.Grayscale(num_output_channels=3))

    default_rotate_deg = float(rotate_deg) if rotate_deg is not None else (15.0 if veintr_aug else 10.0 if minimal else 15.0)

    if veintr_aug:
        return transforms.Compose(
            pre
            + [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomAffine(
                    degrees=default_rotate_deg,
                    translate=(0.10, 0.10),
                    scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ToTensor(),
                GaussianNoiseTransform(std=gaussian_noise_std, p=0.65),
                transforms.RandomErasing(p=max(0.2, random_erasing_p), scale=(0.02, 0.12), value=0.0),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if minimal:
        return transforms.Compose(
            pre
            + [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomAffine(
                    degrees=default_rotate_deg,
                    translate=(0.05, 0.05),
                    scale=(0.97, 1.03),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.18, contrast=0.18),
                transforms.ToTensor(),
                GaussianNoiseTransform(std=gaussian_noise_std * 0.75, p=0.45),
                transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.08), value=0.0),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    spatial: list = [
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(default_rotate_deg),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
    ]

    albu = _build_albu_spatial() if use_albu else None
    if albu is not None:
        spatial.insert(2, albu)  # after rotation, before flip

    color: list = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]

    tensor_ops: list = [
        transforms.ToTensor(),
        GaussianNoiseTransform(std=gaussian_noise_std, p=0.35),
        transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.10), value=0.0),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(pre + spatial + color + tensor_ops)


def build_eval_transform(
    use_clahe: bool = False,
    image_size: int = 224,
    input_mode: str = "gray3",
    rotate_deg: float | None = None,
) -> transforms.Compose:
    ops: list = []
    if input_mode == "raw_clahe_gabor":
        ops.append(RawClaheGaborRGBTransform(clip_limit=3.0, tile_grid_size=(8, 8)))
    else:
        if use_clahe:
            ops.append(CLAHETransform(clip_limit=3.0, tile_grid_size=(8, 8)))
        ops.append(transforms.Grayscale(num_output_channels=3))
    if rotate_deg is not None and abs(float(rotate_deg)) > 1e-6:
        ops.append(FixedRotation(float(rotate_deg)))
    ops.extend([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


# Legacy defaults (no CLAHE, no heavy albumentations)
DEFAULT_TRAIN_TRANSFORM = build_train_transform(
    use_clahe=False,
    use_albu=False,
    image_size=224,
    minimal=False,
    input_mode="gray3",
)
DEFAULT_EVAL_TRANSFORM = build_eval_transform(use_clahe=False, image_size=224, input_mode="gray3")


@dataclass
class SplitInfo:
    mode: str
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


class SubjectImageDataset(Dataset):
    def __init__(
        self,
        subject_to_paths: dict[str, list[str]],
        label_map: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        for subject, paths in subject_to_paths.items():
            if subject not in label_map:
                continue
            label = int(label_map[subject])
            for path in paths:
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("L")
        image = self.transform(image)
        return image, label


def collect_subject_images(
    root: str | Path,
    min_images_per_subject: int = 0,
) -> dict[str, list[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    subject_images: dict[str, list[str]] = {}
    skipped = 0

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        paths = [str(p) for p in sorted(subdir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
        if paths:
            if min_images_per_subject > 0 and len(paths) < min_images_per_subject:
                skipped += 1
                continue
            subject_images[subdir.name] = paths

    if not subject_images:
        raise RuntimeError(f"No images found under: {root}")

    if skipped > 0:
        import logging
        logging.getLogger("pvtree_data").info(
            "Filtered out %d subjects with < %d images", skipped, min_images_per_subject
        )

    return subject_images


def split_subjects(
    subject_images: dict[str, list[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], SplitInfo]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    subjects = sorted(subject_images.keys())
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]

    train_data = {s: subject_images[s] for s in train_subjects}
    val_data = {s: subject_images[s] for s in val_subjects}
    test_data = {s: subject_images[s] for s in test_subjects}

    info = SplitInfo(
        mode="subject_independent",
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
    )
    return train_data, val_data, test_data, info


def _allocate_within_subject_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_eval_images_per_subject: int = 2,
) -> tuple[int, int, int]:
    raw = [total * train_ratio, total * val_ratio, total * test_ratio]
    counts = [int(math.floor(x)) for x in raw]
    eval_min = max(1, int(min_eval_images_per_subject))
    min_counts = [
        1 if train_ratio > 0 and total > 0 else 0,
        eval_min if val_ratio > 0 and total >= (eval_min + 2) else (1 if val_ratio > 0 and total >= 3 else 0),
        eval_min if test_ratio > 0 and total >= (eval_min + 2) else (1 if test_ratio > 0 and total >= 3 else 0),
    ]
    counts = [max(c, m) for c, m in zip(counts, min_counts)]

    while sum(counts) > total:
        reducible = [i for i in range(3) if counts[i] > min_counts[i]]
        if not reducible:
            break
        idx = max(reducible, key=lambda i: (counts[i] - raw[i], counts[i], 1 if i == 0 else 0))
        counts[idx] -= 1

    fractions = [raw[i] - math.floor(raw[i]) for i in range(3)]
    while sum(counts) < total:
        idx = max(range(3), key=lambda i: (fractions[i], raw[i], -counts[i]))
        counts[idx] += 1

    return counts[0], counts[1], counts[2]


def split_images_within_subjects(
    subject_images: dict[str, list[str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_eval_images_per_subject: int = 2,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], SplitInfo]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    rng = random.Random(seed)
    train_data: dict[str, list[str]] = {}
    val_data: dict[str, list[str]] = {}
    test_data: dict[str, list[str]] = {}
    subjects = sorted(subject_images.keys())

    for subject in subjects:
        items = list(subject_images[subject])
        rng.shuffle(items)
        n_train, n_val, n_test = _allocate_within_subject_counts(
            total=len(items),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            min_eval_images_per_subject=min_eval_images_per_subject,
        )
        train_end = n_train
        val_end = n_train + n_val
        train_data[subject] = sorted(items[:train_end])
        val_data[subject] = sorted(items[train_end:val_end])
        test_data[subject] = sorted(items[val_end:val_end + n_test])

    info = SplitInfo(
        mode="within_subject",
        train_subjects=subjects,
        val_subjects=subjects,
        test_subjects=subjects,
    )
    return train_data, val_data, test_data, info


def split_samples_within_subjects(
    subject_images: dict[str, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    rng = random.Random(seed)
    train_data: dict[str, list[str]] = {}
    val_data: dict[str, list[str]] = {}

    for subject, paths in sorted(subject_images.items()):
        items = list(paths)
        rng.shuffle(items)
        if len(items) <= 1:
            train_data[subject] = items
            val_data[subject] = []
            continue

        n_val = max(1, int(round(len(items) * val_ratio)))
        if n_val >= len(items):
            n_val = len(items) - 1

        val_data[subject] = sorted(items[:n_val])
        train_data[subject] = sorted(items[n_val:])

    return train_data, val_data


def build_real_finetune_datasets(
    root: str | Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    train_transform: transforms.Compose | None = None,
    eval_transform: transforms.Compose | None = None,
    min_images_per_subject: int = 0,
    split_mode: str = "subject_independent",
    min_eval_images_per_subject: int = 2,
) -> tuple[SubjectImageDataset, SubjectImageDataset, SubjectImageDataset, int, SplitInfo]:
    train_transform = train_transform or DEFAULT_TRAIN_TRANSFORM
    eval_transform = eval_transform or DEFAULT_EVAL_TRANSFORM

    subject_images = collect_subject_images(root, min_images_per_subject=min_images_per_subject)
    if split_mode not in {"subject_independent", "within_subject"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    if split_mode == "within_subject":
        train_data, val_data, test_data, split_info = split_images_within_subjects(
            subject_images=subject_images,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            min_eval_images_per_subject=min_eval_images_per_subject,
        )
    else:
        train_data, val_data, test_data, split_info = split_subjects(
            subject_images=subject_images,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    all_subjects = sorted(subject_images.keys())
    eval_label_map = {s: i for i, s in enumerate(all_subjects)}

    if split_mode == "within_subject":
        train_label_map = eval_label_map
    else:
        train_subjects = sorted(train_data.keys())
        train_label_map = {s: i for i, s in enumerate(train_subjects)}

    train_ds = SubjectImageDataset(train_data, train_label_map, transform=train_transform)
    val_ds = SubjectImageDataset(val_data, eval_label_map, transform=eval_transform)
    test_ds = SubjectImageDataset(test_data, eval_label_map, transform=eval_transform)

    return train_ds, val_ds, test_ds, len(train_label_map), split_info


def build_synthetic_pretrain_datasets(
    root: str | Path,
    val_ratio: float,
    seed: int,
    train_transform: transforms.Compose | None = None,
    eval_transform: transforms.Compose | None = None,
) -> tuple[SubjectImageDataset, SubjectImageDataset, int]:
    train_transform = train_transform or DEFAULT_TRAIN_TRANSFORM
    eval_transform = eval_transform or DEFAULT_EVAL_TRANSFORM

    subject_images = collect_subject_images(root)
    train_data, val_data = split_samples_within_subjects(
        subject_images=subject_images,
        val_ratio=val_ratio,
        seed=seed,
    )

    subjects = sorted(subject_images.keys())
    label_map = {s: i for i, s in enumerate(subjects)}

    train_ds = SubjectImageDataset(train_data, label_map, transform=train_transform)
    val_ds = SubjectImageDataset(val_data, label_map, transform=eval_transform)

    return train_ds, val_ds, len(label_map)
