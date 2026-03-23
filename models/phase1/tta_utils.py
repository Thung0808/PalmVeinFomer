from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

_ROT_RE = re.compile(r"^rot(?P<deg>[+-]?\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class TTAVariant:
    name: str
    kind: str
    param: float | int | None = None


def parse_tta_variants_arg(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    items = [t.strip() for t in spec.replace(";", ",").split(",")]
    items = [t for t in items if t]
    return items or None


def build_tta_plan(variants: Sequence[str] | None) -> list[TTAVariant]:
    if not variants:
        return []
    plan: list[TTAVariant] = []
    for raw in variants:
        v = str(raw).strip().lower()
        if not v or v in {"orig", "original", "identity", "none"}:
            continue
        if v in {"hflip", "hf", "flip"}:
            plan.append(TTAVariant(name="hflip", kind="hflip"))
            continue
        if v in {"vflip", "vf"}:
            plan.append(TTAVariant(name="vflip", kind="vflip"))
            continue
        if v in {"rot90", "r90"}:
            plan.append(TTAVariant(name="rot90", kind="rot90", param=1))
            continue
        if v in {"rot180", "r180"}:
            plan.append(TTAVariant(name="rot180", kind="rot90", param=2))
            continue
        if v in {"rot270", "r270"}:
            plan.append(TTAVariant(name="rot270", kind="rot90", param=3))
            continue
        if v.startswith("rot"):
            m = _ROT_RE.match(v)
            if not m:
                raise ValueError(f"Unsupported TTA variant: {raw}")
            deg = float(m.group("deg"))
            if abs(deg) >= 360.0:
                deg = deg % 360.0
            if abs(deg) < 1e-6:
                continue
            if abs(deg) % 90.0 < 1e-6:
                k = int(round(deg / 90.0)) % 4
                if k != 0:
                    plan.append(TTAVariant(name=f"rot{int(deg)}", kind="rot90", param=k))
            else:
                plan.append(TTAVariant(name=f"rot{deg:g}", kind="rot", param=deg))
            continue
        raise ValueError(f"Unsupported TTA variant: {raw}")

    # Deduplicate by name while preserving order
    seen: set[str] = set()
    unique: list[TTAVariant] = []
    for item in plan:
        if item.name in seen:
            continue
        seen.add(item.name)
        unique.append(item)
    return unique


def apply_tta_variant(images: torch.Tensor, variant: TTAVariant) -> torch.Tensor:
    if variant.kind == "hflip":
        return torch.flip(images, dims=[3])
    if variant.kind == "vflip":
        return torch.flip(images, dims=[2])
    if variant.kind == "rot90":
        k = int(variant.param or 0) % 4
        return torch.rot90(images, k=k, dims=[2, 3])
    if variant.kind == "rot":
        deg = float(variant.param or 0.0)
        return TF.rotate(images, deg, interpolation=InterpolationMode.BILINEAR, fill=0.0)
    raise ValueError(f"Unsupported TTA variant kind: {variant.kind}")


def iter_tta_images(images: torch.Tensor, plan: Sequence[TTAVariant]) -> Iterable[torch.Tensor]:
    yield images
    for variant in plan:
        yield apply_tta_variant(images, variant)


def forward_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    plan: Sequence[TTAVariant],
    amp_enabled: bool,
) -> torch.Tensor:
    device_type = images.device.type
    if not plan:
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
            return model(images)
    emb_sum = None
    for aug in iter_tta_images(images, plan):
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp_enabled):
            emb = model(aug).float()
        emb_sum = emb if emb_sum is None else (emb_sum + emb)
    return F.normalize(emb_sum, p=2, dim=1)
