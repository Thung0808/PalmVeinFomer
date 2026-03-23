"""
Phase 1: Swin-Tiny backbone + Embedding head + ArcFace
"""

from __future__ import annotations

import math
import logging
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

log = logging.getLogger("phase1_model")

BACKBONE_CHOICES = [
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "swin_base_patch4_window7_224",
    "resnet18_transformer",
    "resnet34_transformer",
    "resnet50_transformer",
]

_RESNET_TRANSFORMER_MAP = {
    "resnet18_transformer": "resnet18",
    "resnet34_transformer": "resnet34",
    "resnet50_transformer": "resnet50",
}


def _safe_timm_create_model(model_name: str, pretrained: bool, **kwargs) -> nn.Module:
    if not pretrained:
        return timm.create_model(model_name, pretrained=False, **kwargs)
    try:
        return timm.create_model(model_name, pretrained=True, **kwargs)
    except Exception as exc:
        log.warning(
            "Could not load pretrained weights for %s (%s). Falling back to random init.",
            model_name,
            exc,
        )
        return timm.create_model(model_name, pretrained=False, **kwargs)


class ResidualStepBlock(nn.Module):
    """
    Lightweight residual enhancement block inspired by RSNet style conv steps.
    Used before Swin self-attention to enhance vein-like local structures.
    """

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class RSStemPatchProj(nn.Module):
    """
    Replaces Swin patch embedding projection conv (stride=4) with:
      conv(stride=2) -> residual steps -> conv(stride=2)
    Output shape stays compatible with Swin PatchEmbed.
    """

    def __init__(self, in_chans: int, embed_dim: int, stem_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
            ResidualStepBlock(stem_channels),
            ResidualStepBlock(stem_channels),
            nn.Conv2d(stem_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention:
    captures channel attention conditioned on vertical/horizontal coordinates.
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        reduction = max(8, reduction)
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        n, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2).contiguous()  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2).contiguous()

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class SwinStageWithCoordinateAttention(nn.Module):
    """
    Wrapper for one Swin stage.
    Swin tensor format here is NHWC, so we permute around CA.
    """

    def __init__(self, stage: nn.Module, channels: int, reduction: int = 32):
        super().__init__()
        self.stage = stage
        self.ca = CoordinateAttention(channels=channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage(x)  # NHWC
        x_bchw = x.permute(0, 3, 1, 2).contiguous()
        x_bchw = self.ca(x_bchw)
        return x_bchw.permute(0, 2, 3, 1).contiguous()


class ResNetTransformerBackbone(nn.Module):
    """
    ResNet feature extractor + lightweight Transformer encoder on spatial tokens.
    Keeps the CNN local inductive bias while adding global token mixing.
    """

    def __init__(
        self,
        resnet_name: str = "resnet18",
        pretrained: bool = True,
        image_size: int = 224,
        transformer_dim: int = 256,
        transformer_depth: int = 3,
        transformer_heads: int = 4,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if transformer_dim % transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")

        base = _safe_timm_create_model(
            resnet_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        act1 = getattr(base, "act1", None)
        if act1 is None:
            act1 = getattr(base, "relu", None)
        if act1 is None:
            raise AttributeError(f"Unsupported ResNet backbone without act1/relu: {resnet_name}")

        self.stem = nn.Sequential(base.conv1, base.bn1, act1, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.image_size = int(image_size)
        self.grid_size = max(1, int(math.ceil(self.image_size / 32)))
        self.token_proj = nn.Conv2d(base.num_features, transformer_dim, kernel_size=1, bias=False)
        self.token_norm = nn.LayerNorm(transformer_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size * self.grid_size, transformer_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=int(transformer_dim * transformer_mlp_ratio),
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        self.out_norm = nn.LayerNorm(transformer_dim)
        self.num_features = transformer_dim

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _resize_pos_embed(self, height: int, width: int) -> torch.Tensor:
        if height == self.grid_size and width == self.grid_size:
            return self.pos_embed
        pos = self.pos_embed.transpose(1, 2).reshape(1, -1, self.grid_size, self.grid_size)
        pos = F.interpolate(pos, size=(height, width), mode="bilinear", align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def freeze_stages(self, num_stages: int = 2) -> None:
        stage_modules = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        upto = min(len(stage_modules), max(0, int(num_stages)) + 1)
        for module in stage_modules[:upto]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_cnn(x)
        x = self.token_proj(x)
        batch_size, _, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)
        tokens = tokens + self._resize_pos_embed(height, width)
        tokens = self.transformer(tokens)
        tokens = self.out_norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled


def _get_stage_output_channels(backbone: nn.Module) -> list[int]:
    channels = []
    for stage in backbone.layers:
        out_dim = int(stage.dim)
        downsample = getattr(stage, "downsample", None)
        if downsample is not None and downsample.__class__.__name__.lower() != "identity":
            out_dim *= 2
        channels.append(out_dim)
    return channels


def build_backbone(
    backbone_name: str = "swin_tiny_patch4_window7_224",
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
    use_rs_patch_embed: bool = True,
    use_coordinate_attention: bool = True,
    ca_reduction: int = 32,
    image_size: int = 224,
    transformer_dim: int = 256,
    transformer_depth: int = 3,
    transformer_heads: int = 4,
    transformer_mlp_ratio: float = 4.0,
    transformer_dropout: float = 0.1,
) -> nn.Module:
    if backbone_name in _RESNET_TRANSFORMER_MAP:
        return ResNetTransformerBackbone(
            resnet_name=_RESNET_TRANSFORMER_MAP[backbone_name],
            pretrained=pretrained,
            image_size=image_size,
            transformer_dim=transformer_dim,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_dropout=transformer_dropout,
        )

    backbone = _safe_timm_create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=0,
        drop_path_rate=drop_path_rate,
        global_pool="avg",
    )
    if use_rs_patch_embed:
        old_proj = backbone.patch_embed.proj
        backbone.patch_embed.proj = RSStemPatchProj(
            in_chans=old_proj.in_channels,
            embed_dim=old_proj.out_channels,
            stem_channels=max(32, old_proj.out_channels // 2),
        )

    if use_coordinate_attention:
        stage_channels = _get_stage_output_channels(backbone)
        wrapped = [
            SwinStageWithCoordinateAttention(stage, ch, reduction=ca_reduction)
            for stage, ch in zip(backbone.layers, stage_channels)
        ]
        backbone.layers = nn.Sequential(*wrapped)

    return backbone


def freeze_backbone_stages(backbone: nn.Module, num_stages: int = 2) -> None:
    """Freeze first `num_stages` stages (layers.0, layers.1)."""
    if hasattr(backbone, "freeze_stages"):
        backbone.freeze_stages(num_stages)
        return
    for name, param in backbone.named_parameters():
        for i in range(num_stages):
            if name == f"layers.{i}" or name.startswith(f"layers.{i}."):
                param.requires_grad = False
                break


def unfreeze_all(backbone: nn.Module) -> None:
    for param in backbone.parameters():
        param.requires_grad = True


class VeinModel(nn.Module):
    """
    Backbone (768-d) → Dropout → Linear → BN → normalize → 512-d embedding.
    Dropout before embedding improves generalization for metric learning.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int = 512,
        backbone_dim: int = 768,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Dropout(p=emb_dropout),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb


class CenterLoss(nn.Module):
    """
    Center Loss: minimize intra-class embedding distance.
    Each class has a learnable center; loss = 0.5 * mean(||emb - center||^2).
    Uses separate optimizer with higher LR (typically 0.5).
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_centers = self.centers[labels]
        diff = embeddings - batch_centers
        return 0.5 * (diff * diff).sum(dim=1).mean()


class ArcMarginProduct(nn.Module):
    """
    ArcFace: additive angular margin.
    s=30, m=0.35 (theo spec).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.35,
        subcenters: int = 1,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.subcenters = max(1, int(subcenters))
        self.weight = Parameter(torch.empty(self.out_features * self.subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self._update_margin_buffers(m)

    def _update_margin_buffers(self, m: float) -> None:
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_margin_scale(self, s: float, m: float) -> None:
        self.s = float(s)
        self.m = float(m)
        self._update_margin_buffers(self.m)

    def forward_no_margin(self, input: torch.Tensor) -> torch.Tensor:
        cosine = self._compute_cosine(input)
        return cosine * self.s

    def _compute_cosine(self, input: torch.Tensor) -> torch.Tensor:
        cosine = nn.functional.linear(
            nn.functional.normalize(input, p=2, dim=1),
            nn.functional.normalize(self.weight, p=2, dim=1),
        )
        if self.subcenters > 1:
            cosine = cosine.view(input.size(0), self.out_features, self.subcenters).max(dim=2).values
        return cosine

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = self._compute_cosine(input)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        output = cosine * (1 - one_hot) + phi * one_hot
        return output * self.s


def build_model(
    num_classes: int,
    backbone_name: str = "swin_tiny_patch4_window7_224",
    embedding_dim: int = 512,
    drop_path_rate: float = 0.2,
    arc_s: float = 30.0,
    arc_m: float = 0.35,
    arc_subcenters: int = 1,
    use_rs_patch_embed: bool = True,
    use_coordinate_attention: bool = True,
    ca_reduction: int = 32,
    pretrained: bool = True,
    emb_dropout: float = 0.1,
    image_size: int = 224,
    transformer_dim: int = 256,
    transformer_depth: int = 3,
    transformer_heads: int = 4,
    transformer_mlp_ratio: float = 4.0,
    transformer_dropout: float = 0.1,
) -> tuple[VeinModel, ArcMarginProduct]:
    backbone = build_backbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        use_rs_patch_embed=use_rs_patch_embed,
        use_coordinate_attention=use_coordinate_attention,
        ca_reduction=ca_reduction,
        image_size=image_size,
        transformer_dim=transformer_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        transformer_mlp_ratio=transformer_mlp_ratio,
        transformer_dropout=transformer_dropout,
    )
    backbone_dim = int(getattr(backbone, "num_features", 768))
    model = VeinModel(backbone, embedding_dim=embedding_dim, backbone_dim=backbone_dim, emb_dropout=emb_dropout)
    arcface = ArcMarginProduct(
        embedding_dim,
        num_classes,
        s=arc_s,
        m=arc_m,
        subcenters=arc_subcenters,
    )
    return model, arcface


def get_embedding_output_dim(model: VeinModel) -> int:
    for layer in model.embedding:
        if isinstance(layer, nn.Linear):
            return int(layer.out_features)
    raise RuntimeError("Could not infer embedding output dimension from model.embedding")
