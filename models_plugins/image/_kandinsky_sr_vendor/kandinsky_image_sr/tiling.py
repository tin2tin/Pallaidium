"""Boundary-aware Hanning tiling used by the ImageSR pipeline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def pad_to_min_tile(image: torch.Tensor, tile: int) -> tuple[torch.Tensor, int, int]:
    _, _, height, width = image.shape
    pad_h = max(0, tile - height)
    pad_w = max(0, tile - width)
    if pad_h == 0 and pad_w == 0:
        return image, height, width
    return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate"), height, width


def boundary_aware_window(
    height: int,
    width: int,
    *,
    top: bool,
    bottom: bool,
    left: bool,
    right: bool,
    device: torch.device,
) -> torch.Tensor:
    wy = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(height, device=device) / height)
    wx = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(width, device=device) / width)
    if top:
        wy[: height // 2] = 1.0
    if bottom:
        wy[height // 2 :] = 1.0
    if left:
        wx[: width // 2] = 1.0
    if right:
        wx[width // 2 :] = 1.0
    return wy[:, None] * wx[None, :]
