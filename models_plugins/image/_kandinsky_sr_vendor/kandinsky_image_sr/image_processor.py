"""Input color management and tensor/PIL conversion."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps

ICC_RENDERING_INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC
ICC_GRAY_MODES = ("L", "LA")


def _is_srgb(profile: ImageCms.ImageCmsProfile) -> bool:
    return ImageCms.getProfileDescription(profile).strip().lower().startswith("srgb")


def _apply_icc(image: Image.Image, profile_bytes: bytes) -> Image.Image:
    source = ImageCms.getOpenProfile(io.BytesIO(profile_bytes))
    if _is_srgb(source):
        return image.convert("RGB")
    if image.mode in ICC_GRAY_MODES:
        image = image.convert("L")
    elif image.mode not in ("RGB", "CMYK"):
        image = image.convert("RGB")
    return ImageCms.profileToProfile(
        image,
        source,
        ImageCms.createProfile("sRGB"),
        renderingIntent=ICC_RENDERING_INTENT,
        outputMode="RGB",
    )


def to_srgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert embedded ICC colors to sRGB."""
    image = ImageOps.exif_transpose(image)
    profile_bytes = image.info.get("icc_profile")
    if not profile_bytes:
        return image.convert("RGB")
    image.load()
    try:
        rgb = _apply_icc(image, profile_bytes)
    except ImageCms.PyCMSError:
        rgb = image.convert("RGB")
    rgb.info.pop("icc_profile", None)
    return rgb


def image_to_tensor(image: Any) -> torch.Tensor:
    """Convert a path, PIL image, NumPy array, or tensor to `(1, 3, H, W)` float RGB."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = to_srgb(opened)
    elif isinstance(image, Image.Image):
        image = to_srgb(image)

    if isinstance(image, Image.Image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise TypeError(f"unsupported image type: {type(image).__name__}")

    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"expected one RGB image shaped (3,H,W) or (1,3,H,W), got {tuple(tensor.shape)}")
    tensor = tensor.float().div(255.0) if tensor.dtype == torch.uint8 else tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError("image contains NaN or infinity")
    return tensor.clamp(0.0, 1.0).cpu()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a `(1, 3, H, W)` float tensor to an 8-bit RGB PIL image."""
    array = tensor.squeeze(0).mul(255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")
