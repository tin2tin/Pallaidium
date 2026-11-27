"""Attention backend selection with a FlashAttention-2 fallback to PyTorch SDPA."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import torch


class AttentionBackend(str, Enum):
    """Attention backends the pipeline can run on.

    `AUTO` uses FlashAttention-2 when the `flash-attn` package is usable on a CUDA device
    and PyTorch SDPA otherwise.
    """

    AUTO = "auto"
    FLASH = "flash"
    NATIVE = "native"


class AttentionConfigurable(Protocol):
    """Any Diffusers model exposing `set_attention_backend`."""

    def set_attention_backend(self, backend: str) -> None: ...


def configure_attention(
    transformer: AttentionConfigurable, backend: AttentionBackend, device: torch.device
) -> AttentionBackend:
    """Switch the transformer to an attention backend.

    Args:
        transformer: Model whose attention backend is switched, e.g. `pipe.transformer`.
        backend: Requested backend. `FLASH` fails loudly when it cannot be used; `AUTO`
            falls back to `NATIVE` instead.
        device: Device the transformer runs on; FlashAttention-2 needs CUDA.

    Returns:
        The backend actually in use.

    Raises:
        ValueError: If `FLASH` is requested on a non-CUDA device.
        RuntimeError: If `FLASH` is requested and Diffusers cannot use `flash-attn`.
    """
    if backend is AttentionBackend.FLASH and device.type != "cuda":
        raise ValueError(f"FlashAttention-2 requires a CUDA device, got {device}")
    if backend is AttentionBackend.AUTO:
        return try_flash_or_native(transformer, device)
    transformer.set_attention_backend(backend.value)
    return backend


def try_flash_or_native(transformer: AttentionConfigurable, device: torch.device) -> AttentionBackend:
    """Use FlashAttention-2 when it is usable on `device`, PyTorch SDPA otherwise."""
    if device.type == "cuda":
        try:
            transformer.set_attention_backend(AttentionBackend.FLASH.value)
        except RuntimeError:
            pass
        else:
            return AttentionBackend.FLASH
    transformer.set_attention_backend(AttentionBackend.NATIVE.value)
    return AttentionBackend.NATIVE
