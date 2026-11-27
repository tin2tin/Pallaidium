"""Diffusers-native tiled pipeline for Kandinsky ImageSR 2.0."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from PIL import Image

from kandinsky_image_sr.image_processor import image_to_tensor, tensor_to_pil
from kandinsky_image_sr.tiling import boundary_aware_window, pad_to_min_tile, tile_positions

DEFAULT_TIMESTEPS = {2: 0.15, 4: 0.244}
CREATIVITY_TIMESTEPS = {
    2: (0.07, 0.086, 0.118, 0.134, 0.15, 0.18, 0.21, 0.27, 0.30),
    4: (0.17, 0.1885, 0.207, 0.2255, 0.244, 0.278, 0.312, 0.346, 0.38),
}


@dataclass
class KandinskyImageSROutput(BaseOutput):
    """Pipeline output containing one super-resolved image."""

    images: list[Image.Image] | np.ndarray | torch.Tensor


def _chunks(items: list[tuple[int, int]], size: int) -> Iterator[list[tuple[int, int]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class KandinskyImageSRPipeline(DiffusionPipeline):
    """Prompt-free one-step flow pipeline for the Kandinsky ImageSR 2.0 transformer.

    The transformer is the official Diffusers `Kandinsky5Transformer3DModel`; text inputs are
    four zero tokens, matching the SR training and evaluation path. The FLUX `AutoencoderKL`
    encodes bicubic-upsampled 1024-pixel tiles and decodes the corrected latent.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components: ClassVar[list[str]] = []
    _exclude_from_cpu_offload: ClassVar[list[str]] = []

    def __init__(
        self,
        transformer: torch.nn.Module,
        vae: torch.nn.Module,
        upscale_factors: tuple[int, ...] | list[int] = (2, 4),
        one_step_t_by_factor: dict[str, float] | None = None,
        tile_size: int = 1024,
        overlap: float = 0.25,
        text_sequence_length: int = 4,
        scale_factor: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.register_modules(transformer=transformer, vae=vae)
        factors = tuple(int(value) for value in upscale_factors)
        timesteps = one_step_t_by_factor or DEFAULT_TIMESTEPS
        timestep_config = {str(key): float(value) for key, value in timesteps.items()}
        self.register_to_config(
            upscale_factors=list(factors),
            one_step_t_by_factor=timestep_config,
            tile_size=int(tile_size),
            overlap=float(overlap),
            text_sequence_length=int(text_sequence_length),
            scale_factor=list(scale_factor),
        )

    @property
    def upscale_factors(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.config.upscale_factors)

    def default_timestep(self, upscale_factor: int) -> float:
        try:
            return float(self.config.one_step_t_by_factor[str(upscale_factor)])
        except KeyError as error:
            raise ValueError(f"no default timestep configured for x{upscale_factor}") from error

    def _resolve_timestep(self, upscale_factor: int, one_step_t: float | None, creativity: int | None) -> float:
        if one_step_t is not None and creativity is not None:
            raise ValueError("one_step_t and creativity are mutually exclusive")
        if creativity is not None:
            if type(creativity) is not int or not 0 <= creativity <= 8:
                raise ValueError(f"creativity must be an integer from 0 to 8, got {creativity!r}")
            values = CREATIVITY_TIMESTEPS.get(upscale_factor)
            if values is None:
                raise ValueError("creativity presets are available only for x2 and x4")
            timestep = values[creativity]
        else:
            timestep = self.default_timestep(upscale_factor) if one_step_t is None else float(one_step_t)
        if not 0.0 < timestep < 1.0:
            raise ValueError(f"one_step_t must lie in (0, 1), got {timestep}")
        baked_timesteps = getattr(self.transformer.config, "baked_timesteps", None)
        if baked_timesteps is not None and not any(abs(timestep - float(value)) <= 1e-6 for value in baked_timesteps):
            raise ValueError(
                f"one_step_t={timestep} is not present in this baked model; "
                f"available values: {[float(value) for value in baked_timesteps]}"
            )
        return timestep

    def _model_dtype(self) -> torch.dtype:
        # Kandinsky 5 intentionally keeps time/modulation parameters in FP32 while the
        # visual and text paths use the requested inference dtype. The first parameter is
        # therefore not representative for model inputs.
        visual_embeddings = getattr(self.transformer, "visual_embeddings", self.transformer)
        return next(visual_embeddings.parameters()).dtype

    def _vae_dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def _autocast(self, device: torch.device, dtype: torch.dtype) -> Any:
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def _encode(self, pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
        vae_input = (pixels * 2.0 - 1.0).to(device=device, dtype=self._vae_dtype())
        encoded = self.vae.encode(vae_input)
        latent_dist = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
        latents = latent_dist.mode()
        return latents.float() * float(self.vae.config.scaling_factor)

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        scaled = (latents / float(self.vae.config.scaling_factor)).to(dtype=self._vae_dtype())
        with self._autocast(latents.device, self._vae_dtype()):
            decoded = self.vae.decode(scaled)
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        elif isinstance(decoded, tuple):
            decoded = decoded[0]
        return (decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5

    def _infer_tiles(self, pixels: torch.Tensor, timestep: float, device: torch.device) -> torch.Tensor:
        lq = self._encode(pixels, device)
        batch, _, height, width = lq.shape
        latent = lq.permute(0, 2, 3, 1).unsqueeze(1)
        use_visual_condition = bool(getattr(self.transformer.config, "visual_cond", False))
        if use_visual_condition:
            mask = torch.ones(batch, 1, height, width, 1, device=device, dtype=latent.dtype)
            hidden_states = torch.cat([latent, latent, mask], dim=-1)
        else:
            hidden_states = latent

        model_dtype = self._model_dtype()
        baked = getattr(self.transformer.config, "baked_timesteps", None) is not None
        text_length = int(self.config.text_sequence_length)
        if baked:
            text = None
            pooled = None
            text_rope_pos = None
        else:
            text_dim = int(getattr(self.transformer.config, "in_text_dim", 3584))
            pooled_dim = int(getattr(self.transformer.config, "in_text_dim2", 768))
            text = torch.zeros(batch, text_length, text_dim, device=device, dtype=model_dtype)
            pooled = torch.zeros(batch, pooled_dim, device=device, dtype=model_dtype)
            text_rope_pos = torch.arange(text_length, device=device)
        patch_t, patch_h, patch_w = tuple(int(v) for v in self.transformer.config.patch_size)
        visual_rope_pos = (
            torch.arange(1 // patch_t, device=device),
            torch.arange(height // patch_h, device=device),
            torch.arange(width // patch_w, device=device),
        )
        model_t = torch.full((batch,), timestep * 1000.0, device=device, dtype=torch.float32)
        with self._autocast(device, model_dtype):
            prediction = self.transformer(
                hidden_states=hidden_states.to(dtype=model_dtype),
                encoder_hidden_states=text,
                timestep=model_t,
                pooled_projections=pooled,
                visual_rope_pos=visual_rope_pos,
                text_rope_pos=text_rope_pos,
                scale_factor=tuple(float(v) for v in self.config.scale_factor),
                return_dict=False,
            )
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        elif hasattr(prediction, "sample"):
            prediction = prediction.sample
        prediction = prediction.squeeze(1).permute(0, 3, 1, 2).float()
        corrected = lq - timestep * prediction
        return self._decode(corrected)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image | np.ndarray | torch.Tensor | str,
        *,
        upscale_factor: int = 4,
        one_step_t: float | None = None,
        creativity: int | None = None,
        tile_size: int | None = None,
        overlap: float | None = None,
        tile_batch_size: int = 1,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> KandinskyImageSROutput | tuple[Any]:
        """Super-resolve one image.

        Args:
            image: Path, PIL image, RGB NumPy array, or one RGB torch tensor.
            upscale_factor: Output factor, either x2 or x4.
            one_step_t: Raw flow timestep/strength in `(0, 1)`; defaults to the calibrated per-factor value.
            creativity: Calibrated level 0..8 for x2/x4. Mutually exclusive with `one_step_t`.
            tile_size: Output tile side; defaults to 1024.
            overlap: LQ tile overlap fraction; defaults to 0.25.
            tile_batch_size: Tiles evaluated together.
            output_type: `"pil"`, `"np"`, or `"pt"`.
            return_dict: Return a `KandinskyImageSROutput`; otherwise a one-item tuple.
        """
        upscale_factor = int(upscale_factor)
        if upscale_factor not in self.upscale_factors:
            raise ValueError(f"upscale_factor must be one of {self.upscale_factors}, got {upscale_factor}")
        if tile_batch_size < 1:
            raise ValueError("tile_batch_size must be at least 1")
        tile_size = int(tile_size or self.config.tile_size)
        overlap = float(self.config.overlap if overlap is None else overlap)
        if tile_size <= 0 or tile_size % upscale_factor:
            raise ValueError("tile_size must be positive and divisible by upscale_factor")
        baked_latent_size = getattr(self.transformer.config, "baked_latent_size", None)
        if baked_latent_size is not None and tile_size != int(baked_latent_size) * 8:
            raise ValueError(f"this baked model requires tile_size={int(baked_latent_size) * 8}, got {tile_size}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must lie in [0, 1)")
        timestep = self._resolve_timestep(upscale_factor, one_step_t, creativity)

        input_tensor = image_to_tensor(image)
        lq_tile = tile_size // upscale_factor
        stride = max(1, int(lq_tile * (1.0 - overlap)))
        padded, original_h, original_w = pad_to_min_tile(input_tensor, lq_tile)
        tops = tile_positions(padded.shape[-2], lq_tile, stride)
        lefts = tile_positions(padded.shape[-1], lq_tile, stride)
        cells = [(row, col) for row in range(len(tops)) for col in range(len(lefts))]
        device = self._execution_device
        output_h, output_w = padded.shape[-2] * upscale_factor, padded.shape[-1] * upscale_factor
        accumulator = torch.zeros(1, 3, output_h, output_w, device=device)
        weights = torch.zeros(1, 1, output_h, output_w, device=device)

        for chunk in _chunks(cells, tile_batch_size):
            tiles = []
            for row, col in chunk:
                top, left = tops[row], lefts[col]
                tile = padded[:, :, top : top + lq_tile, left : left + lq_tile]
                tiles.append(F.interpolate(tile, size=(tile_size, tile_size), mode="bicubic", align_corners=False))
            predictions = self._infer_tiles(torch.cat(tiles).clamp(0.0, 1.0), timestep, device)
            for index, (row, col) in enumerate(chunk):
                window = boundary_aware_window(
                    tile_size,
                    tile_size,
                    top=row == 0,
                    bottom=row == len(tops) - 1,
                    left=col == 0,
                    right=col == len(lefts) - 1,
                    device=device,
                )[None, None]
                top_hr, left_hr = tops[row] * upscale_factor, lefts[col] * upscale_factor
                accumulator[:, :, top_hr : top_hr + tile_size, left_hr : left_hr + tile_size] += (
                    predictions[index : index + 1] * window
                )
                weights[:, :, top_hr : top_hr + tile_size, left_hr : left_hr + tile_size] += window

        tensor = (accumulator / weights.clamp_min(1e-6))[
            :, :, : original_h * upscale_factor, : original_w * upscale_factor
        ].cpu()
        if output_type == "pil":
            output: Any = [tensor_to_pil(tensor)]
        elif output_type == "np":
            output = tensor.permute(0, 2, 3, 1).numpy()
        elif output_type == "pt":
            output = tensor
        else:
            raise ValueError("output_type must be 'pil', 'np', or 'pt'")
        if not return_dict:
            return (output,)
        return KandinskyImageSROutput(images=output)
