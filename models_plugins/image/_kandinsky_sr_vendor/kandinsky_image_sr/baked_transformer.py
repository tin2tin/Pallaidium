"""Text-free Diffusers model for a pre-baked Kandinsky ImageSR transformer."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_kandinsky import (
    Kandinsky5Attention,
    Kandinsky5FeedForward,
    Kandinsky5Modulation,
    Kandinsky5OutLayer,
    Kandinsky5RoPE3D,
    Kandinsky5VisualEmbeddings,
    fractal_flatten,
    fractal_unflatten,
)


class KandinskyImageSRBakedBlock(nn.Module):
    """Kandinsky 5 visual block with zero-text cross-attention replaced by a table."""

    def __init__(self, model_dim: int, time_dim: int, ff_dim: int, head_dim: int, num_timesteps: int) -> None:
        super().__init__()
        self.visual_modulation = Kandinsky5Modulation(time_dim, model_dim, 9)
        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = Kandinsky5Attention(model_dim, head_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = Kandinsky5FeedForward(model_dim, ff_dim)
        self.register_buffer("xattn_bias_table", torch.zeros(num_timesteps, model_dim, dtype=torch.float32))

    def forward(
        self,
        visual_embed: torch.Tensor,
        time_embed: torch.Tensor,
        rope: torch.Tensor,
        sparse_params: dict[str, Any] | None,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_params, _, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )

        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = (
            self.self_attention_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(visual_embed)
        visual_out = self.self_attention(visual_out, rotary_emb=rope, sparse_params=sparse_params)
        visual_embed = (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)

        baked_bias = self.xattn_bias_table.index_select(0, timestep_indices).unsqueeze(1)
        visual_embed = (visual_embed.float() + baked_bias).type_as(visual_embed)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = (
            self.feed_forward_norm(visual_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(visual_embed)
        visual_out = self.feed_forward(visual_out)
        return (visual_embed.float() + gate.float() * visual_out.float()).type_as(visual_embed)


class KandinskyImageSRBakedTransformer3DModel(ModelMixin, ConfigMixin):
    """Kandinsky 5 ImageSR DiT with zero-text conditioning baked for fixed timesteps.

    The text encoder, time/pooled-text encoders, and all visual cross-attention modules are
    absent. Their deterministic zero-text outputs are persistent safetensors buffers.
    """

    _supports_gradient_checkpointing = False
    _keep_in_fp32_modules: ClassVar[list[str]] = [
        "modulation",
        "visual_modulation",
        "time_embed_table",
        "xattn_bias_table",
    ]

    @register_to_config
    def __init__(
        self,
        in_visual_dim: int = 16,
        out_visual_dim: int = 16,
        time_dim: int = 512,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        model_dim: int = 2560,
        ff_dim: int = 10240,
        num_visual_blocks: int = 50,
        axes_dims: tuple[int, int, int] = (16, 24, 24),
        visual_cond: bool = False,
        attention_type: str = "regular",
        baked_timesteps: tuple[float, ...] | list[float] = (0.15, 0.244),
        baked_latent_size: int = 128,
    ) -> None:
        super().__init__()
        if not baked_timesteps:
            raise ValueError("baked_timesteps must not be empty")
        if attention_type != "regular":
            raise ValueError("the baked ImageSR transformer currently supports regular attention only")

        head_dim = sum(axes_dims)
        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond else in_visual_dim
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond
        self.attention_type = attention_type
        self.baked_timesteps = tuple(float(value) for value in baked_timesteps)
        self.baked_latent_size = int(baked_latent_size)

        self.visual_embeddings = Kandinsky5VisualEmbeddings(visual_embed_dim, model_dim, patch_size)
        self.visual_rope_embeddings = Kandinsky5RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [
                KandinskyImageSRBakedBlock(model_dim, time_dim, ff_dim, head_dim, len(self.baked_timesteps))
                for _ in range(num_visual_blocks)
            ]
        )
        self.out_layer = Kandinsky5OutLayer(model_dim, time_dim, out_visual_dim, patch_size)
        self.register_buffer(
            "time_embed_table", torch.zeros(len(self.baked_timesteps), time_dim, dtype=torch.float32)
        )

    def _timestep_indices(self, timestep: torch.Tensor) -> torch.Tensor:
        requested = timestep.float() / 1000.0
        available = torch.tensor(self.baked_timesteps, device=timestep.device, dtype=torch.float32)
        distance, indices = (requested[:, None] - available[None]).abs().min(dim=1)
        if bool((distance > 1e-6).any()):
            bad = requested[distance > 1e-6].detach().cpu().tolist()
            raise ValueError(f"timesteps {bad} are not baked; available values: {list(self.baked_timesteps)}")
        return indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep: torch.Tensor,
        pooled_projections: torch.Tensor | None,
        visual_rope_pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_rope_pos: torch.Tensor | None,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | torch.Tensor:
        del encoder_hidden_states, pooled_projections, text_rope_pos
        if hidden_states.shape[2:4] != (self.baked_latent_size, self.baked_latent_size):
            raise ValueError(
                f"baked transformer expects {self.baked_latent_size}x{self.baked_latent_size} latents, "
                f"got {tuple(hidden_states.shape[2:4])}"
            )
        timestep_indices = self._timestep_indices(timestep)
        time_embed = self.time_embed_table.index_select(0, timestep_indices)
        visual_embed = self.visual_embeddings(hidden_states)
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(
            visual_embed, visual_rope, visual_shape, block_mask=to_fractal
        )

        for block in self.visual_transformer_blocks:
            visual_embed = block(visual_embed, time_embed, visual_rope, sparse_params, timestep_indices)

        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        output = self.out_layer(visual_embed, None, time_embed)
        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)
