# Vendored from https://huggingface.co/WaveCut/MiniMax-H3-Image-Pruned/blob/3a804c0fc3e73a04606fd9618d947b08b5cfc970/transformer/modeling_minimax_h3_image_pruned.py
# Bundled so MiniMax H3 Image loads without trust_remote_code (no per-load code fetch, no silent
# upstream changes). Keep in sync manually if the repo changes. Unmodified below this header.
# Copyright 2026 WaveCut. Derived from diffusers' `transformer_minimax_h3_image.py`
# (Copyright 2026 The MiniMax and HuggingFace Teams), Apache License 2.0.
"""MiniMax-H3-Image transformer with tabulated AdaLN modulation.

The released checkpoint samples on one fixed grid of six timesteps (Parallel Decoding Distillation keeps one output
head per step), and the reference rows of an edit stay at t = 0.999. The input of every AdaLN projection is a function
of the timestep alone, so the modulation of each block takes a fixed set of values. This model stores those values as
tables and drops what computed them:

* `transformer_blocks.*.adaln_proj.linear` (13.0B parameters, 26.0 GB in bfloat16), including the audio third of
  every projection, which the image model never addresses;
* `time_embedder` and `norm_out.linear`.

Each table row holds exactly what the original projection returns for that timestep set, computed with the same
modules and the same matrix shapes (one row for text-to-image steps, two rows for edit steps). Everything else in the
forward pass is the diffusers implementation unchanged.

The last block also skips the rows whose output nothing reads: the text rows and the reference rows of an edit. They
still provide keys and values to the generated rows; their queries, attention output and feed-forward are not
computed. While `skip_dead_rows` is on (the default), the velocity returned for reference rows is zero (the pipeline
steps the generated rows only); `skip_dead_rows = False` restores the full last block.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from diffusers import MiniMaxH3ImageTransformer2DModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3_image import _apply_rotary_emb

# Modality tags of the packed sequence: 0 image, 1 text. MiniMax-H3 also has 2 = audio, which the image model never
# assigns, so the tables keep two rows per timestep instead of three.
NUM_TAGS = 2


class TabulatedModulation(nn.Module):
    """Replaces `MiniMaxH3ImageAdaLayerNormModulation`: returns the six modulation tensors of a timestep set.

    `table` holds, for every supported timestep set, the projection output with the audio rows removed: `[t0 image,
    t0 text, t1 image, t1 text, ...]`, `6 * hidden_size` wide. The block addresses these rows with
    `timestep_indices * NUM_TAGS + token_tags`.
    """

    def __init__(self, rows: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("table", torch.zeros(rows, 6 * hidden_size), persistent=True)

    def forward(self, rows: slice) -> tuple[torch.Tensor, ...]:
        return self.table[rows].chunk(6, dim=-1)


class TabulatedNormOut(nn.Module):
    """Replaces `MiniMaxH3ImageAdaLayerNormOut`: the final RMSNorm with tabulated `shift, scale` rows."""

    def __init__(self, hidden_size: int, rows: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.register_buffer("table", torch.zeros(rows, 2 * hidden_size), persistent=True)

    def forward(self, hidden_states: torch.Tensor, rows: slice, timestep_indices: torch.Tensor) -> torch.Tensor:
        shift, scale = self.table[rows].chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


def _live_last_block(block, hidden_states, rows, adaln_indices, rotary_emb, live_rows):
    """`MiniMaxH3ImageTransformerBlock.forward` for the rows in `live_rows` only.

    Every row still contributes its key and value. The result is `(batch, len(live_rows), hidden)`.
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(rows)

    norm_hidden_states = block.norm1(hidden_states)
    norm_hidden_states = norm_hidden_states * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(
        0, adaln_indices
    )

    attn = block.attn
    if attn.fused_projections:
        query, key, value = attn.to_qkv(norm_hidden_states).chunk(3, dim=-1)
        query = query.index_select(1, live_rows)
    else:
        query = attn.to_q(norm_hidden_states.index_select(1, live_rows))
        key = attn.to_k(norm_hidden_states)
        value = attn.to_v(norm_hidden_states)
    query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim)))
    key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim)))
    value = value.unflatten(-1, (-1, attn.head_dim))
    cos, sin = rotary_emb
    query = _apply_rotary_emb(query, cos.index_select(0, live_rows), sin.index_select(0, live_rows))
    key = _apply_rotary_emb(key, cos, sin)
    processor = attn.processor
    attn_output = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        backend=getattr(processor, "_attention_backend", None),
        parallel_config=getattr(processor, "_parallel_config", None),
    )
    attn_output = attn_output.flatten(2, 3).type_as(query)
    attn_output = attn.to_out[1](attn.to_out[0](attn_output))

    live_indices = adaln_indices.index_select(0, live_rows)
    hidden_states = hidden_states.index_select(1, live_rows) + gate_msa.index_select(0, live_indices) * attn_output
    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp.index_select(0, live_indices)) + shift_mlp.index_select(
        0, live_indices
    )
    return hidden_states + gate_mlp.index_select(0, live_indices) * block.ff(norm_hidden_states)


class MiniMaxH3ImagePrunedTransformer2DModel(MiniMaxH3ImageTransformer2DModel):
    r"""
    [`MiniMaxH3ImageTransformer2DModel`] with the AdaLN modulation read from tables.

    Additional config:
        adaln_plans (`list[list[float]]`):
            The supported timestep sets, each the `timestep` tensor of one transformer call: one value for a
            text-to-image step, the step value and the reference value for an edit step. Table rows follow this
            order.
        reference_timestep (`float`, defaults to `0.999`):
            The timestep the reference rows of an edit are held at.
        auto_map (`dict`, *optional*):
            Remote-code entry for `diffusers.AutoModel`; not used by the model.

    Runtime switches (attributes):
        skip_dead_rows (`bool`, defaults to `True`): run the last block and the output head for the generated rows
            only.
        validate_inputs (`bool`, defaults to `True`): check every call's `token_tags` (image or text only).
    """

    # `time_embedder` is gone; the remaining float32 modules are those of the original model.
    _keep_in_fp32_modules = ["proj_in", "pdd_proj_out", "rope"]
    _no_split_modules = ["MiniMaxH3ImageTransformerBlock", "MiniMaxH3ImageTokenRefinerBlock", "TabulatedNormOut"]
    _tp_plan = None
    _cp_plan = None

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        num_pdd_heads: int = 6,
        adaln_plans: list[list[float]] | None = None,
        reference_timestep: float = 0.999,
        auto_map: dict | None = None,
    ) -> None:
        # `auto_map` is read by `diffusers.AutoModel(..., trust_remote_code=True)`; it is accepted here only so that
        # loading the config does not warn about it.
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            num_pdd_heads=num_pdd_heads,
        )
        if not adaln_plans:
            raise ValueError("`adaln_plans` is empty: this checkpoint has no modulation tables.")

        del self.time_proj
        del self.time_embedder

        # Timestep set -> (block table rows, output table rows, index of the reference timestep in the set or -1).
        self._plans: dict[tuple[float, ...], tuple[slice, slice, int]] = {}
        block_row = out_row = 0
        reference = float(np.float32(reference_timestep))
        for plan in adaln_plans:
            values = tuple(float(np.float32(t)) for t in plan)
            if list(values) != sorted(set(values)):
                raise ValueError(f"A timestep set must be strictly increasing, got {plan}.")
            reference_slot = values.index(reference) if reference in values else -1
            self._plans[values] = (
                slice(block_row, block_row + NUM_TAGS * len(values)),
                slice(out_row, out_row + len(values)),
                reference_slot,
            )
            block_row += NUM_TAGS * len(values)
            out_row += len(values)

        for block in self.transformer_blocks:
            block.adaln_proj = TabulatedModulation(block_row, hidden_size)
        self.norm_out = TabulatedNormOut(hidden_size, out_row, final_norm_eps)

        self.skip_dead_rows = True
        self.validate_inputs = True

    def _plan_of(self, timestep: torch.Tensor) -> tuple[slice, slice, int]:
        values = tuple(timestep.detach().to(device="cpu", dtype=torch.float32).tolist())
        plan = self._plans.get(values)
        if plan is None:
            supported = ", ".join(str(list(v)) for v in self._plans)
            raise ValueError(
                f"Timesteps {list(values)} are not in this checkpoint's modulation tables. MiniMax-H3-Image samples on "
                f"a fixed six-step grid; the tables hold exactly its timestep sets: {supported}."
            )
        return plan

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        image_indices: torch.Tensor,
        text_indices: torch.Tensor,
        pdd_step_index: int,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )
        if self.validate_inputs and int(token_tags.max()) >= NUM_TAGS:
            raise ValueError("MiniMax-H3-Image rows are image (0) or text (1); the audio tag (2) has no table rows.")
        block_rows, out_rows, reference_slot = self._plan_of(timestep)

        rotary_emb = self.rope(position_ids)

        # The packed sequence, as in `MiniMaxH3ImageTransformer2DModel.forward`.
        image_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
        text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
        text_embeds = self.token_refiner(text_embeds)

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, image_indices, image_embeds.to(text_embeds.dtype))

        adaln_indices = timestep_indices * NUM_TAGS + token_tags

        blocks = self.transformer_blocks
        full_blocks = blocks[:-1] if self.skip_dead_rows else blocks
        for block in full_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, block_rows, adaln_indices, rotary_emb
                )
            else:
                hidden_states = block(hidden_states, block_rows, adaln_indices, rotary_emb)

        proj_out = self.pdd_proj_out[pdd_step_index]
        if not self.skip_dead_rows:
            hidden_states = self.norm_out(hidden_states, out_rows, timestep_indices).to(get_parameter_dtype(proj_out))
            output = proj_out(hidden_states).index_select(1, image_indices)
        else:
            # Generated rows: the image rows that are not held at the reference timestep. Reference rows come first.
            if reference_slot < 0:
                live_mask, live_rows = None, image_indices
            else:
                live_mask = timestep_indices.index_select(0, image_indices) != reference_slot
                live_rows = image_indices[live_mask]
            hidden_states = _live_last_block(blocks[-1], hidden_states, block_rows, adaln_indices, rotary_emb, live_rows)
            hidden_states = self.norm_out(hidden_states, out_rows, timestep_indices.index_select(0, live_rows))
            live_output = proj_out(hidden_states.to(get_parameter_dtype(proj_out)))
            if live_mask is None:
                output = live_output
            else:
                output = live_output.new_zeros((live_output.shape[0], image_indices.shape[0], live_output.shape[-1]))
                output[:, live_mask] = live_output

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
