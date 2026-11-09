"""Open-H3-Regenerate-2K-IC — regenerate a low-resolution MiniMax-H3 draft at 1080p / 2K.

NVIDIA's in-context (IC) refiner (https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K-IC,
code https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K/IC, Apache-2.0): the
draft is encoded once and kept as clean *reference* rows in front of the target in MiniMax-H3's Ref2VA
transformer (`transformer_ref`); a rank-384 IC LoRA on that transformer generates the high-resolution
version of the same clip, so faces, lip sync, lettering and layout follow the draft. Sampling is the last
3 (up3, default), 4 (up4) or all 8 (hf8) points of the HyperFlow 8-step grid at CFG 1; `up3` / `up4`
start from the draft latent enlarged by the author 3D-conv latent upscaler, noised to sigma 0.835 / 0.923.

The reference implementation merges HyperFlow and LMS Sharp into bf16 weights and shards the model over
up to 8 GPUs. Here the base is OzzyGT's SDNQ-pruned `transformer_ref` (4-bit by default, ~11.4 GB), so
nothing can be merged; instead the three adapters run unmerged as ONE concatenated-rank LoRA per linear
(HyperFlow r256 + Sharp r64 x strength + IC r384). Two pieces need care because the pruned checkpoint
dropped the AdaLN projections and the time-embedder MLP:

  * HyperFlow's two-time embedding emb(t, r) = e_t + gate * (e_r' - e_t) is evaluated from the base
    time-embedder weights (fetched by byte range from the IC repo's `base_model/` copy, 65 MB) plus the
    adapter's `time_embedder` / `endpoint_time_embedder` LoRAs, then projected onto the pruned model's
    rank-8 curve basis (least squares). Measured against the unpruned AdaLN weights this leaves ~1.3-2.1 %
    modulation error (3.3-6.5 % if HyperFlow's embedding were ignored).
  * LMS Sharp's 50 AdaLN branches take 8-D curve coordinates from the Comfy `adaln_t_table` (read by byte
    range from Comfy-Org/MiniMax-H3) and are kept as a side branch, as in NVIDIA's reference.

The draft is read at 24 fps onto the 1344x768 canvas (124 frames, at least 113 real ones, about 4.7 s; a
16:9 draft is expected, other aspect ratios are centre-cropped). The draft's audio is muxed back in.
The fast non-causal VAE decoder of the reference is not used: the standard H3 VAE decodes (tiled).
4K is not offered — the packed sequence would be ~290k rows on a single GPU.
"""

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
import urllib.request

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename
from .minimax_h3 import _cached_snapshot_dir, _onload_vae_per_call

_IC_REPO = "Efficient-Large-Model/Open-H3-Regenerate-2K-IC"
_IC_FILES = {
    "ic_lora": "lora/open_h3_regenerate_2k_ic_lora.safetensors",
    "hyperflow": "lora/minimax_h3_hyperflow_8step_v1.0.safetensors",
    "sharp": "lora/minimax_h3_lms_v1.0_r64.safetensors",
    "upscaler": "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    "silence": "assets/silence_audio_124f.safetensors",
}
# The base time embedder (4 small tensors) lives in the first shard of the repo's unmodified H3 base copy.
_TE_SHARD = "base_model/transformer_ref/diffusion_pytorch_model-00001-of-00014.safetensors"
_TE_KEYS = ("time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
            "time_embedder.linear_2.weight", "time_embedder.linear_2.bias")
# Comfy adaln_t_table the Sharp curve branches were fitted to ([1025, 8] fp32, pinned by SHA256).
_CURVE = dict(repo="Comfy-Org/MiniMax-H3", revision="e5eb578a89295337b8ff433a035929ce0279e0b6",
              file="diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", tensor="adaln_t_table",
              sha256="c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7")

# output resolution -> (draft canvas h, w, target h, w, rows kept). 1080p is computed at 1088 rows (whole
# latent patches) and the centre 1080 are kept.
_RESOLUTIONS = {
    "1080p": (768, 1344, 1088, 1920, 1080),
    "2k": (768, 1344, 1440, 2560, 1440),
}
_LQ_768P = {384: (384, 672), 512: (512, 896), 576: (576, 1024)}   # 768p output from a smaller draft
_FRAMES, _MIN_REAL = 124, 113
_SHARP_STRENGTH = 1.0
_HF_GATE = 0.25
_REFERENCE_T = 0.999
_FF_CHUNK_ROWS = 16384
_PATCH = (1, 2, 2)
_TASK = ("Enhance <Video 1> into a high-quality version of the same video. "
         "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
         "Restore fine details and remove degradation. Do not continue or change the scene.")


# ---------------------------------------------------------------------------
# weight fetching
# ---------------------------------------------------------------------------

def _hub_file(filename, cache_dir, local_files_only):
    """Cache-first hf_hub_download of a file of the IC repo."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(_IC_REPO, filename, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        if local_files_only:
            raise
        return hf_hub_download(_IC_REPO, filename, cache_dir=cache_dir)


def _read_range(url, start, length, token):
    headers = {"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as response:
        if response.status != 206:
            raise RuntimeError("server ignored the Range request; refusing a full checkpoint download")
        data = response.read()
    if len(data) != length:
        raise RuntimeError("truncated range response")
    return data


def _range_tensors(repo, filename, keys, revision=None):
    """{key: raw bytes + header entry} for some tensors of a remote safetensors file, by HTTP byte range."""
    from huggingface_hub import get_token, hf_hub_url
    url, token = hf_hub_url(repo, filename, revision=revision), get_token()
    header_length, = struct.unpack("<Q", _read_range(url, 0, 8, token))
    header = json.loads(_read_range(url, 8, header_length, token))
    out = {}
    for key in keys:
        entry = header[key]
        start, end = entry["data_offsets"]
        raw = bytearray()
        step = 48 << 20
        for a in range(start, end, step):
            raw += _read_range(url, 8 + header_length + a, min(a + step, end) - a, token)
        out[key] = (bytes(raw), entry)
    return out


def _aux_dir(cache_dir):
    from huggingface_hub import constants
    path = os.path.join(cache_dir or constants.HF_HUB_CACHE, "pallaidium_h3_ic")
    os.makedirs(path, exist_ok=True)
    return path


def _load_time_embedder_base(cache_dir, local_files_only):
    """The four base time-embedder tensors (fp32), cached beside the HF cache after one byte-range fetch."""
    import torch
    from safetensors.torch import load_file, save_file
    path = os.path.join(_aux_dir(cache_dir), "time_embedder_base.safetensors")
    if not os.path.isfile(path):
        if local_files_only:
            raise FileNotFoundError(f"{path} is missing and local_files_only is set")
        print("Open-H3 Regenerate IC: fetching the base time-embedder weights (65 MB, byte range)…")
        raw = _range_tensors(_IC_REPO, _TE_SHARD, _TE_KEYS)
        dtypes = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
        tensors = {k: torch.frombuffer(bytearray(b), dtype=dtypes[e["dtype"]]).reshape(e["shape"]).float().clone()
                   for k, (b, e) in raw.items()}
        save_file(tensors, path + ".tmp")
        os.replace(path + ".tmp", path)
    return load_file(path)


def _load_curve_table(cache_dir, local_files_only):
    """The Comfy [1025, 8] adaln_t_table the Sharp branches were fitted to (SHA256-pinned)."""
    import torch
    path = os.path.join(_aux_dir(cache_dir), "lms_curve.bin")
    if not os.path.isfile(path):
        if local_files_only:
            raise FileNotFoundError(f"{path} is missing and local_files_only is set")
        print("Open-H3 Regenerate IC: fetching the Sharp curve table (32 KB, byte range)…")
        raw, entry = _range_tensors(_CURVE["repo"], _CURVE["file"], [_CURVE["tensor"]], _CURVE["revision"])[_CURVE["tensor"]]
        assert entry["dtype"] == "F32" and entry["shape"] == [1025, 8], entry
        if hashlib.sha256(raw).hexdigest() != _CURVE["sha256"]:
            raise ValueError("adaln_t_table SHA256 mismatch")
        with open(path + ".tmp", "wb") as f:
            f.write(raw)
        os.replace(path + ".tmp", path)
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=torch.float32).reshape(1025, 8).clone()


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

def _make_modules():
    """The torch module classes, built lazily so importing the plugin file never imports torch."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class LoraLinear(nn.Module):
        """y = W x + B (A x): the concatenated-rank HyperFlow + Sharp + IC LoRA, scale folded into B."""

        def __init__(self, base, a, b):
            super().__init__()
            self.base = base
            self.lora_A = nn.Parameter(a, requires_grad=False)
            self.lora_B = nn.Parameter(b, requires_grad=False)

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            out = self.base(x)
            a, b = self.lora_A.to(x.dtype), self.lora_B.to(x.dtype)
            # The packed sequence is ~100-150k rows: add the LoRA branch into `out` in row chunks, in place,
            # instead of materialising a second [rows, out_features] tensor and a third for the sum.
            rows = x.shape[-2]
            if rows <= _FF_CHUNK_ROWS:
                return out.add_(F.linear(F.linear(x, a), b).to(out.dtype))
            for i in range(0, rows, _FF_CHUNK_ROWS):
                out[..., i:i + _FF_CHUNK_ROWS, :].add_(F.linear(F.linear(x[..., i:i + _FF_CHUNK_ROWS, :], a), b).to(out.dtype))
            return out

    class CurveContext:
        """Comfy-table curve coordinates of the model timesteps, shared by the 50 Sharp branches."""

        def __init__(self, table, strength):
            self.table = table
            self.strength = float(strength)
            self.coords = None

        def pre(self, module, args, kwargs):
            t = kwargs["timestep"].float()
            assert t.min() >= 0 and t.max() <= 1
            table = self.table.to(t.device)
            pos = t * (len(table) - 1)
            lo = pos.floor().long().clamp(max=len(table) - 2)
            self.coords = torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))

    class CurveLinear(nn.Module):
        """Pruned AdaLN projection plus the Sharp curve branch: base(x) + strength * B(A(curve(t)))."""

        def __init__(self, base, a, b, context):
            super().__init__()
            self.base, self.context = base, context
            self.register_buffer("curve_a", a)
            self.register_buffer("curve_b", b)

        @property
        def weight(self):
            return self.base.weight

        def forward(self, x):
            out = self.base(x.to(self.base.weight.dtype))
            c = self.context.coords
            assert c is not None and c.shape[0] == x.shape[0], (None if c is None else c.shape, x.shape)
            delta = F.linear(F.linear(c.to(self.curve_a.device), self.curve_a.float()), self.curve_b.float())
            return out + delta.to(out.dtype) * self.context.strength

    class HyperFlowCurveEmbedder(nn.Module):
        """Stands in for the pruned model's table lookup: emb(t, r) of HyperFlow's two-time embedder,
        projected onto the pruned rank-8 curve basis, one row of coordinates per distinct timestep."""

        def __init__(self, basis, te_base, te_lora, endpoint_lora, gate):
            super().__init__()
            basis = basis.double()
            pinv = basis.T @ torch.linalg.inv(basis @ basis.T)            # [2688, 8]: coords = silu(emb) @ pinv
            self.register_buffer("proj", pinv.float())
            self.gate = float(gate)
            self.endpoint_timesteps = None
            for tag, lora in (("t", te_lora), ("r", endpoint_lora)):
                for i in (1, 2):
                    w = te_base[f"time_embedder.linear_{i}.weight"]
                    if lora is not None:
                        w = w + lora[i][1] @ lora[i][0]
                    self.register_buffer(f"w{i}_{tag}", w.float().contiguous())
                    self.register_buffer(f"b{i}_{tag}", te_base[f"time_embedder.linear_{i}.bias"].float().contiguous())

        @staticmethod
        def _sinusoid(t):
            half = 128
            exponent = -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
            e = t.float()[:, None] * torch.exp(exponent)[None]
            e = torch.cat([e.sin(), e.cos()], dim=-1)
            return torch.cat([e[:, half:], e[:, :half]], dim=-1)          # flip_sin_to_cos

        def _embed(self, t, tag):
            h = F.linear(self._sinusoid(t), getattr(self, f"w1_{tag}"), getattr(self, f"b1_{tag}"))
            return F.linear(F.silu(h), getattr(self, f"w2_{tag}"), getattr(self, f"b2_{tag}"))

        def forward(self, timestep):
            if self.endpoint_timesteps is None:
                raise RuntimeError("HyperFlow endpoint timesteps not set for this forward")
            t = timestep.float().flatten().clamp(0.0, 1.0)
            r = self.endpoint_timesteps.to(t.device).float().flatten().clamp(0.0, 1.0)
            assert r.shape == t.shape, (r.shape, t.shape)
            et = self._embed(t, "t")
            emb = et + self.gate * (self._embed(r, "r") - et)
            return F.silu(emb) @ self.proj

    return LoraLinear, CurveContext, CurveLinear, HyperFlowCurveEmbedder


def _convert_sharp(path):
    """LMS Sharp (ComfyUI keys) -> diffusers module names; fused qkv split, fc1 rows [gate, value] -> [value, gate],
    the 50 AdaLN branches (input = 8-D curve) kept apart. Follows NVIDIA's ref2va_ic/sharp.py."""
    import torch
    from safetensors.torch import load_file
    raw = load_file(str(path), device="cpu")
    roots = sorted(k[:-len(".lora_A.weight")] for k in raw if k.endswith(".lora_A.weight"))
    linears, curves = {}, {}
    for src in roots:
        a, b = raw[src + ".lora_A.weight"], raw[src + ".lora_B.weight"]
        name = src.removeprefix("diffusion_model.").replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if name.startswith("blocks."):
            name = "transformer_blocks." + name[len("blocks."):]
        name = name.replace("final_layer.adaln_proj.linear", "norm_out.linear").replace(".attn.out_proj", ".attn.to_out.0")
        if name.endswith(".attn.qkv_proj"):
            for tag, part in zip(("to_q", "to_k", "to_v"), b.chunk(3, dim=0)):
                linears[name[:-len("qkv_proj")] + tag] = (a, part.contiguous())
        elif name.endswith(".mlp.fc1"):
            gate, value = b.chunk(2, dim=0)
            linears[name[:-len(".mlp.fc1")] + ".ff.net.0.proj"] = (a, torch.cat([value, gate], dim=0))
        elif name.endswith(".mlp.fc2"):
            linears[name[:-len(".mlp.fc2")] + ".ff.net.2"] = (a, b)
        elif ".adaln_proj.linear" in name and a.shape[1] == 8:
            curves[name] = (a, b)
        else:
            linears[name] = (a, b)
    return linears, curves


def _collect_lora_parts(hyperflow_path, ic_path, sharp_linears, strength):
    """Concatenate every adapter's factors per target linear: A = [A_hf; A_sharp; A_ic], B = [B_hf, s*B_sharp, B_ic]
    (HyperFlow alpha == rank == 256 and IC alpha == rank == 384, so both have scale 1). bf16 at rest, the
    activation dtype the reference casts to at run time anyway. Returns ({module: (A, B)}, HyperFlow header,
    {1|2: (A, B)} of the two time-embedder LoRAs, same for the endpoint embedder)."""
    import re
    import torch
    from safetensors import safe_open
    key_re = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<m>[AB])\.weight$")
    parts = {}
    time_lora = {"time_embedder": {}, "endpoint_time_embedder": {}}
    meta = None

    def add(module, matrix, tensor, scale=1.0):
        entry = parts.setdefault(module, {"A": [], "B": []})
        entry[matrix].append((tensor * scale if matrix == "B" else tensor).to(torch.bfloat16).contiguous())

    with safe_open(str(hyperflow_path), framework="pt", device="cpu") as f:
        meta = f.metadata()
        assert meta.get("hyperflow") == "true" and float(meta["lora_alpha"]) == float(meta["lora_rank"]), meta
        for key in f.keys():
            m = key_re.match(key)
            assert m, key
            module = m["module"]
            tensor = f.get_tensor(key)
            if module.startswith(("time_embedder.", "endpoint_time_embedder.")):
                owner, linear = module.split(".", 1)
                idx = int(linear.removeprefix("linear_"))
                time_lora[owner].setdefault(idx, [None, None])[0 if m["m"] == "A" else 1] = tensor.float()
            else:
                add(module, m["m"], tensor.float())
    for module, (a, b) in sharp_linears.items():
        add(module, "A", a.float())
        add(module, "B", b.float(), strength)
    with safe_open(str(ic_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            m = key_re.match(key)
            assert m, key
            add(m["module"], m["m"], f.get_tensor(key))
    merged = {}
    for module in list(parts):      # pop as we go: peak host RAM stays ~1x the adapter size, not 2x
        entry = parts.pop(module)
        assert len(entry["A"]) == len(entry["B"]), module
        merged[module] = (torch.cat(entry["A"], 0), torch.cat(entry["B"], 1))
    return merged, meta, time_lora["time_embedder"], time_lora["endpoint_time_embedder"]


def _install_lora(model, merged):
    """Wrap every target linear of `model` (the pruned SDNQ transformer_ref) with its concatenated LoRA."""
    LoraLinear = _make_modules()[0]
    for name, (a, b) in merged.items():
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = parent[int(child)] if child.isdigit() else getattr(parent, child)
        assert a.shape[1] == getattr(base, "in_features", a.shape[1]) and b.shape[0] == getattr(base, "out_features", b.shape[0]), name
        wrapped = LoraLinear(base, a, b)
        if child.isdigit():
            parent[int(child)] = wrapped
        else:
            setattr(parent, child, wrapped)
    return len(merged)


def _lean_attention(attn, x, rotary_emb, rows=_FF_CHUNK_ROWS):
    """MiniMaxH3AttnProcessor with every row-wise op chunked: q/k/v are projected, qk-normed and rotated
    `rows` at a time into preallocated tensors, and `x` is released before the (full-sequence) attention."""
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
    cos, sin = rotary_emb
    batch, length, _ = x.shape
    q = k = v = None
    for i in range(0, length, rows):
        xi = x[:, i:i + rows]
        if attn.fused_projections:
            qi, ki, vi = attn.to_qkv(xi).chunk(3, dim=-1)
        else:
            qi, ki, vi = attn.to_q(xi), attn.to_k(xi), attn.to_v(xi)
        qi = attn.norm_q(qi.unflatten(-1, (attn.heads, -1)))
        ki = attn.norm_k(ki.unflatten(-1, (attn.heads, -1)))
        vi = vi.unflatten(-1, (attn.heads, -1))
        qi = _apply_rotary_emb(qi, cos[i:i + rows], sin[i:i + rows])
        ki = _apply_rotary_emb(ki, cos[i:i + rows], sin[i:i + rows])
        if q is None:
            q = qi.new_empty((batch, length, *qi.shape[2:]))
            k = ki.new_empty((batch, length, *ki.shape[2:]))
            v = vi.new_empty((batch, length, *vi.shape[2:]))
        q[:, i:i + rows], k[:, i:i + rows], v[:, i:i + rows] = qi, ki, vi
        del qi, ki, vi, xi
    del x
    processor = attn.processor
    out = dispatch_attention_fn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                                backend=getattr(processor, "_attention_backend", None), parallel_config=None)
    dtype = q.dtype
    del q, k, v
    out = out.flatten(2, 3).to(dtype)
    result = None
    for i in range(0, length, rows):
        yi = attn.to_out[1](attn.to_out[0](out[:, i:i + rows]))
        if result is None:
            result = yi.new_empty((batch, length, yi.shape[-1]))
        result[:, i:i + rows] = yi
    return result


def _lean_block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None,
                        rows=_FF_CHUNK_ROWS):
    """MiniMaxH3TransformerBlock.forward with the per-row work done `rows` at a time: the reference builds
    several full-sequence [rows, hidden] temporaries for the AdaLN modulation, the gated residuals and the
    q/k/v norm + rotary (~15 GB for one block at 2K); only the attention itself needs the whole sequence.
    Same ops in the same dtype, so the result matches up to GEMM tiling. Updates `hidden_states` in place."""
    import torch
    assert attention_mask is None
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
    length = hidden_states.shape[1]

    def modulated(norm, scale, shift):
        out = torch.empty_like(hidden_states)
        for i in range(0, length, rows):
            ix = adaln_indices[i:i + rows]
            out[:, i:i + rows] = norm(hidden_states[:, i:i + rows]) * (1.0 + scale.index_select(0, ix)) \
                + shift.index_select(0, ix)
        return out

    attn_output = _lean_attention(self.attn, modulated(self.norm1, scale_msa, shift_msa), rotary_emb, rows)
    for i in range(0, length, rows):
        ix = adaln_indices[i:i + rows]
        hidden_states[:, i:i + rows] = hidden_states[:, i:i + rows] + gate_msa.index_select(0, ix) * attn_output[:, i:i + rows]
    del attn_output
    for i in range(0, length, rows):
        ix = adaln_indices[i:i + rows]
        h = hidden_states[:, i:i + rows]
        n = self.norm2(h) * (1.0 + scale_mlp.index_select(0, ix)) + shift_mlp.index_select(0, ix)
        hidden_states[:, i:i + rows] = h + gate_mlp.index_select(0, ix) * self.ff(n)
    return hidden_states


def _install_lean_blocks(model):
    import types
    for block in model.transformer_blocks:
        block.forward = types.MethodType(_lean_block_forward, block)


def _chunk_feed_forward(model, rows=_FF_CHUNK_ROWS):
    """The packed sequence is ~100-150k rows at 1080p / 2K; the feed-forward's 28672-wide activation would be
    several GB, so run it in row chunks (rows are independent in the FF)."""
    import torch
    for block in model.transformer_blocks:
        ff = block.ff
        original = ff.forward

        def chunked(hidden_states, *args, _orig=original, **kwargs):
            if hidden_states.shape[-2] <= rows:
                return _orig(hidden_states, *args, **kwargs)
            return torch.cat([_orig(c, *args, **kwargs) for c in hidden_states.split(rows, dim=-2)], dim=-2)

        ff.forward = chunked


# ---------------------------------------------------------------------------
# draft I/O
# ---------------------------------------------------------------------------

def _cover(x, height, width):
    import torch.nn.functional as F
    h, w = x.shape[-2:]
    if (h, w) == (height, width):
        return x
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    x = F.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False, antialias=scale < 1)
    i, j = (sh - height) // 2, (sw - width) // 2
    return x[..., i:i + height, j:j + width]


def _read_draft(path, height, width):
    """124 frames picked at 24 fps (source frame int(i * fps / 24)), cover-resized and centre-cropped onto the
    canvas. -> float [124, 3, h, w] in [0, 1], number of real frames, source fps. The tail repeats the last frame."""
    import av
    import torch
    out = torch.empty((_FRAMES, 3, height, width), dtype=torch.float32)
    valid = 0
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 24.0)
        wanted = [int(i * fps / 24.0) for i in range(_FRAMES)]
        for index, frame in enumerate(video.decode(stream)):
            if index < wanted[valid]:
                continue
            x = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)[None].float() / 255.0
            x = _cover(x, height, width)
            while valid < _FRAMES and wanted[valid] == index:
                out[valid] = x[0]
                valid += 1
            if valid == _FRAMES:
                break
    if valid < _MIN_REAL:
        raise ValueError(f"{os.path.basename(str(path))}: {valid} frames at 24 fps; the model needs at least "
                         f"{_MIN_REAL} (about 4.7 s)")
    out[valid:] = out[valid - 1]
    return out.clamp(0, 1), valid, fps


from ._h3_conditioner import conditioner_hidden as _conditioner_hidden  # noqa: E402


def _peak(label):
    """Log (and reset) the CUDA peak since the last call — a phase that spills past the card's VRAM shows here."""
    import torch
    if torch.cuda.is_available():
        print(f"Open-H3 Regenerate IC: [{label}] peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB allocated, "
              f"{torch.cuda.max_memory_reserved() / 1e9:.1f} GB reserved")
        torch.cuda.reset_peak_memory_stats()


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _write_video(frames, path, audio_from=None, crf=12):
    """uint8 [T, H, W, 3] -> H.264 24 fps, the first audio stream of `audio_from` muxed in (cut to the video)."""
    import numpy as np
    t, h, w, _ = frames.shape
    cmd = [_ffmpeg_exe(), "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", "24", "-i", "-"]
    if audio_from is not None:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


# ---------------------------------------------------------------------------
# plugin
# ---------------------------------------------------------------------------

class OpenH3Regenerate2KPlugin(ModelPlugin):
    MODEL_ID     = "Efficient-Large-Model/Open-H3-Regenerate-2K-IC"
    DISPLAY_NAME = "Open-H3 Regenerate IC (video upscale to 1080p)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = ("NVIDIA's in-context refiner: regenerates a MiniMax-H3 draft (>= 4.7 s, 24 fps, 16:9) at 1080p, "
                    "keeping faces, lip sync and lettering. Slow: ~20 min per clip on an RTX 4090 (queue it). "
                    "Steps 3 (default), 4 or 8. Optional prompt = the prompt that made the draft. "
                    "Height >= 800 -> 1080p, below -> 768p (from a 384/512/576p draft).")

    INPUTS       = InputSpec.PROMPT | InputSpec.VIDEO
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.VIDEO_STRIP, UISection.RESOLUTION, UISection.STEPS, UISection.SEED,
    ]
    PARAMS       = ParamSpec(width=1920, height=1080, steps=3)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "av", "sdnq", "einops"]

    supports_inpaint          = False
    supports_img2img          = False
    requires_input_strip      = True
    uses_standard_input_strip = False
    show_enhance              = False
    supports_batch            = False

    # -- load ---------------------------------------------------------------

    def load(self, prefs, scene, **kw):
        import psutil
        import torch
        from diffusers import ModularPipeline
        from diffusers.hooks import apply_group_offloading
        from sdnq import SDNQConfig  # noqa: F401 — registers the SDNQ weight loader
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Same base repos and RAM rule as the MiniMax H3 plugin (shared HF cache, no second download).
        bits = 4 if (low_vram() or total_ram_gb < 64) else 8
        repo_id = f"OzzyGT/MiniMax_H3_sdnq_{bits}bit_pruned"
        dtype = torch.bfloat16
        wanted = ("transformer_ref", "text_encoder", "tokenizer", "processor", "vae")

        print(f"Loading Open-H3 Regenerate IC on {repo_id} (transformer_ref only)…")
        try:
            pipe = ModularPipeline.from_pretrained(repo_id, cache_dir=_cache_dir, local_files_only=True)
            from_cache = True
        except (OSError, ValueError):
            if _lfo:
                raise
            pipe = ModularPipeline.from_pretrained(repo_id, cache_dir=_cache_dir, local_files_only=False)
            from_cache = False
        from ._minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
        from ._minimax_h3_vae_fp16 import MiniMaxH3Fp16DecoderVae
        for _name, _cls in (("transformer_ref", MiniMaxH3PrunedTransformer3DModel), ("vae", MiniMaxH3Fp16DecoderVae)):
            pipe._component_specs[_name].type_hint = _cls

        def _todo():
            return [n for n in wanted if pipe._component_specs[n].default_creation_method == "from_pretrained"
                    and getattr(pipe, n, None) is None]

        if from_cache:
            for name in _todo():
                try:
                    local_dir = _cached_snapshot_dir(pipe._component_specs[name].pretrained_model_name_or_path, _cache_dir)
                except OSError:
                    continue  # not cached — fetched online below
                pipe.load_components(names=name, pretrained_model_name_or_path=local_dir, dtype=dtype,
                                     cache_dir=_cache_dir, local_files_only=True)
        rest = _todo()
        if rest:
            if _lfo:
                raise RuntimeError(f"Open-H3 Regenerate IC: not in the local HF cache and local_files_only is set: {rest}")
            print(f"Open-H3 Regenerate IC: not in local cache, fetching online: {rest}")
            pipe.load_components(names=rest, dtype=dtype, cache_dir=_cache_dir, local_files_only=False)
        missing = [n for n in wanted if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(
                f"Open-H3 Regenerate IC: component(s) failed to load: {missing}. Check the console above for the "
                "underlying error (a Windows 'paging file is too small' error means a large safetensors file couldn't "
                "be memory-mapped — increase the Windows pagefile or free RAM, and try again)."
            )

        # fp16-decoder VAE dtype alignment — see minimax_h3.py.
        vae_compute_dtype = pipe.vae.decoder.proj_in.weight.dtype
        pipe.vae.post_quant_conv.to(vae_compute_dtype)
        for name, module in pipe.vae.decoder.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) and name.rsplit(".", 1)[-1].startswith("norm"):
                module.to(torch.float32)
        # 256 / 64 tiling, as the reference (the 2K decode and the fp32 encode would not fit otherwise).
        pipe.vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                               tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)

        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            pipe.transformer_ref = apply_sdnq_options_to_model(pipe.transformer_ref, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        # ---- adapters ------------------------------------------------------
        t0 = time.time()
        paths = {key: _hub_file(name, _cache_dir, _lfo) for key, name in _IC_FILES.items()}
        sharp_linears, sharp_curves = _convert_sharp(paths["sharp"])
        assert len(sharp_curves) == 50, len(sharp_curves)
        merged, hf_meta, te_lora, ep_lora = _collect_lora_parts(paths["hyperflow"], paths["ic_lora"], sharp_linears,
                                                                _SHARP_STRENGTH)
        assert len(te_lora) == 2 and len(ep_lora) == 2 and all(None not in v for v in te_lora.values())
        count = _install_lora(pipe.transformer_ref, merged)
        del merged, sharp_linears
        LoraLinear, CurveContext, CurveLinear, HyperFlowCurveEmbedder = _make_modules()
        model = pipe.transformer_ref
        # HyperFlow's two-time embedder over the pruned curve basis.
        te_base = _load_time_embedder_base(_cache_dir, _lfo)
        basis = model.time_embedder.basis
        if basis.shape != (8, 2688) or not float(basis.abs().sum()) > 0:
            raise RuntimeError("Open-H3 Regenerate IC: the pruned transformer_ref did not load its AdaLN curve basis "
                               "(transformer_ref/adaln_basis.safetensors) — re-download OzzyGT/MiniMax_H3_sdnq_*bit_pruned.")
        model.time_embedder = HyperFlowCurveEmbedder(basis.detach().clone(), te_base, te_lora, ep_lora, _HF_GATE)
        # Sharp's AdaLN curve branches on the Comfy table.
        context = CurveContext(_load_curve_table(_cache_dir, _lfo), _SHARP_STRENGTH)
        for name, (a, b) in sharp_curves.items():
            parent, key = name.rsplit(".", 1)
            holder = model.get_submodule(parent)
            setattr(holder, key, CurveLinear(getattr(holder, key), a.half().contiguous(), b.half().contiguous(), context))
        model.register_forward_pre_hook(context.pre, with_kwargs=True)
        _chunk_feed_forward(model)
        if os.environ.get("PALLAIDIUM_H3IC_REFERENCE_BLOCKS") != "1":
            _install_lean_blocks(model)   # before group offload, so the offload hooks wrap the lean forward
        model.requires_grad_(False)
        sigmas_raw = [float(s) for s in json.loads(hf_meta["hyperflow_sigmas"])]
        shift = float(hf_meta["hyperflow_video_shift"])
        grid = [shift * s / (1.0 + (shift - 1.0) * s) for s in sigmas_raw]      # 9 points, 8 steps
        print(f"Open-H3 Regenerate IC: {count} linears carry HyperFlow + Sharp + IC LoRA in {time.time() - t0:.0f} s; "
              f"sigma grid {[round(s, 4) for s in grid]}")
        from safetensors.torch import load_file
        silence = load_file(paths["silence"])["audio_latents"]

        # ---- upscaler ------------------------------------------------------
        upscaler = _load_upscaler(paths["upscaler"])

        if gfx_device == "mps":
            raise RuntimeError("Open-H3 Regenerate IC needs CUDA")

        onload_device = torch.device(gfx_device)
        offload_device = torch.device("cpu")
        from ...utils.helpers import ensure_group_offload_keep_cpu_weights
        ensure_group_offload_keep_cpu_weights()
        offload = dict(onload_device=onload_device, offload_device=offload_device, use_stream=False, low_cpu_mem_usage=True)
        model.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        # Block-level on the conditioner's two block stacks. Leaf-level does not offload the SDNQ linears
        # (not leaf types it recognises), so the whole ~21 GB 4-bit conditioner sat on the GPU during the
        # prompt encode (measured 23.4 GB peak for a 6k-token presentation).
        apply_group_offloading(pipe.text_encoder.model.language_model, offload_type="block_level",
                               num_blocks_per_group=1, **offload)
        apply_group_offloading(pipe.text_encoder.model.visual, offload_type="block_level",
                               num_blocks_per_group=1, **offload)
        # Qwen3VLModel looks the token embeddings up *before* calling language_model.forward, i.e. before the
        # offload hook has onloaded them: run the lookup wherever the table is and return to the ids' device.
        embed = pipe.text_encoder.model.language_model.embed_tokens
        _embed_forward = embed.forward
        embed.forward = lambda ids, _f=_embed_forward, _m=embed: _f(ids.to(_m.weight.device)).to(ids.device)
        _onload_vae_per_call(pipe.vae, onload_device)

        return {"pipe": pipe, "bits": bits, "grid": grid, "silence": silence, "upscaler": upscaler, "context": context}

    # -- generate -----------------------------------------------------------

    @staticmethod
    def _plan(steps, grid):
        """(sampler name, sigma grid incl. terminal 0, init) for the Steps slider: 3 -> up3, 4 -> up4, 8 -> hf8."""
        name = {3: "up3", 4: "up4", 8: "hf8"}.get(int(steps))
        if name is None:
            name = "up3" if steps <= 3 else ("up4" if steps <= 5 else "hf8")
            print(f"Open-H3 Regenerate IC: Steps {steps} -> sampler {name} (supported: 3 = up3, 4 = up4, 8 = hf8)")
        if name == "hf8":
            return name, grid, "noise"
        return name, grid[8 - int(name[2:]):], "upscaled"

    @staticmethod
    def _geometry(height, draft_h):
        """(output name, canvas h, w, target h, w, rows kept) from the requested output height and the draft."""
        if height >= 1300:
            # 2K (~150k packed rows) overflows a 24 GB card into shared memory and crawls (measured on an
            # RTX 4090: step 1 > 25 min at ~4 % GPU use). Opt in on bigger cards only.
            if os.environ.get("PALLAIDIUM_H3IC_ALLOW_2K") == "1":
                return ("2k", *_RESOLUTIONS["2k"])
            print("Open-H3 Regenerate IC: 2K does not fit in 24 GB VRAM; rendering 1080p "
                  "(set PALLAIDIUM_H3IC_ALLOW_2K=1 on a larger GPU)")
        if height >= 800:
            return ("1080p", *_RESOLUTIONS["1080p"])
        lq = max((h for h in _LQ_768P if h <= max(draft_h, 384)), default=384)
        return ("768p", *_LQ_768P[lq], 768, 1344, 768)

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import numpy as np
        import torch
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3Ref2VAPrepareLayoutStep, MiniMaxH3SetTimestepsStep, patchify_video_latents)
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            MiniMaxH3Ref2VATextEncoderStep)
        from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference

        pipe = pipe_obj["pipe"]
        grid, silence, upscaler, context = pipe_obj["grid"], pipe_obj["silence"], pipe_obj["upscaler"], pipe_obj["context"]
        model = pipe.transformer_ref
        device = torch.device(gfx_device)
        src = inputs.video_path
        if not src:
            raise ValueError("Open-H3 Regenerate IC needs an input video strip (the draft to regenerate).")
        seed = int(inputs.seed)
        caption = (inputs.prompt or "").strip()

        try:
            import av
            with av.open(src) as c:
                src_h = c.streams.video[0].codec_context.height
        except Exception:
            src_h = 768
        res_name, lq_h, lq_w, hq_h, hq_w, keep = self._geometry(int(inputs.height), src_h)
        sampler, sigmas, init = self._plan(int(inputs.steps), grid)
        print(f"Open-H3 Regenerate IC: {res_name} ({hq_w}x{keep}) from a {lq_w}x{lq_h} draft canvas, sampler {sampler}, "
              f"seed {seed}, caption {'set' if caption else 'empty'}")

        # 1. draft -> latent + prompt embeddings ------------------------------------------------------------
        self.set_phase(inputs, "Reading draft")
        pixels, real, fps = _read_draft(src, lq_h, lq_w)
        print(f"Open-H3 Regenerate IC: draft {real} frames at 24 fps (source {fps:.2f} fps)")

        self.set_phase(inputs, "Encoding draft")
        vae = pipe.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, 24, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 24, 1, 1, 1)
        pm = torch.tensor(pipe.pixel_mean).view(1, 3, 1, 1, 1)
        ps = torch.tensor(pipe.pixel_std).view(1, 3, 1, 1, 1)
        with torch.inference_mode():
            x = (pixels.permute(1, 0, 2, 3)[None] - pm) / ps
            z = vae.encode(x.to(device), return_dict=False)[0].sample(
                generator=torch.Generator().manual_seed(seed % (1 << 62)))
            draft_latent = ((z.cpu().float() - mean) / std)[0].bfloat16()
        assert torch.isfinite(draft_latent).all()
        del x, z
        _peak("draft encode")

        self.set_phase(inputs, "Encoding prompt")
        frames_u8 = (pixels * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        prompt = _TASK + "\n\nScene description: " + caption
        references = [MiniMaxH3VideoReference(frames=frames_u8, fps=24.0)]
        step = MiniMaxH3Ref2VATextEncoderStep()
        with torch.inference_mode():
            vision_inputs, img_counts, vid_counts, vid_stamps = step._gather_vision_features(pipe.processor, references, pipe.fps)
            token_ids, token_tags = step._build_presentation(pipe.tokenizer, prompt, references, img_counts, vid_counts,
                                                             vid_stamps, text_tag=pipe.text_tag, video_tag=pipe.video_tag)
            emb = _conditioner_hidden(pipe.text_encoder, pipe.processor, token_ids, vision_inputs,
                                      layer=pipe.text_encoder_layer, device=device, dtype=pipe.text_encoder.dtype)
        prompt_embeds = emb[0].detach().cpu().bfloat16()
        tags = torch.tensor(token_tags, dtype=torch.long)
        prefix = len(prompt_embeds) - len(pipe.tokenizer.encode(prompt, add_special_tokens=False))
        assert 0 < prefix < len(tags) and prompt_embeds.shape == (len(tags), 5120) and torch.isfinite(prompt_embeds).all()
        del emb, vision_inputs, frames_u8
        torch.cuda.empty_cache()
        _peak("prompt encode")

        # 2. packed sequence ----------------------------------------------------------------------------------
        frames_lat = draft_latent.shape[1]
        target_shape = [24, frames_lat, hq_h // 16, hq_w // 16]
        rows = frames_lat * (hq_h // 16 // 2) * (hq_w // 16 // 2)
        values = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            tags, references, [draft_latent[None]], [],
            frames_lat, target_shape[2], target_shape[3], silence.shape[-1], _PATCH, 2, 2, 0)
        position_ids, token_tags_t, video_indices, audio_indices, text_indices, n_cond_v, n_cond_a = values
        assert n_cond_a == 0
        n = n_cond_v
        ref_rows = patchify_video_latents(draft_latent[None].float(), _PATCH)[None]
        assert n == ref_rows.shape[1] and video_indices.numel() == n + rows, (n, ref_rows.shape, video_indices.numel(), rows)
        # channel-major audio rows [1, 2 * A, C] from the [2, C, A] silence latent
        audio = silence[None].permute(0, 1, 3, 2).reshape(1, 2 * silence.shape[-1], silence.shape[1]).contiguous()
        length = int(token_tags_t.numel())

        # 3. upscaled start ------------------------------------------------------------------------------------
        up = None
        if init == "upscaled":
            self.set_phase(inputs, "Upscaling draft latent")
            up = _upscale_latent(upscaler, draft_latent, target_shape, device)          # [24, T, H, W] fp32 (CPU)
            up = patchify_video_latents(up[None], _PATCH)[None]                          # [1, rows, 96]
            _peak("latent upscale")

        # 4. sampling ------------------------------------------------------------------------------------------
        rng = torch.Generator(device="cpu").manual_seed(seed % (1 << 62))
        anchor = (0.999 * ref_rows.float() + 0.001 * torch.randn(ref_rows.shape, generator=rng)).to(device)
        x = torch.randn((1, rows, 96), generator=rng, dtype=torch.float32)
        if init == "upscaled":
            x = (1.0 - sigmas[0]) * up.float() + sigmas[0] * x
        x = x.to(device)
        audio_d = audio.to(device)
        prompt_d = prompt_embeds[None].to(device)
        pos_d, tags_d = position_ids.to(device), token_tags_t.to(device)
        vid_d, aud_d, txt_d = video_indices.to(device), audio_indices.to(device), text_indices.to(device)
        del up

        total = len(sigmas) - 1
        self.set_phase(inputs, f"Regenerating ({sampler}, {total} steps)")
        t_start = time.time()
        with torch.inference_mode():
            for i, (sigma, nxt) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                if inputs.should_cancel is not None and inputs.should_cancel():
                    raise RuntimeError("Cancelled")
                sig = float(torch.as_tensor(sigma, dtype=torch.float32).item())
                t_gen = 1.0 - sig
                times, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                    video_indices=video_indices, audio_indices=audio_indices,
                    num_condition_video_rows=n_cond_v, num_condition_audio_rows=n_cond_a,
                    num_text_tokens=int(text_indices.numel()), video_timestep=t_gen, audio_timestep=1.0,
                    condition_video_timestep=max(t_gen, _REFERENCE_T), condition_audio_timestep=1.0)
                # HyperFlow endpoint: generated rows aim at r = 1 - sigma_next, every other row keeps r = t.
                tt = times.float()
                gen = torch.isclose(tt, torch.tensor(t_gen, dtype=torch.float32), atol=1e-6, rtol=0.0)
                assert int(gen.sum()) == 1, (tt.tolist(), sig)
                r = tt.clone()
                r[gen] = 1.0 - nxt
                model.time_embedder.endpoint_timesteps = r.to(device)
                v = model(hidden_states=torch.cat([anchor, x], 1), audio_hidden_states=audio_d,
                          encoder_hidden_states=prompt_d, timestep=times.to(device), timestep_indices=indices.to(device),
                          token_tags=tags_d, position_ids=pos_d, video_indices=vid_d, audio_indices=aud_d,
                          text_indices=txt_d, attention_kwargs={}, return_dict=False)[0]
                x = x + (sig - nxt) * v[:, n:].float()
                del v
                torch.cuda.empty_cache()
                if inputs.progress_fn is not None:
                    try:
                        inputs.progress_fn(i + 1, total)
                    except Exception:
                        pass
                print(f"Open-H3 Regenerate IC: step {i + 1}/{total} (sigma {sig:.3f} -> {nxt:.3f}) "
                      f"{time.time() - t_start:.0f} s elapsed")
                _peak(f"step {i + 1}")
        assert torch.isfinite(x).all(), "non-finite latents"
        latent = _unpatch_rows(x.cpu(), target_shape).to(torch.bfloat16)
        del x, anchor, audio_d, prompt_d
        torch.cuda.empty_cache()

        # 5. decode + mux ---------------------------------------------------------------------------------------
        self.set_phase(inputs, "Decoding")
        with torch.inference_mode():
            z = latent[None].float() * std + mean
            pixels_out = vae.decode(z.to(device), return_dict=False)[0][0]               # [3, T, H, W]
            pmv, psv = pm[0].to(pixels_out.device), ps[0].to(pixels_out.device)
            chunks = []
            for t in range(0, pixels_out.shape[1], 8):
                piece = (pixels_out[:, t:t + 8].float() * psv + pmv).clamp(0, 1)
                chunks.append((piece * 255).round().byte().permute(1, 2, 3, 0).cpu())
            frames = torch.cat(chunks).numpy()[:real]
        del pixels_out, z
        torch.cuda.empty_cache()
        _peak("decode")
        top = (frames.shape[1] - keep) // 2
        frames = frames[:, top:top + keep]

        self.set_phase(inputs, "Saving")
        dst = solve_path(clean_filename(f"{seed}_h3_regenerate_{res_name}") + ".mp4")
        _write_video(frames, dst, audio_from=src)
        return dst


# ---------------------------------------------------------------------------
# latent helpers
# ---------------------------------------------------------------------------

def _unpatch_rows(rows, shape):
    """Inverse of the row patchify for a latent of `shape` [C, T, H, W]: [1, T*H/2*W/2, 96] -> [C, T, H, W]."""
    c, t, h, w = shape
    return rows[0].reshape(t, h // 2, w // 2, c, 1, 2, 2).permute(3, 0, 4, 1, 5, 2, 6).reshape(c, t, h, w)


def _load_upscaler(checkpoint):
    """The author's ComfyUI latent-upscaler node, with ComfyUI's registration / model lookup shimmed out."""
    import importlib.util
    import sys
    import types
    folder = types.ModuleType("folder_paths")
    folder.folder_names_and_paths = {"latent_upscale_models": ([], set())}
    folder.get_full_path_or_raise = lambda category, name: str(checkpoint)
    shims = {"folder_paths": folder, "comfy.model_management": None, "comfy_api.latest": None}
    saved = {k: sys.modules.get(k, False) for k in shims}
    sys.modules.update(shims)
    try:
        spec = importlib.util.spec_from_file_location(
            "pallaidium_h3_ic_upscaler_node", os.path.join(os.path.dirname(__file__), "_h3_ic_upscaler_node.py"))
        node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(node)
    finally:
        for k, v in saved.items():
            if v is False:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    node.io.NodeOutput = lambda value: (value,)
    return node


_STAGE = (90, 160)   # 2560x1440: larger targets take two passes, as in the reference


def _upscale_latent(node, native, shape, device):
    """Normalized draft latent [24, T, h, w] -> normalized latent [24, T, H, W] at the target size (fp32, CPU)."""
    import torch

    def one(latent, shp):
        mode = {"mode": node.UpscaleMode.TARGET_DIMENSIONS, "width": shp[3] * 16, "height": shp[2] * 16}
        with torch.inference_mode():
            result, = node.MinimaxH3LatentUpscaler3D.execute(
                {"samples": latent[None].float()}, "upscaler", mode, align=32, enable_temporal_chunking=False,
                force_unload=False, device=str(device), precision="bf16")
        out = result["samples"].float()[0]
        assert tuple(out.shape) == tuple(shp), (tuple(out.shape), shp)
        return out.cpu()

    try:
        if shape[2] > _STAGE[0]:
            native = one(native, [*shape[:2], *_STAGE])
        return one(native, shape)
    finally:
        # VRAM is needed for the transformer: park the node's cached model on the CPU. (Not the node's own
        # force_unload — that branch prints a non-ASCII check mark, which a Windows console codec can reject.)
        for cached in getattr(node, "MODEL_CACHE", {}).values():
            cached.to("cpu")
        torch.cuda.empty_cache()
