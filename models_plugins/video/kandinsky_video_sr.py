"""Kandinsky 6.0 Video SR 2.0 (distilled, 2 steps): diffusion video super-resolution x2 / x2.25 / x4.

Weights: kandinskylab/Kandinsky-6.0-VSR-2.0-distilled2steps-Diffusers (~16.6 GB bf16: latent upscaler
7.0 GB, KVAE 6.6 GB, SR DiT 3.0 GB). Runs through diffusers' native Kandinsky6SRPipeline, so the
kandinsky-6-sr package (flash-attn / magi extras, torch>=2.10 pins) is NOT installed.
No quantized / SDNQ build exists on the Hub (checked 2026-11-26); components are offloaded one at a
time with model CPU offload instead (largest single component ~7 GB).

The pipeline only supports x2, x2.25 (1.125x bilinear pre-upscale + x2) and x4.

UI mapping:
  - Scale (scene.kvsr_scale):
      AUTO       -> smallest factor reaching the Resolution target; the result is then fitted inside
                    the target with Lanczos, keeping the aspect ratio (no stretch; matches upstream
                    target_resize_mode="fit").
      2/2.25/4   -> fixed factor, native model output size, Resolution ignored.
  - Strength   -> lq_noise_scale (noise mixed into the low-res latents; 0.7 = model default,
                  higher invents more detail, lower stays closer to the source).
  - Seed       -> tile noise generator.
Input frames are trimmed to 1 + 4k (pipeline requirement) and capped at 361 (15 s @ 24 fps, the
documented maximum). Source audio is muxed back in.
"""

import os
import shutil

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename, load_video_as_np_array, low_vram
from .maxine_vsr_video import _source_has_audio, _mux_audio


_REPO = "kandinskylab/Kandinsky-6.0-VSR-2.0-distilled2steps-Diffusers"
_MAX_FRAMES = 361


_ATTN_CHUNK_BYTES = 2 * 1024 ** 3  # gathered K+V budget per chunk of query blocks


def _block_sparse_attention(q, k, v, mask, block=64):
    """Exact NABLA block-sparse attention without flex / torch.compile.

    q, k, v: (B, h, S, D); mask: (B, h, n, n) bool block mask (n = S // block). Each query block gets
    only its selected key/value blocks gathered (padded to the row max, padding masked out), so memory
    scales with the selected blocks instead of the full S x S score matrix that eager flex builds.
    """
    import torch
    import torch.nn.functional as F

    B, h, S, D = q.shape
    n = S // block
    BH = B * h
    qb = q.reshape(BH, n, block, D)
    kb = k.reshape(BH, n, block, D)
    vb = v.reshape(BH, n, block, D)
    m = mask.expand(B, h, n, n).reshape(BH, n, n)
    cnt = m.sum(-1)                                   # (BH, n), >= 1 (STA prior keeps the diagonal)
    K = int(cnt.max())
    inds = m.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[..., :K]   # (BH, n, K)
    valid = torch.arange(K, device=q.device) < cnt[..., None]                    # (BH, n, K)

    per_qblock = BH * K * block * D * q.element_size() * 2
    c = max(1, min(n, _ATTN_CHUNK_BYTES // max(1, per_qblock)))
    out = torch.empty_like(qb)
    rows = torch.arange(BH, device=q.device)[:, None]
    for i in range(0, n, c):
        j = min(n, i + c)
        idx = inds[:, i:j].reshape(BH, -1)                                   # (BH, cc*K)
        kg = kb[rows, idx].reshape(BH, j - i, K * block, D)
        vg = vb[rows, idx].reshape(BH, j - i, K * block, D)
        am = valid[:, i:j].repeat_interleave(block, dim=-1)[:, :, None, :]   # (BH, cc, 1, K*block)
        out[:, i:j] = F.scaled_dot_product_attention(qb[:, i:j], kg, vg, attn_mask=am)
        del kg, vg, am
    return out.reshape(B, h, S, D)


class _SparseSDPAProcessor:
    """Drop-in for Kandinsky6SRAttnProcessor: same NABLA mask, computed with gathered SDPA.

    Upstream dispatches to flex attention, which needs torch.compile (Inductor) to stay sparse; Inductor
    is unusable with the bundled triton-windows, and eager flex materialises the full score matrix
    (118 GiB per tile at 1440p)."""

    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb, sparse_params):
        import math
        import torch
        from diffusers.models.transformers.transformer_kandinsky6_sr import apply_rotary

        query = attn.to_query(hidden_states).unflatten(-1, (attn.num_heads, -1))
        key = attn.to_key(hidden_states).unflatten(-1, (attn.num_heads, -1))
        value = attn.to_value(hidden_states).unflatten(-1, (attn.num_heads, -1))
        query = attn.query_norm(query.float()).type_as(query)
        key = attn.key_norm(key.float()).type_as(key)
        query = apply_rotary(query, rotary_emb).transpose(1, 2)   # (B, h, S, D)
        key = apply_rotary(key, rotary_emb).transpose(1, 2)
        value = value.transpose(1, 2)

        # Same block mask as upstream nabla_block_mask (64-token blocks, CDF threshold + STA prior).
        B, h, S, D = query.shape
        s1 = S // 64
        qa = query.reshape(B, h, s1, 64, D).mean(-2)
        ka = key.reshape(B, h, s1, 64, D).mean(-2).transpose(-2, -1)
        attn_map = torch.softmax((qa @ ka) / math.sqrt(D), dim=-1)
        vals, inds = attn_map.sort(-1)
        mask = (vals.cumsum_(-1) >= 1 - sparse_params["threshold"]).int().gather(-1, inds.argsort(-1))
        mask = torch.logical_or(mask.bool(), sparse_params["sta_mask"].bool())
        del qa, ka, attn_map, vals, inds

        out = _block_sparse_attention(query, key, value, mask)
        return attn.out_layer(out.transpose(1, 2).flatten(2, 3))


def _install_sparse_attention(transformer):
    from diffusers.models.transformers.transformer_kandinsky6_sr import Kandinsky6SRAttention
    n = 0
    for mod in transformer.modules():
        if isinstance(mod, Kandinsky6SRAttention):
            mod.processor = _SparseSDPAProcessor()
            n += 1
    print(f"Kandinsky Video SR: block-sparse SDPA attention on {n} layers (no flex/compile)")


_DECODE_TILE = 48      # latent px per tile side (x16 = 768 px); covers a pipeline tile in one call (10 GiB peak with channels_last_3d)
_DECODE_OVERLAP = 8    # latent px overlap between tiles, linearly feathered


def _ramp(n, overlap, lead, trail, device):
    import torch
    w = torch.ones(n, device=device)
    o = min(overlap, n)
    if o > 0:
        r = (torch.arange(o, device=device, dtype=torch.float32) + 1) / (o + 1)
        if lead:
            w[:o] = r
        if trail:
            w[-o:] = torch.minimum(w[-o:], r.flip(0))
    return w


def _install_channels_last_convs(vae):
    """Run the KVAE Conv3d layers in channels_last_3d.

    On torch 2.9.1 / cuDNN 9.10 (Windows), bf16/fp16 Conv3d with contiguous NCDHW tensors never picks
    cuDNN for k>1 kernels: it falls back to aten::slow_conv_dilated3d (vol2col + ~170k tiny copy/fill
    launches per decode). In channels_last_3d the same conv dispatches to cuDNN: ~10x faster decode and
    about half the memory, output equal within bf16 rounding."""
    import torch
    import diffusers.models.autoencoders.autoencoder_kandinsky6_sr as kv

    cl = torch.channels_last_3d
    for mod in vae.modules():
        if isinstance(mod, torch.nn.Conv3d):
            mod.weight.data = mod.weight.data.contiguous(memory_format=cl)
    cls = kv.Kandinsky6SRSafeConv3d
    if not getattr(cls, "_pallaidium_cl3d", False):
        orig = cls.forward

        def forward(self, x):
            return orig(self, x.contiguous(memory_format=cl))

        cls.forward = forward
        cls._pallaidium_cl3d = True


def _install_tiled_decode(vae):
    """Wrap vae.decode with overlapping spatial tiles (full temporal pass per tile, so the causal cache
    stays intact). A full-frame decode at ~1440x2520 asks for a single 23 GiB conv allocation."""
    import torch
    orig = vae.decode
    f = vae.spatial_compression_ratio

    def decode(z, return_dict=True):
        _, _, _, H, W = z.shape
        if H <= _DECODE_TILE and W <= _DECODE_TILE:
            return orig(z, return_dict=return_dict)
        step = _DECODE_TILE - _DECODE_OVERLAP
        ys = list(range(0, max(1, H - _DECODE_OVERLAP), step))
        xs = list(range(0, max(1, W - _DECODE_OVERLAP), step))
        out = weight = None
        for y0 in ys:
            y1 = min(H, y0 + _DECODE_TILE)
            y0 = max(0, y1 - _DECODE_TILE)
            for x0 in xs:
                x1 = min(W, x0 + _DECODE_TILE)
                x0 = max(0, x1 - _DECODE_TILE)
                tile = orig(z[..., y0:y1, x0:x1], return_dict=False)[0].float()
                if out is None:
                    B, C, T = tile.shape[:3]
                    out = torch.zeros(B, C, T, H * f, W * f, device="cpu")
                    weight = torch.zeros(1, 1, 1, H * f, W * f, device="cpu")
                ov = _DECODE_OVERLAP * f
                wy = _ramp(tile.shape[-2], ov, y0 > 0, y1 < H, tile.device)
                wx = _ramp(tile.shape[-1], ov, x0 > 0, x1 < W, tile.device)
                w = (wy[:, None] * wx[None, :])[None, None, None]
                out[..., y0 * f:y1 * f, x0 * f:x1 * f] += (tile * w).cpu()
                weight[..., y0 * f:y1 * f, x0 * f:x1 * f] += w.cpu()
                del tile
                torch.cuda.empty_cache()
        res = (out / weight.clamp_min(1e-6)).to(device=z.device, dtype=z.dtype)
        if not return_dict:
            return (res,)
        from diffusers.models.autoencoders.vae import DecoderOutput
        return DecoderOutput(sample=res)

    vae.decode = decode


def _run_sr(pipe, frames, resolution_scale, lq_noise_scale, generator, num_steps=2,
            min_overlap=0.2, progress=None, cancelled=None, phase=None):
    """Kandinsky6SRPipeline.__call__ re-staged for a 24 GB card (same maths, same tiles, same noise order).

    Upstream denoises and decodes tile by tile under model CPU offload, so the 6.6 GB VAE and the transformer
    swap on and off the GPU for every tile, and the fp32 output canvas (5+ GB at 1440p x 121 frames) plus all
    low-res tiles sit on the GPU the whole run; together that needed ~34 GB. Here every model is moved to the
    GPU once for its stage and back, and all intermediates and the canvas stay on the CPU:
      encode (VAE) -> latent upscale (bank) -> denoise all tiles (transformer) -> decode all tiles (VAE).
    """
    import math
    import torch
    import torch.nn.functional as F
    from diffusers.pipelines.kandinsky6.pipeline_kandinsky6_sr import (
        retrieve_timesteps, _tile_positions, _hann_window_2d,
    )
    from diffusers.utils.torch_utils import randn_tensor

    cuda = torch.device("cuda")
    cpu = torch.device("cpu")

    def _check():
        if cancelled and cancelled():
            raise RuntimeError("Cancelled by user")

    def _stage(module):
        for m in (pipe.vae, pipe.transformer, pipe.latent_upscaler):
            if m is not None and m is not module:
                m.to(cpu)
        torch.cuda.empty_cache()
        if module is not None:
            module.to(cuda)

    dtype = pipe.transformer.dtype
    snap = pipe.vae_scale_factor_spatial

    video = pipe.video_processor.preprocess_video(frames)          # CPU, (1, 3, T, H, W) in [-1, 1]
    pipe.check_inputs(video, resolution_scale, num_steps, lq_noise_scale, min_overlap, 1)
    tiling_scale, pre_upscale = (2, 1.125) if resolution_scale == 2.25 else (int(resolution_scale), 1.0)
    if pre_upscale != 1.0:
        b, c, t, h, w = video.shape
        tgt = (round(h * pre_upscale / snap) * snap, round(w * pre_upscale / snap) * snap)
        fr = F.interpolate(video.permute(0, 2, 1, 3, 4).flatten(0, 1), size=tgt, mode="bilinear",
                           align_corners=False)
        video = fr.unflatten(0, (b, t)).permute(0, 2, 1, 3, 4)
    _, _, num_frames, height, width = video.shape

    base_h, base_w = min(pipe.transformer_tile_sizes, key=lambda hw: abs(hw[1] / hw[0] - width / height))
    tile_h, tile_w = base_h // tiling_scale, base_w // tiling_scale
    if height < tile_h or width < tile_w:
        raise ValueError(f"Input must be at least {tile_w}x{tile_h} px for x{resolution_scale:g}.")
    grid = [(t, l) for t in _tile_positions(height, tile_h, min_overlap, snap)
            for l in _tile_positions(width, tile_w, min_overlap, snap)]
    n_tiles = len(grid)
    total = 1 + n_tiles * (num_steps + 1)
    done = 0

    def _tick(k=1):
        nonlocal done
        done += k
        if progress:
            progress(min(done, total), total)

    with torch.inference_mode():
        # 1. Encode the whole clip (VAE only).
        if phase: phase("Encoding")
        _stage(pipe.vae)
        latents = pipe.encode_video(video.to(cuda)).to(cpu)
        del video
        _tick()
        _check()

        # 2. Low-res latent tiles at the trained tile resolution.
        use_lu = pipe.latent_upscaler is not None and tiling_scale in pipe.latent_upscaler.config.scales
        if not use_lu:
            raise RuntimeError("Kandinsky Video SR: the latent upscaler is missing from the bundle.")
        _stage(pipe.latent_upscaler)
        lq_tiles = []
        for top, left in grid:
            lt = latents[:, :, :, top // snap:(top + tile_h) // snap, left // snap:(left + tile_w) // snap]
            lq_tiles.append(pipe.latent_upscaler(lt.to(cuda), scale=tiling_scale, return_dict=False)[0]
                            .to(dtype).to(cpu))
        del latents
        _check()

        # 3. Denoise every tile (transformer only); same RNG order as upstream (one noise draw per tile).
        if phase: phase(f"Denoising {n_tiles} tiles")
        _stage(pipe.transformer)
        refined = []
        for i, lq in enumerate(lq_tiles):
            _check()
            lq = lq.to(cuda).permute(0, 2, 3, 4, 1)                  # (1, T, H, W, C)
            noise = randn_tensor(lq.shape, generator=generator, device=cuda, dtype=dtype)
            x = math.sqrt(1 - lq_noise_scale ** 2) * lq + lq_noise_scale * noise
            timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, cuda)
            for t in timesteps:
                anchor = torch.zeros_like(x)
                anchor_mask = torch.zeros((*x.shape[:-1], 1), dtype=dtype, device=cuda)
                pred = pipe.transformer(hidden_states=torch.cat([x, anchor, anchor_mask], dim=-1),
                                        timestep=t.expand(x.shape[0]), return_dict=False)[0]
                x = pipe.scheduler.step(pred, t, x, return_dict=False)[0]
                _tick()
            refined.append(x.permute(0, 4, 1, 2, 3).to(cpu))
            del lq, noise, x, pred
        del lq_tiles

        # 4. Decode every tile (VAE only) and blend on the CPU with upstream's Hann windows.
        if phase: phase(f"Decoding {n_tiles} tiles")
        _stage(pipe.vae)
        out_h, out_w = height * tiling_scale, width * tiling_scale
        # System RAM matters here: the weights of all three models already sit on the CPU (~16.6 GB), so the
        # canvas is fp16 (values in [-1, 1] times a window sum of a few, well inside fp16 range/precision;
        # half of the 5+ GB an fp32 canvas takes at 1440p x 121 frames) and is turned into frames one at a
        # time instead of through full-video float copies.
        from PIL import Image

        acc = torch.zeros((3, num_frames, out_h, out_w), dtype=torch.float16)
        # Upstream blends with Hann windows whose border weights are ~1e-6; tile*window at that scale underflows
        # in fp16. The window sums depend only on the grid, so they are built first and every tile is added
        # already normalised (tile * window / sum): each contribution is a fraction of the final value in
        # [-1, 1], fp16-safe, and the weights are exactly upstream's.
        window = _hann_window_2d(base_h, base_w, cpu)
        wacc = torch.zeros((out_h, out_w), dtype=torch.float32)
        for top, left in grid:
            y, x0 = top * tiling_scale, left * tiling_scale
            wacc[y:y + base_h, x0:x0 + base_w] += window
        for (top, left), lat in zip(grid, refined):
            _check()
            tile = pipe.decode_latents(lat.to(cuda))[0]               # (3, T, base_h, base_w) in [-1, 1]
            y, x0 = top * tiling_scale, left * tiling_scale
            w = (window / wacc[y:y + base_h, x0:x0 + base_w]).to(tile.device)
            acc[:, :, y:y + base_h, x0:x0 + base_w] += (tile * w).to(torch.float16).cpu()
            del tile, w
            _tick()
        _stage(None)
        del refined, wacc
        frames_out = []
        for ti in range(num_frames):
            f = acc[:, ti].float().clamp_(-1, 1).add_(1).mul_(127.5).round_().to(torch.uint8)
            frames_out.append(Image.fromarray(f.permute(1, 2, 0).numpy()))
        del acc
        return frames_out


def _pick_scale(ratio):
    if ratio <= 2.0:
        return 2.0
    if ratio <= 2.25:
        return 2.25
    return 4.0


class KandinskyVideoSRPlugin(ModelPlugin):
    MODEL_ID     = _REPO
    DISPLAY_NAME = "Kandinsky Super Resolution"
    MODEL_TYPE   = "video"
    DESCRIPTION  = (
        "Kandinsky 6.0 Video SR 2.0: 2-step diffusion video upscaler (x2 / x2.25 / x4, tiled, up to "
        "15 s). Set the target size with Resolution"
    )

    INPUTS      = InputSpec.VIDEO
    UI_SECTIONS = [UISection.VIDEO_STRIP, UISection.RESOLUTION, UISection.IMAGE_STRENGTH, UISection.SEED]
    PARAMS      = ParamSpec(width=1920, height=1080, steps=2, guidance=1.0, strength=0.7)

    REQUIRED_PACKAGES         = ["torch", "diffusers", "PIL"]
    supports_inpaint          = False
    supports_img2img          = False
    requires_input_strip      = True
    uses_standard_input_strip = False
    show_enhance              = False
    supports_batch            = False

    def is_available(self):
        try:
            from diffusers import Kandinsky6SRPipeline  # noqa: F401
        except ImportError:
            return False
        return True

    def on_model_selected(self, scene, context):
        # Strength (image_power) is shared across plugins and not reset by the generic
        # model-change code, so apply this model's default here.
        scene.image_power = self.PARAMS.strength

    def draw_post_seed_ui(self, col, context):
        col.prop(context.scene, "kvsr_scale")

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Kandinsky6SRPipeline

        hub = dict(cache_dir=prefs.hf_cache_dir or None, dtype=torch.bfloat16)
        print(f"Kandinsky Video SR: loading ({_REPO})…")
        try:
            pipe = Kandinsky6SRPipeline.from_pretrained(_REPO, local_files_only=True, **hub)
        except (OSError, ValueError):
            if prefs.local_files_only:
                raise
            pipe = Kandinsky6SRPipeline.from_pretrained(_REPO, local_files_only=False, **hub)
        # 16.6 GB of weights: never resident all at once on a 24 GB card.
        _install_sparse_attention(pipe.transformer)
        # KVAE encodes/decodes in causal temporal segments (16 frames by default); one 16-frame segment at
        # ~720x1260 asks for a 22 GiB conv allocation. The causal convs carry their padding state across
        # segments, so a shorter segment gives the identical result with ~4x less activation memory.
        # Must stay a multiple of the temporal compression (4).
        import diffusers.models.autoencoders.autoencoder_kandinsky6_sr as _kvae
        _kvae.SEGMENT_FRAMES = 4
        _install_channels_last_convs(pipe.vae)
        _install_tiled_decode(pipe.vae)
        # No accelerate offload hooks: _run_sr moves each model to the GPU once for its own stage.
        return pipe

    def generate(self, pipe, inputs: ModelInputs, scene, prefs):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from diffusers.utils import export_to_video

        vid_path = inputs.video_path
        if not vid_path:
            raise ValueError("Kandinsky Video SR needs an input video strip.")

        self.set_phase(inputs, "Reading video")
        frames = load_video_as_np_array(vid_path)
        n = min(len(frames), _MAX_FRAMES)
        n = ((n - 1) // 4) * 4 + 1 if n >= 1 else 0
        if n < 5:
            raise ValueError("Kandinsky Video SR needs at least 5 input frames.")
        frames = [Image.fromarray(np.asarray(f)).convert("RGB") for f in frames[:n]]

        cap = cv2.VideoCapture(vid_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or inputs.fps
        cap.release()

        src_w, src_h = frames[0].size
        mode = str(getattr(scene, "kvsr_scale", "AUTO"))
        target = None
        if mode in ("2", "2.25", "4"):
            scale = float(mode)
        else:
            tw = int(inputs.width) or src_w * 2
            th = int(inputs.height) or src_h * 2
            # Fit inside the target, keeping the source aspect ratio (even dims for H.264), and pick
            # the factor from that same fit ratio, so no larger factor is rendered only to be downscaled.
            fit = min(tw / src_w, th / src_h)
            scale = _pick_scale(fit - 1e-6)
            target = (max(2, int(round(src_w * fit / 2)) * 2), max(2, int(round(src_h * fit / 2)) * 2))
        noise = max(0.0, min(1.0, float(inputs.strength)))

        self.set_phase(inputs, f"Upscaling x{scale:g} ({n} frames)")
        print(f"Kandinsky Video SR: {src_w}x{src_h}x{n} -> x{scale:g} ({mode}), "
              f"lq_noise {noise:.2f}, target {target or 'native'}")
        generator = torch.Generator("cpu").manual_seed(int(inputs.seed))
        noise = max(0.01, noise)  # pipeline requires lq_noise_scale in (0, 1]
        out = _run_sr(
            pipe, frames, scale, noise, generator,
            num_steps=2,  # distilled checkpoint; pipeline default (4) is wrong here
            progress=inputs.progress_fn,
            cancelled=inputs.should_cancel,
            phase=lambda p: self.set_phase(inputs, p),
        )

        if target and out and out[0].size != target:
            out = [f.resize(target, Image.LANCZOS) for f in out]

        self.set_phase(inputs, "Saving")
        dst_path = solve_path(clean_filename(str(inputs.seed) + "_kandinsky_vsr") + ".mp4")
        tmp_path = export_to_video(out, fps=src_fps)
        shutil.move(tmp_path, dst_path)

        if _source_has_audio(vid_path):
            self.set_phase(inputs, "Muxing audio")
            muxed_path = dst_path.replace(".mp4", "_muxed.mp4")
            if _mux_audio(dst_path, vid_path, muxed_path):
                os.replace(muxed_path, dst_path)
        return dst_path
