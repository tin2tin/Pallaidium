"""Text-to-song via YuE2-3B (OzzyGT/YuE2-3B-Diffusers, Modular Diffusers blocks).

Full songs with vocals from a style description + section-tagged lyrics.
Plans an ABC score first, then generates codec tokens (25/s), synthesizes
acoustic latents and decodes 48 kHz stereo. Song length follows the lyrics
(capped by the model at 9000 tokens ≈ 6 min). Weights are CC BY-NC 4.0.
"""

import sys

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename


# ---------------------------------------------------------------------------
# CUDA-graph decoding without FlashAttention
#
# YuE2's CUDA-graph token decoder (cuda_graph.GraphAR, in the remote blocks
# repo) calls torch.ops.aten._flash_attention_forward directly. Windows torch
# wheels are built without it ("USE_FLASH_ATTENTION was not enabled for
# build"), leaving only eager decoding — 12-25 tok/s, i.e. 6-12 min of token
# generation for a 6-min song. The masked attention below covers the one call
# shape GraphAR uses and is graph-capturable: measured 90-100 tok/s on an
# RTX 4090 at equal VRAM, matching fp32 SDPA to ~1e-3.
# ---------------------------------------------------------------------------

def _masked_varlen_attention(query, key, value, cu_seqlens_q, cu_seqlens_k, max_q, max_k,
                             dropout_p, is_causal, return_debug_mask, seqused_k=None, **_):
    """Stand-in for aten._flash_attention_forward as GraphAR calls it: one query
    token per branch; each branch's keys packed into a fixed `max_k`-slot block,
    of which the first `seqused_k` are valid. Only output [0] is used."""
    import torch
    branches = cu_seqlens_q.numel() - 1
    heads, head_dim = query.shape[1], query.shape[2]
    kv_heads = key.shape[1]
    q = query.view(branches, kv_heads, heads // kv_heads, head_dim)   # GQA: head h -> kv head h // group
    k = key.view(branches, max_k, kv_heads, head_dim).transpose(1, 2)   # B, KVH, L, D
    v = value.view(branches, max_k, kv_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * (head_dim ** -0.5)
    slots = torch.arange(max_k, device=query.device)
    hidden = slots[None, :] >= seqused_k[:, None].to(slots.dtype)         # B, L
    scores = scores.masked_fill(hidden[:, None, None, :], float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1).to(v.dtype), v)      # B, KVH, G, D
    return (out.reshape(branches, heads, head_dim),)


class _Delegate:
    """Forwards every attribute to `_real` except those set on the instance."""
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FlashOp(_Delegate):
    def __call__(self, *args, **kwargs):
        return _masked_varlen_attention(*args, **kwargs)


def _patch_gqa_attention(transformer):
    """Expand grouped K/V heads before attention instead of passing enable_gqa.

    Without FlashAttention, SDPA with enable_gqa=True can only use the math
    backend, which materialises the full attention matrix. Synthesis attends a
    whole song at once (one chunk up to the 24k context), so a 6-min song needs
    20+ GB for that matrix alone — measured 17.7 GB extra for one 8k×16k call
    vs 0.03 GB with repeated K/V (memory-efficient kernel). Same fix the
    transformer already applies on MPS. Only YuE2's transformer module is patched."""
    mod = sys.modules[type(transformer).__module__]
    original = mod.dispatch_attention_fn
    if getattr(original, "_yue2_patched", False):
        return

    def dispatch(query, key, value, *args, enable_gqa=False, **kwargs):
        if enable_gqa:
            groups = query.shape[2] // key.shape[2]   # (batch, seq, heads, dim)
            key = key.repeat_interleave(groups, dim=2)
            value = value.repeat_interleave(groups, dim=2)
        return original(query, key, value, *args, enable_gqa=False, **kwargs)

    dispatch._yue2_patched = True
    mod.dispatch_attention_fn = dispatch


def _patch_graph_attention(blocks):
    """Give the remote cuda_graph module a `torch` whose
    ops.aten._flash_attention_forward is _masked_varlen_attention. Only that
    module sees the proxy; the global torch (and every other plugin) is untouched."""
    import torch
    blocks_mod = sys.modules[type(blocks).__module__]
    graph_mod = sys.modules[blocks_mod.GraphAR.__module__]
    if getattr(graph_mod.torch, "_yue2_patched", False):
        return
    aten = _Delegate(torch.ops.aten)
    aten._flash_attention_forward = _FlashOp(torch.ops.aten._flash_attention_forward)
    ops = _Delegate(torch.ops)
    ops.aten = aten
    proxy = _Delegate(torch)
    proxy.ops = ops
    proxy._yue2_patched = True
    graph_mod.torch = proxy


class YuE2Plugin(ModelPlugin):
    MODEL_ID     = "OzzyGT/YuE2-3B-Diffusers"
    BLOCKS_ID    = "OzzyGT/YuE2-Modular"
    DISPLAY_NAME = "Music: YuE2 (Songs)"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = (
        "Full songs with vocals from style tags + lyrics ([Verse], [Chorus] …). "
        "Prompt = style (genre, voice, instruments). Length follows the lyrics. Non-commercial weights"
    )

    INPUTS       = InputSpec.PROMPT | InputSpec.MUSIC_PARAMS
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.MUSIC_PARAMS,
        UISection.STEPS,
        UISection.SEED,
    ]
    # steps = midpoint ODE steps per acoustic chunk (model default 32)
    PARAMS            = ParamSpec(steps=32)
    REQUIRED_PACKAGES = ["torch", "diffusers", "tiktoken", "soundfile"]
    supports_inpaint  = False
    supports_img2img  = False
    show_enhance      = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import ModularPipelineBlocks

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        print(f"Loading {self.BLOCKS_ID} + {self.MODEL_ID}…")
        # Load the blocks first (avoids the missing-components warnings).
        blocks = ModularPipelineBlocks.from_pretrained(
            self.BLOCKS_ID,
            trust_remote_code=True,
            components_repo=self.MODEL_ID,
            trust_components_code=True,
            cache_dir=_cache_dir,
            local_files_only=_lfo,
        )
        pipe = blocks.init_pipeline()
        # The VAE must stay FP32 — the release decodes in FP32 with TF32 off.
        pipe.load_components(
            dtype={"transformer": torch.bfloat16, "vae": torch.float32},
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        # load_components() only warns on a per-component failure — fail loudly.
        missing = [n for n in ("transformer", "vae", "tokenizer") if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(
                f"YuE2: component(s) failed to load: {missing}. See the console above for the underlying error."
            )

        # ~7 GB transformer (bf16) + VAE: no offload path — autoregressive
        # decoding with CUDA graphs needs everything resident on one device.
        pipe.to(gfx_device)
        if gfx_device == "cuda" and not torch.backends.cuda.is_flash_attention_available():
            _patch_graph_attention(blocks)
            _patch_gqa_attention(pipe.transformer)
            print("YuE2: torch has no FlashAttention — CUDA-graph decoding uses masked attention, "
                  "and grouped K/V heads are expanded so synthesis uses memory-efficient attention.")
        return {"pipe": pipe}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        pipe = pipe_obj["pipe"]

        style = (inputs.prompt or "").strip()
        if inputs.bpm and "bpm" not in style.lower():
            style = f"{style}, {inputs.bpm} BPM" if style else f"{inputs.bpm} BPM"
        if inputs.key_scale and inputs.key_scale.lower() not in style.lower():
            style = f"{style}, {inputs.key_scale}" if style else inputs.key_scale
        ts = (inputs.time_signature or "").strip()
        if ts:
            ts = ts if "/" in ts else f"{ts}/4"
            if ts not in style:
                style = f"{style}, {ts} time" if style else f"{ts} time"
        lyrics = (inputs.lyrics or "").strip()
        if not style:
            raise ValueError("YuE2: enter a style prompt (genre, voice, instruments, tempo).")
        if not lyrics:
            print("YuE2: no lyrics given — the song will be generated from the style tags alone.")

        seed = abs(int(inputs.seed)) % (2 ** 63)

        cot = getattr(scene, "yue2_cot", "full") or "full"
        if cot not in ("full", "melody", "off"):
            cot = "full"

        # Supplied ABC score (re-take): only valid when score planning is on.
        abc = None
        abc_path = (getattr(scene, "yue2_abc_path", "") or "").strip()
        if abc_path and cot != "off":
            import bpy
            import os
            abc_file = bpy.path.abspath(abc_path)
            if not os.path.isfile(abc_file):
                raise FileNotFoundError(f"YuE2: Re-take Score file not found: {abc_file}")
            with open(abc_file, encoding="utf-8") as f:
                abc = f.read().strip() or None
            if abc:
                print(f"YuE2: using supplied ABC score {abc_file}")
        elif abc_path:
            print("YuE2: Re-take Score ignored because Score Mode is Off.")

        # Song-token sampling. max_tokens = seconds × 25 (codec rate); the
        # model's own cap is 9000 tokens and min_tokens (200) must fit below it.
        max_len = int(getattr(scene, "yue2_max_length", 0) or 0)
        semantic_sampling = {
            "temperature":        float(getattr(scene, "yue2_temperature",        1.0)),
            "top_p":              float(getattr(scene, "yue2_top_p",              0.95)),
            "top_k":              int(getattr(scene, "yue2_top_k",              100)),
            "repetition_penalty": float(getattr(scene, "yue2_repetition_penalty", 1.2)),
        }
        if max_len > 0:
            semantic_sampling["max_tokens"] = min(9000, max(200, max_len * 25))

        # Guidance lives on the guider component's config; reset it every run
        # so a previous job's value never leaks into this one (0 → model default).
        guidance = float(getattr(scene, "yue2_guidance", 0.0) or 0.0)
        pipe.guider.register_to_config(scale=guidance if guidance > 0 else None)

        # Progress: the two autoregressive stages report one token at a time
        # with no known total, so surface them as phase labels; synthesis and
        # decode report (completed, total) and drive the progress bar.
        counts = {}

        def on_token(stage, _token):
            n = counts.get(stage, 0) + 1
            counts[stage] = n
            if stage == "abc":
                if n == 1 or n % 100 == 0:
                    self.set_phase(inputs, f"Planning score ({n} tokens)")
            elif n == 1 or n % 25 == 0:  # semantic: 25 tokens = 1 s of audio
                self.set_phase(inputs, f"Composing ({n // 25}s of audio)")

        last_stage = [None]

        def on_progress(stage, completed, total):
            if stage != last_stage[0]:
                last_stage[0] = stage
                self.set_phase(inputs, "Synthesizing" if stage == "synthesis" else "Decoding")
            if inputs.progress_fn is not None:
                inputs.progress_fn(completed, max(1, total))

        kwargs = dict(
            style=style,
            lyrics=lyrics,
            cot=cot,
            abc=abc,
            seed=seed,
            semantic_sampling=semantic_sampling,
            ode_steps=max(1, int(inputs.steps)),
            vae_tile_frames=max(1, int(getattr(scene, "yue2_vae_tile_frames", 1024))),
            vae_tile_overlap_frames=max(0, int(getattr(scene, "yue2_vae_tile_overlap", 16))),
            on_token=on_token,
            on_progress=on_progress,
        )
        if inputs.should_cancel is not None:
            kwargs["cancelled"] = inputs.should_cancel

        print(
            f"YuE2: style='{style}' cot={cot} seed={seed} ode_steps={kwargs['ode_steps']} "
            f"sampling={semantic_sampling} guidance={guidance or 'default'}"
        )
        self.set_phase(inputs, "Planning score")
        # CUDA graphs: ~4-7x faster token decoding. Without FlashAttention,
        # load() patched the decoder's attention (see _patch_graph_attention).
        use_graph = gfx_device == "cuda"
        state, graph_error = None, None
        try:
            state = pipe(**kwargs, use_cuda_graph=use_graph)
        except InterruptedError:
            raise
        except RuntimeError as e:
            if not use_graph:
                raise
            graph_error = str(e)
        if state is None:
            # Retry outside the except block: the failed attempt's traceback
            # (and the GPU buffers its frames reference) is released first.
            import gc
            import torch
            print(f"YuE2: CUDA-graph decoding failed ({graph_error}); retrying without CUDA graphs…")
            counts.clear()
            last_stage[0] = None
            gc.collect()
            torch.cuda.empty_cache()
            state = pipe(**kwargs, use_cuda_graph=False)

        if state.get("abc_truncated") or state.get("semantic_truncated"):
            print("YuE2: generation hit the token limit — the song may end abruptly (shorten the lyrics).")

        self.set_phase(inputs, "Saving")
        import soundfile as sf
        audio = state.get("audios")[0].T.float().numpy()
        sample_rate = state.get("sample_rate")
        filename = solve_path(clean_filename(str(seed) + "_" + style[:30]) + ".wav")
        sf.write(filename, audio, sample_rate)
        print(f"YuE2 song saved: {filename} ({audio.shape[0] / sample_rate:.1f}s @ {sample_rate} Hz)")

        # Keep the ABC score beside the song so it can be fed back via Re-take Score.
        score = state.get("score")
        if score:
            import os
            abc_out = os.path.splitext(filename)[0] + ".abc"
            try:
                with open(abc_out, "w", encoding="utf-8") as f:
                    f.write(score)
                print(f"YuE2 score saved: {abc_out}")
            except OSError as e:
                print(f"YuE2: could not save the ABC score ({e})")
        return filename

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "yue2_cot")
        row = col.row(align=True)
        row.enabled = scene.yue2_cot != "off"
        row.prop(scene, "yue2_abc_path")
        col.prop(scene, "yue2_max_length")
        col.prop(scene, "yue2_temperature")
        col.prop(scene, "yue2_show_advanced")
        if scene.yue2_show_advanced:
            sub = col.column(align=True)
            sub.prop(scene, "yue2_top_p")
            sub.prop(scene, "yue2_top_k")
            sub.prop(scene, "yue2_repetition_penalty")
            col.prop(scene, "yue2_guidance")
            sub = col.column(align=True)
            sub.prop(scene, "yue2_vae_tile_frames")
            sub.prop(scene, "yue2_vae_tile_overlap")
        return False
