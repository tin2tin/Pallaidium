"""Voice cloning TTS via Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-Base).

Runtime compatibility shims for Blender 5.2's newer transformers build.
All patches are applied in load() before any qwen_tts / faster_qwen3_tts
module is imported, so a clean pip-installed package just works.
"""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename


# ---------------------------------------------------------------------------
# Patch 1 – check_model_inputs decorator factory
# ---------------------------------------------------------------------------
def _patch_check_model_inputs():
    """Make check_model_inputs work as @decorator() factory.

    Newer transformers defines check_model_inputs(func) directly (no factory),
    but qwen_tts calls it as @check_model_inputs() with no arguments.
    """
    import sys, importlib, functools

    for key in list(sys.modules.keys()):
        if "qwen_tts" in key or "faster_qwen3_tts" in key:
            del sys.modules[key]

    for mod_name in (
        "transformers.utils.generic",
        "transformers.utils.doc",
        "transformers.utils",
        "transformers",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        fn = getattr(mod, "check_model_inputs", None)
        if fn is None or getattr(fn, "_pallaidium_patched", False):
            continue
        _orig = fn
        def _compat(func=None, _o=_orig):
            if func is None:
                return lambda f: f
            return _o(func)
        _compat._pallaidium_patched = True
        try:
            functools.update_wrapper(_compat, _orig)
        except Exception:
            pass
        setattr(mod, "check_model_inputs", _compat)
        return


# ---------------------------------------------------------------------------
# Patch 2 – ROPE_INIT_FUNCTIONS missing 'default' key
# ---------------------------------------------------------------------------
def _patch_rope_init_functions():
    """Add missing 'default' entry to ROPE_INIT_FUNCTIONS for newer transformers."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:
        return
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    import torch

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


# ---------------------------------------------------------------------------
# Patch 3 – qwen_tts create_causal_mask call-site compatibility
#   Old API: create_causal_mask(input_embeds=…, cache_position=…, …)
#   New API: create_causal_mask(inputs_embeds=…, …)   (cache_position removed)
# ---------------------------------------------------------------------------
def _patch_qwen_tts_mask_kwargs():
    """Fix mask_kwargs in modeling_qwen3_tts.py and modeling_qwen3_tts_tokenizer_v2.py.

    Old API: create_causal_mask(input_embeds=…, cache_position=…, …)
    New API: create_causal_mask(inputs_embeds=…, …)   (cache_position removed)
    """
    import sys, importlib

    try:
        from transformers.masking_utils import (
            create_causal_mask as _new_ccm,
            create_sliding_window_causal_mask as _new_scm,
        )
    except ImportError:
        return

    def _make_compat(fn):
        def _compat_mask(**kw):
            if "input_embeds" in kw and "inputs_embeds" not in kw:
                kw["inputs_embeds"] = kw.pop("input_embeds")
            kw.pop("cache_position", None)
            return fn(**kw)
        _compat_mask._pallaidium_compat = True
        return _compat_mask

    for mod_name in (
        "qwen_tts.core.models.modeling_qwen3_tts",
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue

        for fn_name, new_fn in [
            ("create_causal_mask", _new_ccm),
            ("create_sliding_window_causal_mask", _new_scm),
        ]:
            if not hasattr(mod, fn_name):
                continue
            if getattr(getattr(mod, fn_name), "_pallaidium_compat", False):
                continue
            setattr(mod, fn_name, _make_compat(new_fn))


# ---------------------------------------------------------------------------
# Patch 4 – faster_qwen3_tts graph files: StaticCache / DynamicCache API
# ---------------------------------------------------------------------------
def _patch_faster_qwen3_tts():
    """
    Fix faster_qwen3_tts for newer transformers StaticCache / DynamicCache API:

    1. StaticLayer.lazy_initialization now requires (key_states, value_states) —
       both predictor_graph and talker_graph called it with one arg.

    2. create_causal_mask no longer accepts 'cache_position'; position is inferred
       from cache.get_seq_length() (i.e. cumulative_length). We must temporarily
       set cumulative_length = i before building each mask, then restore it.

    3. 'input_embeds' renamed to 'inputs_embeds' in create_causal_mask.

    4. DynamicCache is no longer subscriptable; use .layers[li].keys / .values.
    """
    import sys, torch
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    # ---- predictor_graph --------------------------------------------------
    pg_mod = sys.modules.get("faster_qwen3_tts.predictor_graph")
    if pg_mod is not None:
        PG = pg_mod.PredictorGraph

        def _pg_set_cache_pos(self, pos):
            for layer in self.static_cache.layers:
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length.fill_(pos)

        def _pg_init_cache_layers(self):
            config = self.pred_model.config
            nkv = getattr(config, "num_key_value_heads", config.num_attention_heads)
            hd  = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            dummy = torch.zeros(1, nkv, 1, hd, dtype=self.dtype, device=self.device)
            for layer in self.static_cache.layers:
                if not layer.is_initialized:
                    layer.lazy_initialization(dummy, dummy)  # two args

        def _pg_make_attn_mask(self, input_embeds, cache_position):
            # Set cumulative_length so the new create_causal_mask infers correct offset
            self._set_cache_position(int(cache_position[0].item()))
            mask_fn = (create_sliding_window_causal_mask
                       if self.has_sliding_layers else create_causal_mask)
            mask = mask_fn(
                config=self.pred_model.config,
                inputs_embeds=input_embeds,
                attention_mask=None,
                past_key_values=self.static_cache,
            )
            if self.has_sliding_layers:
                sliding = create_sliding_window_causal_mask(
                    config=self.pred_model.config,
                    inputs_embeds=input_embeds,
                    attention_mask=None,
                    past_key_values=self.static_cache,
                )
                return {"full_attention": mask, "sliding_attention": sliding}
            return {"full_attention": mask}

        PG._set_cache_position  = _pg_set_cache_pos
        PG._init_cache_layers   = _pg_init_cache_layers
        PG._make_attn_mask      = _pg_make_attn_mask

    # ---- talker_graph -----------------------------------------------------
    tg_mod = sys.modules.get("faster_qwen3_tts.talker_graph")
    if tg_mod is not None:
        TG = tg_mod.TalkerGraph

        def _tg_set_cache_pos(self, pos):
            for layer in self.static_cache.layers:
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length.fill_(pos)

        def _tg_init_cache_layers(self):
            config = self.model.config
            nkv = getattr(config, "num_key_value_heads", config.num_attention_heads)
            hd  = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            dummy = torch.zeros(1, nkv, 1, hd, dtype=self.dtype, device=self.device)
            for layer in self.static_cache.layers:
                if not layer.is_initialized:
                    layer.lazy_initialization(dummy, dummy)  # two args

        def _tg_build_attention_masks(self, attention_mask=None):
            dummy = torch.zeros(1, 1, self.hidden_size, dtype=self.dtype, device=self.device)
            max_len = self.max_seq_len
            self.attn_mask_table = [None] * max_len
            mask_fn = (create_sliding_window_causal_mask
                       if self.model.config.sliding_window is not None
                       else create_causal_mask)

            # Save cumulative_lengths so prefill_kv state is not corrupted
            attn_layers = [l for l in self.static_cache.layers
                           if hasattr(l, "cumulative_length")]
            saved = [int(l.cumulative_length.item()) for l in attn_layers]

            for i in range(max_len):
                self._set_cache_position(i)
                self.attn_mask_table[i] = mask_fn(
                    config=self.model.config,
                    inputs_embeds=dummy,
                    attention_mask=attention_mask,
                    past_key_values=self.static_cache,
                )

            # Restore cumulative_lengths
            for layer, s in zip(attn_layers, saved):
                layer.cumulative_length.fill_(s)

            if self.attn_mask is None:
                self.attn_mask = self.attn_mask_table[0].clone()
            else:
                self.attn_mask.copy_(self.attn_mask_table[0])

        def _tg_prefill_kv(self, past_key_values):
            self.static_cache.reset()
            seq_len = 0
            for li in range(self.num_layers):
                # New API: DynamicCache is not subscriptable; use .layers[li]
                cache_layer = past_key_values.layers[li]
                k, v = cache_layer.keys, cache_layer.values
                seq_len = k.shape[2]
                if seq_len > self.max_seq_len:
                    raise RuntimeError(
                        f"Input too long: prefill={seq_len} > max_seq_len={self.max_seq_len}."
                    )
                cache_pos = torch.arange(seq_len, device=self.device)
                self.static_cache.update(k, v, li, {"cache_position": cache_pos})
            return seq_len

        TG._set_cache_position    = _tg_set_cache_pos
        TG._init_cache_layers     = _tg_init_cache_layers
        TG._build_attention_masks = _tg_build_attention_masks
        TG.prefill_kv             = _tg_prefill_kv

    # ---- fast_generate: force parity_mode to bypass CUDA-graph complexity ---
    gen_mod = sys.modules.get("faster_qwen3_tts.generate")
    if gen_mod is not None and hasattr(gen_mod, "fast_generate"):
        _orig_fast_generate = gen_mod.fast_generate
        if not getattr(_orig_fast_generate, "_pallaidium_patched", False):
            def _parity_fast_generate(*args, **kwargs):
                kwargs["parity_mode"] = True
                return _orig_fast_generate(*args, **kwargs)
            _parity_fast_generate._pallaidium_patched = True
            gen_mod.fast_generate = _parity_fast_generate
            # Also patch inside faster_qwen3_tts.model which imports it directly
            model_mod = sys.modules.get("faster_qwen3_tts.model")
            if model_mod is not None and hasattr(model_mod, "fast_generate"):
                model_mod.fast_generate = _parity_fast_generate


class Qwen3TTSPlugin(ModelPlugin):
    MODEL_ID     = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    DISPLAY_NAME = "TTS: Qwen3 (voice clone)"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Voice cloning TTS — requires speaker audio + reference text"

    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF_REQ | InputSpec.TEXT_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.AUDIO_DURATION,
        UISection.AUDIO_REF,
        UISection.TEXT_REF,
        UISection.SEED,
    ]
    PARAMS       = ParamSpec(audio_ref_required=True)
    REQUIRED_PACKAGES = ["torch", "soundfile", "faster_qwen3_tts"]

    def load(self, prefs, scene, **kw):
        import torch
        # 1) Drop any cached module state and fix check_model_inputs
        _patch_check_model_inputs()
        # 2) Patch ROPE before the first import of transformers rope utils
        _patch_rope_init_functions()

        from faster_qwen3_tts import FasterQwen3TTS

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading Qwen3-TTS on {device}…")
        model = FasterQwen3TTS.from_pretrained(
            self.MODEL_ID,
            dtype=torch.bfloat16,
        )

        # 3) from_pretrained imported predictor_graph, talker_graph, and qwen_tts —
        #    now the submodules exist in sys.modules so the patches actually land.
        import importlib, sys
        # Force-import generate so we can patch fast_generate too
        if "faster_qwen3_tts.generate" not in sys.modules:
            importlib.import_module("faster_qwen3_tts.generate")

        _patch_qwen_tts_mask_kwargs()
        _patch_faster_qwen3_tts()

        return {"pipe": None, "model": model, "vocoder": None, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        import soundfile as sf
        import random

        model = pipe_obj["model"]
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        seed = inputs.seed
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        if not inputs.audio_ref:
            raise ValueError("Qwen3-TTS requires a speaker reference audio file.")
        if not inputs.text_ref:
            raise ValueError("Qwen3-TTS requires a reference transcription text file.")

        output_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")

        self.set_phase(inputs, "Generating")
        print(f"Qwen3-TTS generating…  ref_audio={inputs.audio_ref}")
        wavs, sr = model.generate_voice_clone(
            text=inputs.prompt,
            language="English",
            ref_audio=inputs.audio_ref,
            ref_text=inputs.text_ref,
        )
        if not wavs:
            raise RuntimeError("Qwen3-TTS: generation returned no audio.")

        sf.write(output_path, wavs[0], sr)
        print("Qwen3-TTS audio saved:", output_path)
        return output_path
