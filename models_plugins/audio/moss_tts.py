"""Expressive multilingual TTS via the MOSS-TTS family (OpenMOSS-Team/MOSS-TTS).

Zero-shot voice cloning, 31 languages, token-level duration control and inline
[pause Ns] markers.  Several model variants are selectable in the UI:

  * MOSS-TTS-v1.5      (8B)   flagship: cloning + multilingual + duration control
  * MOSS-TTS-Nano      (0.1B) lightweight, CPU-friendly, 48 kHz
  * MOSS-VoiceGenerator(1.7B) text-prompt voice design, no reference audio
  * MOSS-TTSD-v1.0     (8B)   multi-speaker dialogue

The model is loaded through transformers ``trust_remote_code=True`` — it runs on
the already-installed transformers/torch/soundfile stack and does NOT need the
MOSS pip package, so nothing is added to requirements.

Inline pauses can be embedded in the prompt, e.g.  "Hello[pause 1.5s]world".
"""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename

# UI variant key -> HuggingFace repo id
_VARIANT_REPOS = {
    "v1.5":     "OpenMOSS-Team/MOSS-TTS-v1.5",
    "nano":     "OpenMOSS-Team/MOSS-TTS-Nano",
    "voicegen": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "ttsd":     "OpenMOSS-Team/MOSS-TTSD-v1.0",
}
_DEFAULT_VARIANT = "v1.5"

# Small variants are loaded in fp32: their audio LM is numerically unstable in
# bfloat16 and can emit an immediate end-of-audio token (0 frames generated).
# fp32 is exactly what MOSS's own stability fallback upcasts to, and these models
# are tiny so the memory cost is negligible. The large 8B variants stay in bf16.
_FP32_VARIANTS = {"nano", "voicegen"}


class MossTTSPlugin(ModelPlugin):
    MODEL_ID     = "MOSS-TTS"
    DISPLAY_NAME = "TTS: MOSS-TTS"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = (
        "Expressive multilingual TTS: zero-shot voice cloning, 31 languages, "
        "duration control & inline pauses (OpenMOSS-Team/MOSS-TTS)"
    )

    # MOSS reads its own moss_ref_audio_path (drawn in draw_custom_ui), so it does
    # not use the shared AUDIO_REF / ref_audio_path collection path.
    INPUTS       = InputSpec.PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.SEED,
    ]
    PARAMS = ParamSpec()
    # MOSS runs via transformers trust_remote_code; both libs are already pinned
    # in requirements.  No MOSS pip package is required.
    REQUIRED_PACKAGES = ["transformers", "soundfile", "bitsandbytes"]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _variant_of(scene) -> str:
        v = getattr(scene, "moss_model_variant", _DEFAULT_VARIANT)
        return v if v in _VARIANT_REPOS else _DEFAULT_VARIANT

    @staticmethod
    def _load_models(repo_id: str, prefs, variant: str = _DEFAULT_VARIANT):
        """Load processor + model for *repo_id*; return (processor, model, device)."""
        import torch
        from transformers import AutoModel, AutoProcessor

        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        if not use_cuda or variant in _FP32_VARIANTS:
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        # 4-bit quantization for large variants via bitsandbytes (saves ~12 GB VRAM).
        bnb_config = None
        if use_cuda and variant not in _FP32_VARIANTS:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                )
                print(f"MOSS-TTS: using 4-bit NF4 quantization (compute in {dtype})")
            except ImportError:
                print("MOSS-TTS: bitsandbytes not available, falling back to bf16")

        local_only = bool(getattr(prefs, "local_files_only", False))
        print(f"Loading MOSS-TTS '{repo_id}' ({variant}) on {device} as {dtype} "
              f"(local_files_only={local_only})...")

        # Defensive: a prior MOSS-Nano run may have repointed the global HF
        # dynamic-module cache to a directory that isn't on sys.path. That breaks
        # trust_remote_code imports for the next model (this is what blocked v1.5
        # after Nano ran). Restore the default modules cache and re-register it.
        try:
            import os as _os
            import transformers.dynamic_module_utils as _dmu
            from huggingface_hub.constants import HF_HOME as _HF_HOME
            _default_modules = _os.path.join(_HF_HOME, "modules")
            _os.environ["HF_MODULES_CACHE"] = _default_modules
            _dmu.HF_MODULES_CACHE = _default_modules
            _dmu.init_hf_modules()
        except Exception as _e:
            print(f"MOSS-TTS modules-cache reset skipped ({_e}).")

        # MOSS-TTS remote code sets _keys_to_ignore_on_load_unexpected as a list,
        # but transformers 5.9 does `list | set` (set-union) which raises TypeError.
        # Patch _adjust_missing_and_unexpected_keys to convert list→set before the
        # union, then restore. Wraps both processor (it loads the audio tokenizer)
        # and model loading.
        import transformers.modeling_utils as _mu
        _orig_adjust = _mu.PreTrainedModel._adjust_missing_and_unexpected_keys

        def _patched_adjust(self_, loading_info):
            if isinstance(getattr(self_, "_keys_to_ignore_on_load_unexpected", None), list):
                self_._keys_to_ignore_on_load_unexpected = set(
                    self_._keys_to_ignore_on_load_unexpected
                )
            return _orig_adjust(self_, loading_info)

        # Only forward local_files_only when enabled. The v1.5/TTSD processor
        # forwards **kwargs into ProcessorMixin.__init__, which rejects unknown
        # kwargs ("Unexpected keyword argument local_files_only"); omitting it on
        # the default (False) path keeps both processor styles happy.
        _extra = {"local_files_only": True} if local_only else {}

        def _load_processor(src):
            return AutoProcessor.from_pretrained(src, trust_remote_code=True, **_extra)

        def _load_model(src):
            kw = dict(trust_remote_code=True, **_extra)
            if bnb_config is not None:
                kw["quantization_config"] = bnb_config
                kw["device_map"] = "auto"
            else:
                kw["dtype"] = dtype
            m = AutoModel.from_pretrained(src, **kw)
            if bnb_config is None:
                if variant not in _FP32_VARIANTS:
                    m = m.to(dtype)
                m = m.to(device)
            return m

        _mu.PreTrainedModel._adjust_missing_and_unexpected_keys = _patched_adjust
        try:
            # Processor: try the repo id first (clean for Nano). MOSS's v1.5/TTSD
            # processor does `Path(repo_id)`, which on Windows mangles the id into a
            # backslash path hf_hub_download rejects — so fall back to a local
            # snapshot directory, whose OS-native separators load correctly.
            try:
                processor = _load_processor(repo_id)
            except Exception as e:
                print(f"MOSS-TTS processor load by repo id failed ({e}); retrying via local snapshot...")
                from huggingface_hub import snapshot_download
                local_dir = snapshot_download(repo_id, local_files_only=local_only)
                processor = _load_processor(local_dir)

            # The audio tokenizer processes raw fp32 waveforms — keep it fp32
            # but move it to the compute device.
            if hasattr(processor, "audio_tokenizer") and processor.audio_tokenizer is not None:
                processor.audio_tokenizer = processor.audio_tokenizer.to(device)

            # Model: ALWAYS load from the repo id. transformers resolves the
            # modeling file's relative remote-code imports (e.g. the v1.5 text
            # normalizer) correctly from a repo id, but NOT from a local snapshot
            # dir (FileNotFoundError on the sibling .py). The model's own code has
            # no Path(repo_id) bug, so the repo id works here.
            model = _load_model(repo_id)
        finally:
            _mu.PreTrainedModel._adjust_missing_and_unexpected_keys = _orig_adjust

        # MOSS's reference infer.py forces sdpa attention (also what the stability
        # fallback selects). Harmless if the model lacks the method.
        try:
            model._set_attention_implementation("sdpa")
        except Exception:
            pass

        # MOSS's _resolve_hf_cache_dir() returns a ".cache" folder *next to the
        # cached remote-code module* — already deeply nested. When it lazily
        # downloads the text tokenizer there, the path exceeds Windows' 260-char
        # limit (WinError 206). Redirect it to the standard HF cache *home*.
        # IMPORTANT: MOSS then derives HF_MODULES_CACHE = <this>/modules and sets
        # it GLOBALLY. Returning HF_HOME makes that resolve to the default
        # (~/.cache/huggingface/modules), so the global change is a harmless no-op
        # and other trust_remote_code plugins (Florence-2, etc.) are unaffected.
        try:
            from huggingface_hub.constants import HF_HOME as _HF_HOME
            model._resolve_hf_cache_dir = lambda: _HF_HOME
        except Exception:
            pass

        model.eval()
        p = next(model.parameters(), None)
        if p is not None:
            print(f"MOSS-TTS loaded: dtype={p.dtype}, device={p.device}")
        return processor, model, device

    def load(self, prefs, scene, **kw):
        variant = self._variant_of(scene)
        repo_id = _VARIANT_REPOS[variant]
        try:
            processor, model, device = self._load_models(repo_id, prefs, variant)
        except OSError as e:
            if getattr(prefs, "local_files_only", False):
                raise OSError(
                    "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
                ) from e
            print(f"MOSS-TTS preload failed ({e}), will load on first generate.")
            processor, model, device = None, None, None
        except Exception as e:
            print(f"MOSS-TTS preload failed ({e}), will load on first generate.")
            processor, model, device = None, None, None

        return {
            "pipe": None,
            "processor": processor,
            "model": model,
            "device": device,
            "repo_id": repo_id if model is not None else None,
        }

    def _ensure_loaded(self, pipe_obj, scene, prefs):
        """(Re)load the model when missing or when the chosen variant changed."""
        variant = self._variant_of(scene)
        repo_id = _VARIANT_REPOS[variant]
        if pipe_obj.get("model") is None or pipe_obj.get("repo_id") != repo_id:
            processor, model, device = self._load_models(repo_id, prefs, variant)
            pipe_obj["processor"], pipe_obj["model"], pipe_obj["device"] = processor, model, device
            pipe_obj["repo_id"] = repo_id
        return pipe_obj

    # ------------------------------------------------------------------ generate

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch

        if getattr(prefs, "local_files_only", False) and pipe_obj.get("model") is None \
                and pipe_obj.get("repo_id") is None:
            raise OSError(
                "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
            )

        self.set_phase(inputs, "Loading model")
        pipe_obj = self._ensure_loaded(pipe_obj, scene, prefs)
        processor = pipe_obj["processor"]
        model     = pipe_obj["model"]
        device    = pipe_obj["device"]

        # ---- Read model-specific widgets off the (worker) scene namespace ----
        cfg = {
            "variant":         self._variant_of(scene),
            "language":        getattr(scene, "moss_language",         "AUTO"),
            "duration_tokens": int(getattr(scene, "moss_duration_tokens", 0)),
            "max_new_tokens":  int(getattr(scene, "moss_max_new_tokens", 4096)),
            "temperature":     float(getattr(scene, "moss_temperature", 1.7)),
            "top_p":           float(getattr(scene, "moss_top_p",       0.8)),
            "top_k":           int(getattr(scene, "moss_top_k",         25)),
            # MOSS uses its own moss_ref_audio_path (independent of the shared
            # ref_audio_path), carried through the queue like the other moss_* props.
            "ref_audio":       (getattr(scene, "moss_ref_audio_path", "") or "").strip() or None,
        }
        # VoiceGenerator designs a voice from the text prompt — it takes no
        # reference audio, so ignore any stale Speaker Ref. value.
        if cfg["variant"] == "voicegen":
            cfg["ref_audio"] = None

        # Seed (these generation paths have no generator arg) ------------------
        if inputs.seed:
            torch.manual_seed(int(inputs.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(inputs.seed))

        output_path = solve_path(clean_filename(str(inputs.seed) + "_" + inputs.prompt) + ".wav")
        self.set_phase(inputs, "Generating")
        print(
            f"MOSS-TTS | variant={cfg['variant']} | ref={'yes' if cfg['ref_audio'] else 'no'} "
            f"| lang={cfg['language']!r} | dur_tokens={cfg['duration_tokens']} "
            f"| temp={cfg['temperature']} | top_p={cfg['top_p']} | top_k={cfg['top_k']}"
        )

        # MOSS variants expose different inference APIs. Dispatch by capability:
        #   * Nano / single-model variants implement model.inference(...) which
        #     tokenizes, generates and writes the wav file itself.
        #   * v1.5-style variants use processor.build_user_message + model.generate.
        try:
            if hasattr(model, "inference"):
                self._generate_via_inference(model, processor, device, inputs, cfg, output_path)
            elif hasattr(processor, "build_user_message"):
                self._generate_via_chat(model, processor, device, inputs, cfg, output_path)
            else:
                raise RuntimeError(
                    f"MOSS-TTS variant '{cfg['variant']}' exposes no known inference API "
                    f"(no model.inference and no processor.build_user_message)."
                )
        except Exception as e:
            import traceback
            print(f"MOSS-TTS generation failed: {e}")
            traceback.print_exc()

        return output_path

    @staticmethod
    def _generate_via_inference(model, processor, device, inputs, cfg, output_path) -> None:
        """Nano-style high-level API: model.inference() writes the wav itself."""
        import torch

        ref_audio = cfg["ref_audio"]
        mode = "voice_clone" if ref_audio else "continuation"
        # Nano counts output length in audio "frames"; reuse the duration widget.
        max_new_frames = cfg["duration_tokens"] if cfg["duration_tokens"] > 0 else 300

        # do_sample=True: TTS is stochastic — greedy decoding collapses to an
        # immediate end-of-audio token (0 frames). Sampling is now stable because
        # the prompt is well-formed (we let the model self-load its text tokenizer
        # below; the AutoProcessor-returned tokenizer encoded differently and
        # produced non-finite logits). The temp/top_p/top_k widgets drive sampling.
        # Do NOT pass text_tokenizer: MOSS's reference infer.py lets the model load
        # its own text tokenizer internally via _load_text_tokenizer().
        kwargs = dict(
            text=inputs.prompt,
            output_audio_path=str(output_path),
            mode=mode,
            device=device,
            do_sample=True,
            audio_temperature=cfg["temperature"],
            audio_top_p=cfg["top_p"],
            audio_top_k=cfg["top_k"],
            max_new_frames=int(max_new_frames),
        )
        if ref_audio:
            kwargs["reference_audio_path"] = ref_audio

        try:
            with torch.no_grad():
                model.inference(**kwargs)
        except RuntimeError as e:
            # The decoder raises "padded input size per channel: (0)" when the LM
            # produced no audio frames — usually a reference the model can't clone
            # (low sample-rate / noisy) or text in an unsupported language.
            if "padded input size" in str(e) or "Kernel size can't be greater" in str(e):
                raise RuntimeError(
                    "MOSS-TTS generated no audio (0 frames). Try a cleaner 24kHz+ "
                    "speaker reference, remove the reference, lower the temperature, "
                    "or use text in a well-supported language."
                ) from e
            raise
        print(f"MOSS-TTS saved: {output_path}")

    @staticmethod
    def _generate_via_chat(model, processor, device, inputs, cfg, output_path) -> None:
        """v1.5-style API: build_user_message + model.generate + processor.decode."""
        import numpy as np
        import soundfile as sf
        import torch

        lang_code = None if cfg["language"] == "AUTO" else cfg["language"]
        msg_kwargs: dict = {"text": inputs.prompt}
        if lang_code:
            msg_kwargs["language"] = lang_code
        if cfg["ref_audio"]:
            msg_kwargs["reference"] = [cfg["ref_audio"]]
        if cfg["duration_tokens"] > 0:
            msg_kwargs["tokens"] = cfg["duration_tokens"]

        message = processor.build_user_message(**msg_kwargs)
        batch = processor([message], mode="generation")
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # v1.5's MossTTSDelayModel.generate() is a CUSTOM method (not the standard
        # transformers generate): it has no do_sample/stopping_criteria and uses
        # text_*/audio_* sampling params. The audio_* params map to our widgets;
        # sampling turns on automatically when temperature > 0. No per-token
        # progress hook is available here (custom generate loop).
        gen_kwargs: dict = {
            "input_ids":         input_ids,
            "attention_mask":    attention_mask,
            "max_new_tokens":    cfg["max_new_tokens"],
            "audio_temperature": cfg["temperature"],
            "audio_top_p":       cfg["top_p"],
            "audio_top_k":       cfg["top_k"],
        }

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        decoded = processor.decode(outputs)
        audio = decoded[0].audio_codes_list[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().to("cpu", dtype=torch.float32).numpy()
        audio = np.asarray(audio).flatten()
        sampling_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
        sf.write(str(output_path), audio, sampling_rate)
        print(f"MOSS-TTS saved: {output_path} ({sampling_rate} Hz)")

    # ------------------------------------------------------------------ UI

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        variant = self._variant_of(scene)

        col.prop(scene, "moss_model_variant")
        col.prop(scene, "moss_language")
        col.separator()

        # Voice clone — reference audio. Hidden for VoiceGenerator, which designs
        # a voice from the text prompt and takes no reference audio. MOSS uses its
        # own moss_ref_audio_path, independent of the shared ref_audio_path.
        if variant != "voicegen":
            row = col.row(align=True)
            row.prop(scene, "moss_ref_audio_path", text="Speaker Ref.")
            row.operator(
                "sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER",
            ).target_prop = "moss_ref_audio_path"
            col.separator()

        col.prop(scene, "moss_duration_tokens")
        col.prop(scene, "moss_max_new_tokens")
        col.separator()

        col.prop(scene, "moss_temperature")
        col.prop(scene, "moss_top_p")
        col.prop(scene, "moss_top_k")

        return False
