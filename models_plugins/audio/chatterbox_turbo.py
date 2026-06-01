"""Fast TTS and voice cloning via ChatterboxTurbo (ResembleAI/chatterbox)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename, split_text_for_tts


def _cast_conds_float32(conds, torch):
    """Cast any float64 tensors in Conditionals to float32 in-place."""
    if conds is None:
        return
    for attr, val in list(conds.t3.__dict__.items()):
        if torch.is_tensor(val) and val.dtype == torch.float64:
            setattr(conds.t3, attr, val.float())
    for k, v in list(conds.gen.items()):
        if torch.is_tensor(v) and v.dtype == torch.float64:
            conds.gen[k] = v.float()


class ChatterboxTurboPlugin(ModelPlugin):
    MODEL_ID     = "ChatterboxTurbo"
    DISPLAY_NAME = "TTS/VC: Chatterbox Turbo"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Fast text-to-speech and voice cloning via Chatterbox Turbo"

    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.AUDIO_DURATION,
        UISection.AUDIO_REF,
        UISection.CHAT_PARAMS,
        UISection.SEED,
    ]
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "torchaudio", "chatterbox"]

    def load(self, prefs, scene, **kw):
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        # Patch prepare_conditionals so ve_embed (numpy float64) is cast to float32.
        # The tts_turbo.py library may already be cached in memory, so we patch the
        # class method at runtime rather than relying on the file edit alone.
        if not getattr(ChatterboxTurboTTS.prepare_conditionals, "_pallaidium_patched", False):
            _orig_prep = ChatterboxTurboTTS.prepare_conditionals
            def _patched_prepare(self_m, wav_fpath, exaggeration=0.5, norm_loudness=True):
                _orig_prep(self_m, wav_fpath, exaggeration=exaggeration, norm_loudness=norm_loudness)
                _cast_conds_float32(self_m.conds, torch)
            _patched_prepare._pallaidium_patched = True
            ChatterboxTurboTTS.prepare_conditionals = _patched_prepare

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading ChatterboxTurboTTS on {device}…")
        try:
            model = ChatterboxTurboTTS.from_pretrained(device=device)
            # Also cast the built-in conds.pt tensors (may have been saved as float64)
            _cast_conds_float32(model.conds, torch)
        except Exception as e:
            print(f"ChatterboxTurboTTS preload failed ({e}), will load on first generate.")
            model = None

        return {"pipe": None, "model": model, "vocoder": None, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        import torchaudio as ta
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = pipe_obj["model"]
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        seed = inputs.seed
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        output_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")

        # Pre-process audio reference to float32 to avoid "expected Float but found Double" errors
        import tempfile, os
        audio_ref_path = inputs.audio_ref
        _tmp_ref = None
        if audio_ref_path:
            ref_wav, ref_sr = ta.load(audio_ref_path)
            ref_wav = ref_wav.float()
            _tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            ta.save(_tmp_ref.name, ref_wav, ref_sr)
            _tmp_ref.close()
            audio_ref_path = _tmp_ref.name

        self.set_phase(inputs, "Generating")
        if inputs.is_voice_clone and inputs.audio_ref:
            print(f"ChatterboxTurbo voice cloning: {inputs.audio_ref}")
            vc_model = ChatterboxTurboTTS.from_pretrained(device)
            wav = vc_model.generate(audio_prompt_path=audio_ref_path)
            ta.save(output_path, wav, vc_model.sr)
        else:
            if model is None:
                model = ChatterboxTurboTTS.from_pretrained(device=device)
                pipe_obj["model"] = model

            chunks = split_text_for_tts(inputs.prompt)
            all_chunks = []
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                print(f"Synthesizing chunk {idx + 1}/{len(chunks)}")
                try:
                    wav_chunk = model.generate(
                        chunk,
                        audio_prompt_path=audio_ref_path,
                        exaggeration=inputs.exaggeration,
                        cfg_weight=inputs.pace,
                        temperature=inputs.temperature,
                    )
                    all_chunks.append(wav_chunk.flatten())
                except Exception as e:
                    print(f"Chunk {idx + 1} failed: {e}")

            if all_chunks:
                final_wav = torch.cat(all_chunks, dim=0)
                ta.save(output_path, final_wav.unsqueeze(0), model.sr)
                print(f"ChatterboxTurbo TTS saved: {output_path}")
            else:
                print("ChatterboxTurbo: no audio generated.")

        if _tmp_ref is not None:
            try:
                os.unlink(_tmp_ref.name)
            except OSError:
                pass

        return output_path
