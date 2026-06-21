"""Multilingual TTS and voice cloning via Chatterbox V3 (ResembleAI/chatterbox)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename, split_text_for_tts


class ChatterboxMultilingualPlugin(ModelPlugin):
    MODEL_ID     = "ChatterboxMultilingual"
    DISPLAY_NAME = "TTS/VC: Chatterbox Multilingual"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Multilingual text-to-speech and voice cloning (23 languages) via Chatterbox V3"

    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        # UISection.AUDIO_DURATION,
        UISection.AUDIO_REF,
        UISection.CHAT_PARAMS,
        UISection.SEED,
    ]
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "torchaudio", "chatterbox"]

    def load(self, prefs, scene, **kw):
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading ChatterboxMultilingualTTS on {device}…")
        try:
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        except Exception as e:
            print(f"ChatterboxMultilingualTTS preload failed ({e}), will load on first generate.")
            model = None

        return {"pipe": None, "model": model, "vocoder": None, "feature_extractor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        import torchaudio as ta
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox.vc import ChatterboxVC

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

        language_id = getattr(scene, "chatterbox_mtl_language", "en")

        output_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")

        self.set_phase(inputs, "Generating")
        if inputs.is_voice_clone and inputs.audio_ref:
            print(f"Chatterbox Multilingual voice cloning: {inputs.audio_ref}")
            vc_model = ChatterboxVC.from_pretrained(device)
            wav = vc_model.generate(audio=inputs.audio_ref)
            ta.save(output_path, wav, vc_model.sr)
        else:
            if model is None:
                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
                pipe_obj["model"] = model

            chunks = split_text_for_tts(inputs.prompt)
            all_chunks = []
            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                print(f"Synthesizing chunk {idx + 1}/{len(chunks)} (lang={language_id})")
                try:
                    wav_chunk = model.generate(
                        chunk,
                        language_id=language_id,
                        audio_prompt_path=inputs.audio_ref,
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
                print(f"Chatterbox Multilingual TTS saved: {output_path}")
            else:
                print("Chatterbox Multilingual: no audio generated.")

        return output_path

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "chatterbox_mtl_language")
        return False
