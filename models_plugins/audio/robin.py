"""Text-to-speech with voice design, voice cloning and re-voicing via Robin (FixedSeed/Robin).

3.58B model (2.17B planner + 1.41B flow-matching decoder), 48 kHz output, English and
Japanese. bf16 peaks ~8-10 GB VRAM.

  Prompt                    -> the text to say; [laughs] [sigh] [crying] [shouting] work inline
  Voice                     -> the voice in words ("a clear, confident, mid-pitched voice")
  Speaker Ref. (+Text Ref.) -> clone: speak in that clip's voice (Text Ref. = its transcript)
  Re-voice + Speaker Ref.   -> perform that recording (max 45 s) again in the Voice / LoRA
                               voice, keeping words, timing and delivery; the prompt must be
                               exactly what the recording says
  Robin LoRA                -> Robin's own character voice

Long text is split into chunks of ~35 words; each chunk is checked (silence, too short/long)
and re-rolled up to twice, then the chunks are level-matched and joined.

The model ships its own ``robin`` package (no pip install): ``_robin_runtime`` puts the
repo's ``package/`` on sys.path and builds the v3 speech tokenizer with a private
s3tokenizer 0.3.0 (the add-on pins 0.2.0). License: Fixed Seed Open License.
"""

import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename
from . import _robin_runtime as rt


class RobinPlugin(ModelPlugin):
    MODEL_ID     = rt.REPO_ID
    DISPLAY_NAME = "TTS: Robin (voice design, EN/JA, 48 kHz)"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = ("Text-to-speech with voice design from a description, voice cloning (Speaker Ref.), "
                    "re-voicing of a recording and inline [laughs]/[sigh] tags (FixedSeed/Robin)")

    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF | InputSpec.TEXT_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.AUDIO_REF,
        UISection.TEXT_REF,
        UISection.STEPS,
        UISection.GUIDANCE,
        UISection.SEED,
    ]
    # steps = decoder steps (Robin's default is 32; 64 listened clearly cleaner, less voice doubling).
    # guidance = cfg: above 1 follows the Voice description more closely (try 3); only used
    # with a Voice description and no Speaker Ref.
    PARAMS       = ParamSpec(steps=64, guidance=1.0)
    # s3tokenizer / onnx are checked in is_available(); the robin package comes from the repo.
    REQUIRED_PACKAGES = []

    supports_inpaint = False
    supports_img2img = False
    show_enhance     = False
    supports_batch   = True

    def is_available(self) -> bool:
        import importlib.util as ilu
        return all(ilu.find_spec(m) is not None
                   for m in ("torch", "torchaudio", "safetensors", "tokenizers", "soundfile", "onnx", "einops"))

    def load(self, prefs, scene, **kw):
        import torch

        local = bool(getattr(prefs, "local_files_only", False))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Robin on {device} (bf16)...")
        robin = rt.load_robin(device, torch.bfloat16, local_files_only=local)
        return {"pipe": None, "model": robin.planner, "vocoder": None,
                "feature_extractor": None, "robin": robin, "lora_loaded": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import soundfile as sf
        import torch

        robin = pipe_obj["robin"]
        local = bool(getattr(prefs, "local_files_only", False))

        text = (inputs.prompt or "").strip()
        if not text:
            raise ValueError("Robin: the prompt is empty - write the text to say.")
        voice = (getattr(scene, "robin_voice", "") or "").strip() or None
        lang = getattr(scene, "robin_lang", "en")
        revoice = bool(getattr(scene, "robin_revoice", False))
        use_lora = bool(getattr(scene, "robin_lora", False))
        lora_scale = float(getattr(scene, "robin_lora_scale", 1.0))
        ref = inputs.audio_ref if inputs.audio_ref and os.path.isfile(inputs.audio_ref) else None
        ref_text = (inputs.text_ref or "").strip() or None
        cfg = max(1.0, float(inputs.guidance or 1.0))
        steps = max(8, int(inputs.steps or 64))

        # LoRA: attach/detach only when the requested state changed since the last job.
        want = round(lora_scale, 4) if use_lora else None
        if pipe_obj.get("lora_loaded") != want:
            if use_lora:
                if robin.lora is None:
                    robin.load_lora(rt.ensure_lora(local), lora_scale)
                else:
                    robin.set_lora_scale(lora_scale)
            else:
                robin.remove_lora()
            pipe_obj["lora_loaded"] = want

        mode = "re-voice" if revoice else ("clone" if ref else ("voice design" if voice else "default voice"))
        seed = int(inputs.seed)
        self.set_phase(inputs, "Generating speech")
        print(f"Robin | {mode} | lang={lang} | lora={use_lora and lora_scale} | cfg={cfg} | steps={steps} | seed={seed} | voice={voice!r}")

        try:
            with torch.no_grad():
                wav, n = rt.synthesize(
                    robin, text, voice=None if (ref and not revoice) else voice, lang=lang, seed=seed, cfg=cfg, steps=steps,
                    reference=ref, reference_text=ref_text, revoice=revoice, lora_scale=lora_scale,
                    progress_fn=inputs.progress_fn, should_cancel=inputs.should_cancel,
                )
        except rt.cancelled_types():
            print("Robin: cancelled")
            return solve_path(clean_filename(f"{seed}_{text}") + ".wav")

        out = solve_path(clean_filename(f"{seed}_{text}") + ".wav")
        sf.write(out, wav, rt.SAMPLE_RATE, subtype="PCM_24")
        print(f"Robin saved {len(wav) / rt.SAMPLE_RATE:.1f}s ({n} chunk{'s' if n != 1 else ''}) -> {out}")
        return out

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "robin_revoice")
        sub = col.column()
        sub.active = not (scene.ref_audio_path and not scene.robin_revoice)
        sub.prop(scene, "robin_voice")
        col.prop(scene, "robin_lang")
        col.prop(scene, "robin_lora")
        if scene.robin_lora:
            col.prop(scene, "robin_lora_scale")
        return False
