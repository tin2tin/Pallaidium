"""Danish TTS via Plapre 3 (syvai/plapre-3).

335M LlamaForCausalLM emitting Kanade 25 Hz content codes, decoded to 24 kHz
by the Kanade codec + Vocos. Runs on plain transformers (the plapre library
needs vLLM); the prompt layout mirrors plapre/tasks.py.

Voice: one of five built-in speakers, or clone a Speaker Ref. clip (its
transcript in Text Ref. helps). Long text is split into sentences; each one's
leading/trailing silence is trimmed and over-long internal pauses (the model
likes to hold commas) are shortened before the sentences are joined.
"""

import json
import os
import re

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename

_REPO         = "syvai/plapre-3"
_KANADE_REPO  = "frothywater/kanade-25hz-clean"
_SR           = 24000
_AUDIO_VOCAB  = 12800
_REF_AUDIO_CAP = 250           # voice-reference audio tokens (10 s at 25 Hz)
_MAX_NEW_TOKENS = 500          # 20 s per sentence
_SILENCE_DB   = -40.0          # frame counts as silent this far below the clip's loudest frame
_EDGE_KEEP_S  = 0.02           # breath of silence left at each trimmed edge
_SPEAKERS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "utils", "plapre_speakers.json"
)

_SENT_END = (".", "?", "!", ":", "…")


def _normalize_numbers(text):
    try:
        from num2words import num2words
    except ImportError:
        print("Plapre: num2words not installed - digits are read as-is.")
        return text

    def _replace(m):
        raw = m.group()
        try:
            return num2words(float(raw.replace(",", ".")), lang="da")
        except (ValueError, OverflowError):
            return raw

    return re.sub(r"\d+(?:[,\.]\d+)?", _replace, text)


def _normalize_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    return _normalize_numbers(text).strip()


def _ensure_terminal_punct(text):
    # The stop signal is strongest on sentence-final text.
    t = text.rstrip()
    if not t:
        return t
    if t.endswith(","):
        t = t[:-1]
    return t if t.endswith(_SENT_END) else t + "."


def _tighten_pauses(wav, max_pause):
    """Trim leading/trailing silence and shorten internal pauses to max_pause s (0 = keep)."""
    import numpy as np

    hop = _SR // 100                      # 10 ms frames
    n = len(wav) // hop
    if n == 0:
        return wav
    frames = wav[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    silent = db < db.max() + _SILENCE_DB
    voiced = np.flatnonzero(~silent)
    if voiced.size == 0:
        return wav
    edge = int(_EDGE_KEEP_S * 100)
    first, last = max(0, voiced[0] - edge), min(n, voiced[-1] + 1 + edge)
    keep = np.zeros(n, dtype=bool)
    keep[first:last] = True
    max_f = int(round(max_pause * 100))
    if max_f > 0:
        i = first
        while i < last:
            if not silent[i]:
                i += 1
                continue
            j = i
            while j < last and silent[j]:
                j += 1
            if j - i > max_f:             # keep both ends of the pause, drop the middle
                keep[i + max_f // 2 : j - (max_f - max_f // 2)] = False
            i = j
    return frames[keep].reshape(-1)


def _split_sentences(text):
    # 2+ word chars before the punctuation keeps "H.C." / "f." intact.
    parts = re.split(r"(?<=\w{2}[.!?])\s+", _normalize_text(text))
    out = []
    for p in parts:
        p = re.sub(r"^[-–—]\s+", "", p.strip())  # Danish dialogue dash
        if p:
            out.append(p)
    return out


class Plapre3Plugin(ModelPlugin):
    MODEL_ID     = _REPO
    DISPLAY_NAME = "TTS: Plapre 3 (Danish)"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Danish text-to-speech with built-in voices and voice cloning (syvai/plapre-3)"

    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF | InputSpec.TEXT_REF
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.AUDIO_REF,
        UISection.TEXT_REF,
        UISection.SEED,
    ]
    PARAMS = ParamSpec()
    REQUIRED_PACKAGES = ["kanade_tokenizer", "vocos"]

    def load(self, prefs, scene, **kw):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from kanade_tokenizer import KanadeModel, load_vocoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        local = bool(getattr(prefs, "local_files_only", False))

        print(f"Loading Plapre 3 on {device}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(_REPO, local_files_only=local)
        except (ValueError, KeyError, ImportError, OSError):
            # Some transformers builds can't resolve the custom tokenizer_class.
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(_REPO, local_files_only=local)

        # fp32 is the training precision; lower precision adds end-of-utterance artifacts.
        model = AutoModelForCausalLM.from_pretrained(
            _REPO, dtype=torch.float32, local_files_only=local,
        ).to(device).eval()

        speaker_proj = torch.nn.Linear(128, model.config.hidden_size)
        speaker_proj.load_state_dict(torch.load(
            hf_hub_download(_REPO, "speaker_proj.pt", local_files_only=local),
            map_location="cpu",
        ))
        speaker_proj = speaker_proj.to(device).eval()

        kanade = KanadeModel.from_pretrained(_KANADE_REPO).eval().to(device)
        vocoder = load_vocoder(kanade.config.vocoder_name).to(device)

        with open(_SPEAKERS_JSON, encoding="utf-8") as fh:
            speakers = {
                name: torch.tensor(emb, dtype=torch.float32, device=device)
                for name, emb in json.load(fh).items()
            }

        return {
            "pipe": None, "model": model, "vocoder": vocoder, "feature_extractor": None,
            "tokenizer": tokenizer, "speaker_proj": speaker_proj,
            "kanade": kanade, "speakers": speakers,
        }

    @staticmethod
    def _encode_reference(kanade, path, device):
        """Kanade-encode the first 10 s of a clip -> (content tokens, speaker emb, trimmed?)."""
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1)))
        if sr != _SR:
            wav = torchaudio.functional.resample(wav, sr, _SR)
        cap = _REF_AUDIO_CAP * _SR // 25
        trimmed = wav.shape[0] > cap
        feats = kanade.encode(wav[:cap].to(device))
        return (
            feats.content_token_indices.cpu().tolist(),
            feats.global_embedding.float(),
            trimmed,
        )

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import numpy as np
        import soundfile as sf
        import torch
        from kanade_tokenizer import vocode

        model        = pipe_obj["model"]
        tokenizer    = pipe_obj["tokenizer"]
        speaker_proj = pipe_obj["speaker_proj"]
        kanade       = pipe_obj["kanade"]
        vocoder      = pipe_obj["vocoder"]
        device       = model.device

        seed = inputs.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        g = tokenizer.convert_tokens_to_ids
        audio_base = g("<audio_0>")
        stops = [tokenizer.eos_token_id, g("</audio>")]

        # Voice: a Speaker Ref. clip clones (its tokens + embedding go in the
        # prompt); otherwise a built-in speaker embedding.
        ref_path = inputs.audio_ref if inputs.audio_ref and os.path.isfile(inputs.audio_ref) else None
        prefix = []
        if ref_path:
            ref_tokens, spk_emb, trimmed = self._encode_reference(kanade, ref_path, device)
            ref_text = (inputs.text_ref or "").strip()
            if trimmed and ref_text:
                print("Plapre: Speaker Ref. is longer than 10 s - Text Ref. ignored (only 10 s is used).")
                ref_text = ""
            ref_text_ids = tokenizer.encode(_normalize_text(ref_text), add_special_tokens=False) if ref_text else []
            prefix = (
                [g("<ref_text>")] + ref_text_ids + [g("</ref_text>")]
                + [g("<ref_audio>")] + [audio_base + int(t) for t in ref_tokens[:_REF_AUDIO_CAP]]
                + [g("</ref_audio>")]
            )
            voice_label = f"clone {os.path.basename(ref_path)}"
        else:
            name = getattr(scene, "plapre_speaker", "tor")
            if name not in pipe_obj["speakers"]:
                name = "tor"
            spk_emb = pipe_obj["speakers"][name]
            voice_label = name

        temperature = max(0.05, float(inputs.temperature))
        sentence_pause = max(0.0, float(getattr(scene, "plapre_sentence_pause", 0.1)))
        max_pause = max(0.0, float(getattr(scene, "plapre_max_pause", 0.3)))
        sentences = _split_sentences(inputs.prompt)
        print(f"Plapre 3 | voice={voice_label} | temperature={temperature:.2f} | sentences={len(sentences)}")

        embed = model.get_input_embeddings()
        with torch.no_grad():
            spk_hidden = speaker_proj(spk_emb.to(device).float()).unsqueeze(0)

        def _synth(sentence):
            ids = prefix + [g("<text>")] + tokenizer.encode(
                _ensure_terminal_punct(sentence), add_special_tokens=False
            ) + [g("<audio>")]
            with torch.no_grad():
                inp = torch.cat([spk_hidden, embed(torch.tensor(ids, device=device))], 0).unsqueeze(0)
                out = model.generate(
                    inputs_embeds=inp,
                    attention_mask=torch.ones(inp.shape[:2], dtype=torch.long, device=device),
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=True, temperature=temperature, top_p=0.95, top_k=50,
                    eos_token_id=stops, pad_token_id=tokenizer.eos_token_id,
                )[0].tolist()
            content = []
            for t in out:
                if audio_base <= t < audio_base + _AUDIO_VOCAB:
                    content.append(t - audio_base)
                elif content:
                    break
            return content

        self.set_phase(inputs, "Generating")
        silence = np.zeros(int(sentence_pause * _SR), dtype=np.float32)
        chunks = []
        for i, sentence in enumerate(sentences):
            print(f"[{i + 1}/{len(sentences)}] {sentence[:60]}")
            content = _synth(sentence)
            if not content:
                print("  no audio tokens - retrying once")
                content = _synth(sentence)
            if content:
                with torch.no_grad():
                    mel = kanade.decode(
                        global_embedding=spk_emb.to(device).float(),
                        content_token_indices=torch.tensor(content, dtype=torch.long, device=device),
                    )
                    wav = vocode(vocoder, mel.unsqueeze(0))
                if chunks:
                    chunks.append(silence)
                chunks.append(_tighten_pauses(wav.squeeze().float().cpu().numpy(), max_pause))
            else:
                print("  skipped: no audio generated")
            if inputs.progress_fn is not None:
                inputs.progress_fn(i + 1, len(sentences))

        output_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".wav")
        if chunks:
            sf.write(output_path, np.concatenate(chunks), _SR)
            print(f"Plapre 3 saved: {output_path}")
        else:
            print("Plapre 3: no audio generated.")
        return output_path

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        sub = col.column()
        sub.active = not scene.ref_audio_path
        sub.prop(scene, "plapre_speaker")
        col.prop(scene, "chat_temperature", text="Temperature")
        col.prop(scene, "plapre_sentence_pause")
        col.prop(scene, "plapre_max_pause")
        return False
