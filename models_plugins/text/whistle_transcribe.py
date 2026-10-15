"""
Whistle Transcription — tiny CPU speech-to-text directly into VSE text strips.

Whistle (Cactus-Compute/whistle) is a 17 MB on-device ASR model running on the
Needle C engine through ctypes: no torch, no GPU, so it never competes with
diffusion models for VRAM. Languages: English, German, French, Spanish,
Italian, Dutch and Polish (auto-detected unless one is chosen).

Unlike the Faster Whisper plugin (proportional timing per segment), subtitles
are cut from Whistle's word-level timestamps, so every chunk starts and ends
on the spoken words. Optional keywords bias decoding towards names and terms.

Requirements (install via add-on preferences → Install):
  cactus-needle   (pure Python; only needs huggingface_hub)
  soundfile, soxr (already in the main requirements)

First run downloads the Needle engine (Cactus-Compute/needle3, ~1 MB DLL) and
the weights (Cactus-Compute/whistle, ~17 MB) into ~/.cache/cactus-needle.

Workflow:
  1. Add a SOUND strip to the VSE and select it.
  2. Switch to the Text model panel and pick "Transcribe: Whistle (CPU)".
  3. Choose a language and optional keywords.
  4. Click Generate — subtitle strips appear on a free timeline channel.
"""

import os
import re
import time

# cactus-needle sends anonymous usage pings unless disabled.
os.environ.setdefault("NEEDLE_TELEMETRY", "0")

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs
from ...utils.helpers import set_ai_metadata_from_dict


_SAMPLE_RATE = 16000
_LANGUAGES   = ("en", "de", "fr", "es", "it", "nl", "pl")

# Audio is fed in windows of about this length, cut at the quietest point
# near the end, so long files report progress and stay within the engine's
# result buffer.
_WINDOW_S      = 30.0
_CUT_SEARCH_S  = 5.0

# Subtitle chunking from word timestamps
_MAX_CHARS     = 80     # two lines of ~40 chars
_MAX_DURATION  = 7.0    # seconds on screen
_PAUSE_BREAK   = 0.8    # silence between words that starts a new subtitle
_MIN_DURATION  = 1.0    # seconds — short chunks are held longer if room


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_16k_mono(path: str):
    """Read any soundfile-supported file as 16 kHz mono float32."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != _SAMPLE_RATE and mono.size:
        try:
            import soxr
            mono = soxr.resample(mono, rate, _SAMPLE_RATE, quality="HQ")
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(rate), _SAMPLE_RATE)
            mono = resample_poly(mono, _SAMPLE_RATE // g, int(rate) // g)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _split_windows(audio) -> list:
    """Return (start_sample, end_sample) windows of ~_WINDOW_S, each cut at the
    quietest 20 ms frame within the last _CUT_SEARCH_S of the window."""
    import numpy as np

    total  = len(audio)
    win    = int(_WINDOW_S * _SAMPLE_RATE)
    search = int(_CUT_SEARCH_S * _SAMPLE_RATE)
    frame  = int(0.02 * _SAMPLE_RATE)
    spans: list = []
    start = 0
    while start < total:
        end = start + win
        if end >= total:
            spans.append((start, total))
            break
        seg = audio[end - search:end]
        n   = len(seg) // frame
        rms = np.sqrt((seg[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
        cut = end - search + int(np.argmin(rms)) * frame + frame // 2
        spans.append((start, cut))
        start = cut
    return spans


# ---------------------------------------------------------------------------
# Subtitle helpers
# ---------------------------------------------------------------------------

def _format_two_lines(text: str) -> str:
    """Split long text at the word nearest the midpoint."""
    if len(text) <= 45:
        return text
    mid    = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    closest = min(spaces, key=lambda x: abs(x - mid))
    return text[:closest] + "\n" + text[closest + 1:]


def _words_to_subtitles(words: list) -> list:
    """Group timed words into subtitle chunks.

    A chunk closes when adding the next word would exceed _MAX_CHARS or
    _MAX_DURATION, when the speaker pauses longer than _PAUSE_BREAK, or after
    sentence-ending punctuation once the chunk holds a reasonable amount.
    """
    subs: list = []
    cur: list  = []

    def _flush():
        if cur:
            subs.append({
                "start": cur[0]["start"],
                "end":   cur[-1]["end"],
                "text":  " ".join(w["word"] for w in cur),
            })
            cur.clear()

    for w in words:
        if cur:
            text_len = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
            if (
                text_len > _MAX_CHARS
                or w["end"] - cur[0]["start"] > _MAX_DURATION
                or w["start"] - cur[-1]["end"] > _PAUSE_BREAK
            ):
                _flush()
        cur.append(w)
        if re.search(r"[.!?]$", w["word"]) and sum(len(x["word"]) + 1 for x in cur) >= 20:
            _flush()
    _flush()

    # Hold short chunks a little longer, never overlapping the next one.
    for i, s in enumerate(subs):
        limit = subs[i + 1]["start"] if i + 1 < len(subs) else s["end"] + _MIN_DURATION
        if s["end"] - s["start"] < _MIN_DURATION:
            s["end"] = min(s["start"] + _MIN_DURATION, limit)
    return subs


def _find_free_channel(seq_editor, start_frame: int, end_frame: int,
                        start_ch: int = 1) -> int:
    """Return the lowest channel number entirely free in [start_frame, end_frame)."""
    all_strips = list(seq_editor.strips_all)
    ch = max(1, start_ch)
    while True:
        for seq in all_strips:
            if (
                seq.channel == ch
                and seq.frame_final_start < end_frame
                and (seq.frame_final_start + seq.frame_final_duration) > start_frame
            ):
                break
        else:
            return ch
        ch += 1


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class WhistleTranscribePlugin(ModelPlugin):
    MODEL_ID     = "whistle-transcribe"
    DISPLAY_NAME = "Transcribe: Whistle (CPU)"
    MODEL_TYPE   = "text"
    DESCRIPTION  = (
        "Tiny CPU speech-to-text (17 MB, no GPU) for EN/DE/FR/ES/IT/NL/PL. "
        "Select a SOUND strip and Generate — subtitle strips are timed from "
        "word timestamps on a free VSE channel."
    )

    INPUTS      = InputSpec(0)   # plugin locates the SOUND strip itself
    UI_SECTIONS = []
    PARAMS      = ParamSpec()

    REQUIRED_PACKAGES = ["needle", "soundfile"]

    requires_input_strip = True   # user must select a strip
    requires_main_thread = True   # generate() calls bpy directly
    supports_batch       = False  # deterministic transcription → batch makes identical copies

    def load(self, prefs, scene, **kw) -> dict:
        return {}

    def generate(self, pipe, inputs: ModelInputs, scene, prefs):
        import bpy
        from needle import Whistle

        lang_code = getattr(scene, "whisper_language", "auto") or "auto"
        if lang_code != "auto" and lang_code not in _LANGUAGES:
            print(
                f"Whistle Transcribe: language {lang_code!r} is not supported "
                f"(only {', '.join(_LANGUAGES)}) — auto-detecting instead."
            )
            lang_code = "auto"
        language = None if lang_code == "auto" else lang_code
        keywords_str = getattr(scene, "whistle_keywords", "") or ""
        keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] or None

        # Strips must land in the sequencer scene (Blender 5.x), see
        # faster_whisper_transcribe.py.
        seq_scene = (
            getattr(bpy.context, "sequencer_scene", None)
            or getattr(bpy.context.workspace, "sequencer_scene", None)
            or scene
        )
        seq_editor = seq_scene.sequence_editor
        if not seq_editor:
            print("Whistle Transcribe: No sequence editor found.")
            return None

        render = scene.render
        fps    = render.fps / render.fps_base

        # ── 0. locate the source SOUND strip ───────────────────────────────
        # In queue mode inputs.audio_ref carries the resolved path; otherwise
        # search the sequence editor (same order as the Whisper plugin).
        # clip_start / clip_len (seconds into the file) restrict transcription
        # to the visible part of a trimmed strip.
        clip_start, clip_len = 0.0, None
        if inputs.audio_ref and os.path.isfile(inputs.audio_ref):
            audio_path        = inputs.audio_ref
            strip_start_frame = inputs.insert_frame_start
        else:
            _pallaidium_dir = "Pallaidium_Media"

            def _is_source_sound(strip):
                if strip.type != "SOUND":
                    return False
                if not getattr(strip, "sound", None):
                    return False
                p = bpy.path.abspath(strip.sound.filepath)
                return _pallaidium_dir not in p and os.path.isfile(p)

            candidate = seq_editor.active_strip
            if candidate and not _is_source_sound(candidate):
                candidate = None
            if candidate is None:
                for seq in seq_editor.strips_all:
                    if seq.select and _is_source_sound(seq):
                        candidate = seq
                        break
            if candidate is None:
                target = inputs.insert_frame_start
                if target:
                    for seq in seq_editor.strips_all:
                        if (
                            _is_source_sound(seq)
                            and seq.frame_final_start <= target
                            < seq.frame_final_start + seq.frame_final_duration
                        ):
                            candidate = seq
                            break
            if candidate is None:
                source_strips = [s for s in seq_editor.strips_all if _is_source_sound(s)]
                if source_strips:
                    candidate = max(source_strips, key=lambda s: s.frame_final_duration)
            if candidate is None:
                print(
                    "Whistle Transcribe: No source SOUND strip found. "
                    "Add your audio file to the VSE as a SOUND strip and select it."
                )
                return None

            audio_path        = bpy.path.abspath(candidate.sound.filepath)
            strip_start_frame = candidate.frame_final_start
            clip_start        = getattr(candidate, "frame_offset_start", 0) / fps
            clip_len          = candidate.frame_final_duration / fps

        if not os.path.isfile(audio_path):
            print(f"Whistle Transcribe: Audio file not found: {audio_path}")
            return None

        # ── 1. load audio ───────────────────────────────────────────────────
        self.set_phase(inputs, "Step 1: Reading audio")
        audio = _load_audio_16k_mono(audio_path)
        a0 = int(clip_start * _SAMPLE_RATE)
        a1 = len(audio) if clip_len is None else a0 + int(clip_len * _SAMPLE_RATE)
        audio = audio[a0:a1]
        audio_duration = len(audio) / _SAMPLE_RATE
        if audio_duration < 0.1:
            print("Whistle Transcribe: Audio is empty.")
            return None

        # ── 2. load model (engine + weights download on first run) ─────────
        self.set_phase(inputs, "Step 2: Loading Whistle")
        print("Whistle Transcribe: Loading model (first run downloads ~18 MB) …")
        t_load = time.time()
        # Large result buffer: word-timestamp JSON for a 30 s window is small,
        # but the engine fails rather than truncates if it does not fit.
        model = Whistle(buffer_size=1 << 22)
        print(f"Whistle Transcribe: Model ready in {time.time() - t_load:.1f}s")

        # ── 3. transcribe window by window ──────────────────────────────────
        lang_label = lang_code if lang_code != "auto" else "auto-detect"
        self.set_phase(inputs, f"Step 3: Transcribing ({lang_label})")
        print(
            f"Whistle Transcribe: Transcribing {os.path.basename(audio_path)!r}  "
            f"({audio_duration:.1f}s, language={lang_label!r}"
            f"{', keywords=' + repr(keywords) if keywords else ''}) …"
        )

        t0 = time.time()
        words: list = []
        detected = None
        for s0, s1 in _split_windows(audio):
            offset = s0 / _SAMPLE_RATE
            result = model.transcribe(
                audio[s0:s1],
                # Lock auto-detection to the first window's language so a
                # quiet or ambiguous window can't switch mid-file.
                language=language or detected,
                keywords=keywords,
                word_timestamps=True,
            )
            if not detected and result.get("language"):
                detected = result["language"]
                print(f"Whistle Transcribe: Detected language: {detected!r}")
            for w in result.get("words") or []:
                text = str(w.get("word", "")).strip()
                if text:
                    words.append({
                        "word":  text,
                        "start": offset + float(w["start"]),
                        "end":   offset + float(w["end"]),
                    })
            if inputs.progress_fn is not None:
                inputs.progress_fn(int(s1 / _SAMPLE_RATE), int(audio_duration))
            print(
                f"  [{s1 / _SAMPLE_RATE:6.1f}s / {audio_duration:.1f}s]  "
                f"{(result.get('text') or '').strip()}"
            )

        print(
            f"Whistle Transcribe: Transcription done in {time.time() - t0:.1f}s  "
            f"({len(words)} words)"
        )

        # ── 4. build subtitles from word timestamps ─────────────────────────
        self.set_phase(inputs, "Step 4: Building subtitles")
        all_subs = _words_to_subtitles(words)
        if not all_subs:
            print("Whistle Transcribe: No speech found — no subtitles generated.")
            return None

        print(f"Whistle Transcribe: {len(all_subs)} subtitle chunks ready.")

        # ── 5. insert text strips ───────────────────────────────────────────
        self.set_phase(inputs, "Step 5: Inserting VSE strips")
        last_end_frame = strip_start_frame + int(all_subs[-1]["end"] * fps) + 2
        _ch_hint = inputs.insert_channel if inputs.insert_channel > 0 else 1
        channel  = _find_free_channel(seq_editor, strip_start_frame, last_end_frame,
                                       start_ch=_ch_hint)

        print(f"Whistle Transcribe: Inserting {len(all_subs)} strips on channel {channel}")

        # Stamped on every subtitle strip: the metadata viewer shows the run's
        # settings and Redo restores them from whichever subtitle is selected.
        meta = {
            "output_type":       "text",
            "model":             self.MODEL_ID,
            "whisper_language":  lang_code,
            "detected_language": detected or "",
            "whistle_keywords":  keywords_str,
            "sound_path":        audio_path,
        }

        created = 0
        for sub in all_subs:
            sub_start_f = strip_start_frame + int(sub["start"] * fps)
            sub_end_f   = strip_start_frame + int(sub["end"]   * fps)
            length_f    = max(1, sub_end_f - sub_start_f)
            disp_text   = _format_two_lines(sub["text"])

            new_strip = seq_editor.strips.new_effect(
                name=disp_text[:63],
                type="TEXT",
                frame_start=sub_start_f,
                length=length_f,
                channel=channel,
            )
            # Blender 4.x / 5.x compatibility
            if hasattr(new_strip, "right_handle"):
                new_strip.right_handle = sub_start_f + length_f
            else:
                try:
                    new_strip.frame_final_end = sub_start_f + length_f
                except Exception:
                    pass

            new_strip.text        = disp_text
            new_strip.wrap_width  = 0.68
            new_strip.font_size   = 16
            new_strip.location[0] = 0.5
            new_strip.location[1] = 0.2
            new_strip.anchor_x    = "CENTER"
            new_strip.anchor_y    = "TOP"
            new_strip.alignment_x = "LEFT"
            new_strip.use_shadow  = True
            new_strip.use_box     = True
            new_strip.box_color   = (0, 0, 0, 0.7)
            set_ai_metadata_from_dict(new_strip, meta, verbose=False)
            created += 1

        print(
            f"Whistle Transcribe: Done — {created} subtitle strips on channel {channel}."
        )
        return None

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "whisper_language")
        lang = getattr(scene, "whisper_language", "auto")
        if lang != "auto" and lang not in _LANGUAGES:
            col.label(text="Whistle: EN/DE/FR/ES/IT/NL/PL only — will auto-detect", icon="ERROR")
        col.prop(scene, "whistle_keywords")
        return False
