"""
Faster Whisper Transcription — multilingual ASR directly into VSE text strips.

Detected speech is broken into broadcast-standard subtitle chunks using the
same character-count / proportional-timing algorithm as Hviske, then inserted
as TEXT strips on a single free VSE channel.

Requirements (install via add-on preferences → Install):
  faster-whisper   (pulls in ctranslate2 automatically — no transformers dep)
  soundfile        (already in the main requirements)

Workflow:
  1. Add a SOUND strip to the VSE and select it.
  2. Switch to the Text model panel and pick "Transcribe: Faster Whisper".
  3. Choose a model size and language (default: Large-v3-turbo, auto-detect).
  4. Click Generate — subtitle strips appear on a free timeline channel.

Model download sizes (first run only, cached afterwards):
  tiny ~39 MB  ·  base ~74 MB  ·  small ~244 MB  ·  medium ~769 MB
  large-v3-turbo ~809 MB  ·  large-v3 ~3.1 GB
"""

import gc
import os
import re
import time

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs


# ---------------------------------------------------------------------------
# Model download sizes for console messages
# ---------------------------------------------------------------------------
_MODEL_SIZES = {
    "tiny":           "~39 MB",
    "base":           "~74 MB",
    "small":          "~244 MB",
    "medium":         "~769 MB",
    "large-v3-turbo": "~809 MB",
    "large-v3":       "~3.1 GB",
}

# Systran HuggingFace repo names used by faster-whisper for cache lookup
_HF_REPOS = {
    "tiny":           "Systran/faster-whisper-tiny",
    "base":           "Systran/faster-whisper-base",
    "small":          "Systran/faster-whisper-small",
    "medium":         "Systran/faster-whisper-medium",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
    "large-v3":       "Systran/faster-whisper-large-v3",
}


def _is_model_cached(model_size: str, cache_dir: str | None) -> bool:
    """Return True if the model appears to be in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir
        repo_id = _HF_REPOS.get(model_size, "")
        if not repo_id:
            return False
        cache_info = scan_cache_dir(cache_dir or None)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Subtitle helpers — identical to hviske_subtitles.py
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


def _break_into_subtitle_chunks(start: float, end: float, text: str,
                                 max_chars: int = 80) -> list:
    """Slice one long utterance into broadcast-standard subtitle chunks.

    Proportional timing: each chunk's duration is proportional to its
    character count within the parent segment, capped at a comfortable
    reading speed (~8 chars/s, minimum 2 s).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts   = re.split(r"(?<=[.!?,]) +", text)
    chunks: list = []
    current = ""

    for part in parts:
        fits = len(current) + len(part) + (1 if current else 0) <= max_chars
        if fits:
            current = (current + " " + part) if current else part
        else:
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                words = part.split()
                temp  = ""
                for w in words:
                    if len(temp) + len(w) + (1 if temp else 0) <= max_chars:
                        temp = (temp + " " + w) if temp else w
                    else:
                        chunks.append(temp)
                        temp = w
                current = temp
            else:
                current = part
    if current:
        chunks.append(current)

    total_chars = sum(len(c) for c in chunks)
    total_dur   = end - start
    subs        = []
    cur_start   = start

    for c in chunks:
        if not total_chars:
            break
        frac     = len(c) / total_chars
        orig_dur = total_dur * frac
        disp_dur = min(orig_dur, max(2.0, len(c) * 0.12))
        subs.append({
            "start": cur_start,
            "end":   cur_start + disp_dur,
            "text":  c.strip(),
        })
        cur_start += orig_dur

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

class FasterWhisperTranscribePlugin(ModelPlugin):
    MODEL_ID     = "faster-whisper-transcribe"
    DISPLAY_NAME = "Transcribe: Faster Whisper"
    MODEL_TYPE   = "text"
    DESCRIPTION  = (
        "Multilingual speech-to-text via Faster Whisper. "
        "Select a SOUND strip, choose model size and language, then Generate. "
        "Subtitle strips appear on a free VSE channel with proportional timing."
    )

    INPUTS      = InputSpec(0)   # plugin locates the SOUND strip itself
    UI_SECTIONS = []
    PARAMS      = ParamSpec()

    REQUIRED_PACKAGES = ["faster_whisper"]

    requires_input_strip = True   # user must select a strip
    requires_main_thread = True   # generate() calls bpy directly

    def load(self, prefs, scene, **kw) -> dict:
        return {}

    def generate(self, pipe, inputs: ModelInputs, scene, prefs):
        import bpy
        import torch
        from faster_whisper import WhisperModel

        _cache_dir = prefs.hf_cache_dir or None

        model_size   = getattr(scene, "whisper_model_size", "large-v3-turbo")
        lang_code    = getattr(scene, "whisper_language",   "auto")
        language     = None if lang_code == "auto" else lang_code

        seq_editor = scene.sequence_editor
        if not seq_editor:
            print("Whisper Transcribe: No sequence editor found.")
            return None

        render = scene.render
        fps    = render.fps / render.fps_base

        # ── 0. locate the source SOUND strip ───────────────────────────────
        # In queue mode inputs.audio_ref carries the resolved path; in
        # interactive mode we search the sequence editor (same as Hviske).
        if inputs.audio_ref and os.path.isfile(inputs.audio_ref):
            audio_path        = inputs.audio_ref
            strip_start_frame = inputs.insert_frame_start
            audio_offset      = 0.0
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
                # Try to find a SOUND strip by timeline position (queue mode
                # where insert_frame_start marks the source strip's location).
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
                    "Whisper Transcribe: No source SOUND strip found. "
                    "Add your audio file to the VSE as a SOUND strip and select it."
                )
                return None

            audio_path        = bpy.path.abspath(candidate.sound.filepath)
            strip_start_frame = candidate.frame_final_start
            audio_offset      = getattr(candidate, "frame_offset_start", 0) / fps

        if not os.path.isfile(audio_path):
            print(f"Whisper Transcribe: Audio file not found: {audio_path}")
            return None

        # ── 1. download / load model ────────────────────────────────────────
        cached = _is_model_cached(model_size, _cache_dir)
        size_str = _MODEL_SIZES.get(model_size, "")

        if cached:
            phase_label = f"Step 1: Loading {model_size} from cache"
            print(f"Whisper Transcribe: Loading {model_size!r} from local cache …")
        else:
            if prefs.local_files_only:
                raise OSError(
                    f"Whisper model '{model_size}' not found in local cache. "
                    "Uncheck 'Use Local Files Only' in Add-on Preferences to download it."
                )
            phase_label = f"Step 1: Downloading {model_size} ({size_str}) — first run"
            print(
                f"Whisper Transcribe: Downloading {model_size!r} ({size_str}) from "
                f"Systran/faster-whisper-{model_size} — this only happens once."
            )
            print("Whisper Transcribe: Download progress is shown in the HuggingFace console output above.")

        self.set_phase(inputs, phase_label)

        device       = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        t_load = time.time()
        w_model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=_cache_dir,
        )
        print(
            f"Whisper Transcribe: Model ready on {device} ({compute_type}) "
            f"in {time.time() - t_load:.1f}s"
        )

        # ── 2. transcribe ───────────────────────────────────────────────────
        lang_label = lang_code if lang_code != "auto" else "auto-detect"
        self.set_phase(inputs, f"Step 2: Transcribing ({lang_label})")
        print(
            f"Whisper Transcribe: Transcribing {os.path.basename(audio_path)!r}  "
            f"language={lang_label!r} …"
        )

        t0 = time.time()
        segments_gen, info = w_model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=False,   # Hviske-style chunking uses segment timestamps
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        audio_duration = max(1.0, getattr(info, "duration", 1.0))
        detected_lang  = getattr(info, "language", lang_code)
        print(f"Whisper Transcribe: Detected language: {detected_lang!r}  duration: {audio_duration:.1f}s")

        # Consume the lazy generator, reporting progress per segment
        seg_list: list = []
        for seg in segments_gen:
            seg_list.append(seg)
            elapsed_audio = seg.end
            if inputs.progress_fn is not None:
                inputs.progress_fn(int(elapsed_audio), int(audio_duration))
            print(
                f"  [{elapsed_audio:6.1f}s / {audio_duration:.1f}s]  {seg.text.strip()}"
            )

        print(
            f"Whisper Transcribe: Transcription done in {time.time() - t0:.1f}s  "
            f"({len(seg_list)} segments)"
        )

        del w_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # ── 3. build subtitles — Hviske algorithm ───────────────────────────
        self.set_phase(inputs, "Step 3: Building subtitles")
        all_subs: list = []
        for seg in seg_list:
            if not seg.text.strip():
                continue
            all_subs.extend(_break_into_subtitle_chunks(seg.start, seg.end, seg.text))
        all_subs.sort(key=lambda x: x["start"])

        if not all_subs:
            print("Whisper Transcribe: No subtitles generated.")
            return None

        print(f"Whisper Transcribe: {len(all_subs)} subtitle chunks ready.")

        # ── 4. insert text strips ───────────────────────────────────────────
        self.set_phase(inputs, "Step 4: Inserting VSE strips")
        last_end_frame = strip_start_frame + int(all_subs[-1]["end"] * fps) + 2
        _ch_hint = inputs.insert_channel if inputs.insert_channel > 0 else 1
        channel  = _find_free_channel(seq_editor, strip_start_frame, last_end_frame,
                                       start_ch=_ch_hint)

        print(f"Whisper Transcribe: Inserting {len(all_subs)} strips on channel {channel}")

        created = 0
        for sub in all_subs:
            sub_start_f = strip_start_frame + int((sub["start"] - audio_offset) * fps)
            sub_end_f   = strip_start_frame + int((sub["end"]   - audio_offset) * fps)
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
                new_strip.right_handle = sub_end_f
            else:
                try:
                    new_strip.frame_final_end = sub_end_f
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
            created += 1

        print(
            f"Whisper Transcribe: Done — {created} subtitle strips on channel {channel}."
        )
        return None

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "whisper_model_size")
        col.prop(scene, "whisper_language")
        return False
