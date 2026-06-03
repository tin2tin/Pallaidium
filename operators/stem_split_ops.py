"""Stem splitter: separate a SOUND/MOVIE strip into individual stems via demucs-onnx."""

import threading
import bpy
from bpy.types import Operator

from ..utils.helpers import find_first_empty_channel, solve_path, render_strip_to_wav

STEM_NAMES_4 = ["vocals", "drums", "bass", "other"]
STEM_NAMES_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]

_state = {
    "running": False,
    "phase": "",
    "progress": 0.0,
    "stems_data": None,
    "error": None,
}


def _insert_stems(ctx, stems_dict):
    import soundfile as sf

    scene       = ctx["scene"]
    frame_start = ctx["frame_start"]
    frame_end   = ctx["frame_end"]
    src_channel = ctx["src_channel"]
    strip_name  = ctx["strip_name"]
    selected    = ctx["selected"]
    audio_path  = ctx["audio_path"]

    try:
        sr = sf.info(audio_path).samplerate
    except Exception:
        sr = 44100

    next_min_ch = src_channel + 1
    ed = scene.sequence_editor

    for stem_name in selected:
        arr      = stems_dict[stem_name]          # shape: (channels, samples) float32
        out_path = solve_path(f"{stem_name}_{strip_name}.wav")
        sf.write(out_path, arr.T, sr)             # .T → (samples, channels)

        ch = find_first_empty_channel(frame_start, frame_end)
        ch = max(ch, next_min_ch)
        next_min_ch = ch + 1

        ed.strips.new_sound(
            name=stem_name,
            filepath=out_path,
            channel=ch,
            frame_start=frame_start,
        )
        print(f"[Stem Splitter] Inserted '{stem_name}' on channel {ch}: {out_path}")

    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "SEQUENCE_EDITOR":
                area.tag_redraw()


class SEQUENCER_OT_stem_split(Operator):
    """Separate the active audio/video strip into individual stems via demucs-onnx"""

    bl_idname  = "sequencer.stem_split"
    bl_label   = "Split Stems"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            not _state["running"]
            and context.scene.sequence_editor is not None
            and context.scene.sequence_editor.active_strip is not None
            and context.scene.sequence_editor.active_strip.type in ("SOUND", "MOVIE")
        )

    def invoke(self, context, event):
        scene   = context.scene
        model   = scene.stem_split_model
        all_stems = STEM_NAMES_6 if model == "htdemucs_6s" else STEM_NAMES_4
        selected  = [s for s in all_stems if getattr(scene, f"stem_split_{s}")]
        if not selected:
            self.report({"ERROR"}, "Select at least one stem")
            return {"CANCELLED"}
        return self.execute(context)

    def execute(self, context):
        scene   = context.scene
        strip   = scene.sequence_editor.active_strip
        model   = scene.stem_split_model
        all_stems = STEM_NAMES_6 if model == "htdemucs_6s" else STEM_NAMES_4
        selected  = [s for s in all_stems if getattr(scene, f"stem_split_{s}")]

        _state.update(running=True, phase="Rendering strip to WAV…", progress=0.0,
                      stems_data=None, error=None)

        # Pre-render on main thread — handles trimming, effects, any format
        audio_path = render_strip_to_wav(context, strip)
        if not audio_path:
            _state["running"] = False
            self.report({"ERROR"}, "Failed to render strip audio to WAV")
            return {"CANCELLED"}

        frame_start = strip.frame_final_start
        frame_end   = strip.frame_final_start + strip.frame_final_duration
        src_channel = strip.channel
        strip_name  = strip.name

        _ctx = {
            "scene":       scene,
            "audio_path":  audio_path,
            "frame_start": frame_start,
            "frame_end":   frame_end,
            "src_channel": src_channel,
            "strip_name":  strip_name,
            "selected":    selected,
        }

        context.window_manager.progress_begin(0, 100)

        def _worker():
            try:
                import tqdm.std as _tqdm_std
                _orig_init   = _tqdm_std.tqdm.__init__
                _orig_update = _tqdm_std.tqdm.update
                _bars: dict  = {}

                def _p_init(self2, *a, **kw):
                    _orig_init(self2, *a, **kw)
                    if not getattr(self2, "disable", False):
                        _bars[id(self2)] = [self2.n or 0, self2.total or 0]

                def _p_update(self2, n=1):
                    result = _orig_update(self2, n)
                    entry  = _bars.get(id(self2))
                    if entry is not None:
                        entry[0] = self2.n or 0
                        entry[1] = self2.total or 0
                    total_b = sum(v[1] for v in _bars.values() if v[1] > 0)
                    done_b  = sum(v[0] for v in _bars.values())
                    _state["progress"] = (done_b / total_b) if total_b > 0 else 0.0
                    return result

                _tqdm_std.tqdm.__init__ = _p_init
                _tqdm_std.tqdm.update   = _p_update

                try:
                    _state["phase"] = "Downloading model (first run only)…"
                    from demucs_onnx import separate
                    _state["phase"] = "Separating stems…"
                    # Pass stems=None to run the full model; filter by selected
                    # after — specialist-per-stem breaks guitar/piano on htdemucs_6s.
                    all_stems = separate(
                        audio_path,
                        model=model,
                        stems=None,
                        providers="auto",
                        progress=True,
                    )
                    # Keep only what the user asked for
                    stems = {k: v for k, v in all_stems.items() if k in selected}
                    _state["stems_data"] = stems
                    _state["progress"]   = 1.0
                finally:
                    _tqdm_std.tqdm.__init__ = _orig_init
                    _tqdm_std.tqdm.update   = _orig_update

            except Exception as exc:
                import traceback
                _state["error"] = f"{exc}\n{traceback.format_exc()}"
            finally:
                _state["running"] = False

        threading.Thread(target=_worker, daemon=True).start()

        def _tick():
            try:
                wm = bpy.context.window_manager
                wm.progress_update(int(_state["progress"] * 100))

                if _state["running"]:
                    for win in wm.windows:
                        for area in win.screen.areas:
                            if area.type == "SEQUENCE_EDITOR":
                                area.tag_redraw()
                    return 0.2

                wm.progress_end()

                if _state["error"]:
                    print(f"[Stem Splitter] Error:\n{_state['error']}")
                    return None

                _insert_stems(_ctx, _state["stems_data"])
            except Exception as exc:
                print(f"[Stem Splitter] Timer error: {exc}")
            return None

        bpy.app.timers.register(_tick, first_interval=0.2)
        return {"FINISHED"}
