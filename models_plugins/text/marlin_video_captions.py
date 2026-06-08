"""
Marlin Video Captions — dense video captioning and event search directly into Blender.

Caption mode  : runs marlin.caption() → inserts a Scene overview strip and one
                TEXT strip per event on the VSE timeline.
Find mode     : runs marlin.caption(), filters events by query string, and adds
                timeline markers (prefixed "MARLIN:") for every match.  Markers
                are kept until the query string changes.

Model : lunahr/Marlin-2B-ungated  (ungated — no HF token required)
New dep: qwen-vl-utils  (installed by DependencyManager phase 1.5, --no-deps)
"""

import os

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs

#_MODEL_ID = "lunahr/Marlin-2B-ungated"
_MODEL_ID = "tintwotin/Marlin-2B-SDNQ-int8"

# ── Scene props registered at import time ─────────────────────────────────────
# draw_custom_ui() runs on every panel redraw, so the props must exist from the
# moment this module is imported (same pattern as joyimage_edit.py).
def _marlin_query_updated(self, context):
    """Remove MARLIN: markers immediately when the user edits the query string."""
    if not context or not getattr(context, "scene", None):
        return
    scene = context.scene
    to_rm = [m for m in scene.timeline_markers if m.name.startswith("MARLIN:")]
    for m in to_rm:
        scene.timeline_markers.remove(m)
    # Reset the 'last executed' sentinel so the next Generate always runs.
    if hasattr(scene, "marlin_last_query"):
        scene.marlin_last_query = ""


try:
    import bpy as _bpy

    if not hasattr(_bpy.types.Scene, "marlin_mode"):
        _bpy.types.Scene.marlin_mode = _bpy.props.EnumProperty(
            name="Mode",
            items=[
                ("CAPTION", "Caption", "Extract all events as VSE text strips"),
                ("FIND",    "Find",    "Add timeline markers for all occurrences"),
            ],
            default="CAPTION",
        )
    if not hasattr(_bpy.types.Scene, "marlin_find_query"):
        _bpy.types.Scene.marlin_find_query = _bpy.props.StringProperty(
            name="Query",
            description="Event to search for — e.g. 'person enters room'",
            default="",
            options={"TEXTEDIT_UPDATE"},
            update=_marlin_query_updated,
        )
    if not hasattr(_bpy.types.Scene, "marlin_last_query"):
        _bpy.types.Scene.marlin_last_query = _bpy.props.StringProperty(
            name="Last Executed Query",
            description="Internal: last query that produced the current MARLIN markers",
            default="",
            options={"HIDDEN"},
        )
except Exception:
    pass


# ── Helpers (adapted from hviske_subtitles) ───────────────────────────────────

def _format_two_lines(text: str) -> str:
    if len(text) <= 45:
        return text
    mid    = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    closest = min(spaces, key=lambda x: abs(x - mid))
    return text[:closest] + "\n" + text[closest + 1:]


def _find_free_channels(seq_editor, start_frame: int, end_frame: int,
                        count: int, start_ch: int = 1) -> list:
    all_strips = list(seq_editor.strips_all)
    free: list = []
    ch = max(1, start_ch)
    while len(free) < count:
        for seq in all_strips:
            if (
                seq.channel == ch
                and seq.frame_final_start < end_frame
                and (seq.frame_final_start + seq.frame_final_duration) > start_frame
            ):
                break
        else:
            free.append(ch)
        ch += 1
    return free


def _apply_text_strip_style(strip, font_size: int = 16, y: float = 0.2) -> None:
    strip.wrap_width  = 0.68
    strip.font_size   = font_size
    strip.location[0] = 0.5
    strip.location[1] = y
    strip.anchor_x    = "CENTER"
    strip.anchor_y    = "TOP"
    strip.alignment_x = "LEFT"
    strip.use_shadow  = True
    strip.use_box     = True
    strip.box_color   = (0, 0, 0, 0.7)


def _make_preview_clip(src_path: str, out_path: str,
                       start_s: float, end_s: float, width: int = 448) -> None:
    """Extract [start_s, end_s), resize to width×h, write H.264 MKV.
    Keeps the source video's own frame rate. MKV avoids MP4's strict DTS
    monotonicity requirement; H.264 is universally available in any PyAV build."""
    import av
    with av.open(src_path) as inp:
        v_in = next((s for s in inp.streams if s.type == "video"), None)
        if v_in is None:
            raise ValueError("no video stream found")
        src_w, src_h = v_in.width, v_in.height
        if not src_w or not src_h:
            raise ValueError(f"unknown source dimensions {src_w}×{src_h}")
        out_h   = max(2, (round(width * src_h / src_w) // 2) * 2)
        src_fps = float(v_in.average_rate or v_in.guessed_rate or 25)
        if start_s > 0:
            inp.seek(int(start_s * 1_000_000))
        with av.open(out_path, "w") as outp:
            v_out         = outp.add_stream("libx264", rate=round(src_fps))
            v_out.width   = width
            v_out.height  = out_h
            v_out.pix_fmt = "yuv420p"
            v_out.options = {"preset": "ultrafast", "crf": "28",
                             "bf": "0", "tune": "zerolatency"}
            for frame in inp.decode(v_in):
                if frame.pts is None:
                    continue
                t = float(frame.pts * v_in.time_base)
                if t < start_s - 0.1:
                    continue
                if t >= end_s:
                    break
                frame     = frame.reformat(width=width, height=out_h, format="yuv420p")
                frame.pts = None  # let encoder assign sequential timestamps
                for pkt in v_out.encode(frame):
                    outp.mux(pkt)
            for pkt in v_out.encode(None):
                outp.mux(pkt)


# Sampling rate used in both load() (env var) and generate() (FPS_MAX_FRAMES).
_SAMPLE_FPS = 1.0

# ── Plugin ────────────────────────────────────────────────────────────────────

class MarlinVideoCaptionsPlugin(ModelPlugin):
    MODEL_ID     = _MODEL_ID
    DISPLAY_NAME = "Marlin: Video Captions"
    MODEL_TYPE   = "text"
    DESCRIPTION  = (
        "Dense video captioning with second-precise timestamps. "
        "Caption mode inserts a Scene overview strip and one text strip per event. "
        "Find mode adds timeline markers for every occurrence of a queried event."
    )

    INPUTS      = InputSpec(0)
    UI_SECTIONS = []
    PARAMS      = ParamSpec()

    REQUIRED_PACKAGES    = ["qwen_vl_utils", "sdnq"]
    requires_input_strip = True
    requires_main_thread = False  # inference runs in worker; bpy writes via bpy.app.timers

    def load(self, prefs, scene, **kw) -> dict:
        import os as _os
        # Set before transformers import — Qwen reads these at module-import time.
        _os.environ["FPS"]                       = str(_SAMPLE_FPS)
        _os.environ["VIDEO_MAX_PIXELS"]          = "100352"
        _os.environ["FPS_MIN_FRAMES"]            = "4"
        _os.environ["FORCE_QWENVL_VIDEO_READER"] = "cv2"  # torchcodec hangs on Windows
        _os.environ["TOKENIZERS_PARALLELISM"]    = "false"

        import torch
        import sdnq  # registers SDNQ quantizer with transformers before from_pretrained
        from transformers import AutoModelForCausalLM

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

        local     = getattr(prefs, "local_files_only", False)
        cache_dir = getattr(prefs, "hf_cache_dir", None) or None

        common = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
            low_cpu_mem_usage=True,
            local_files_only=local,
            cache_dir=cache_dir,
        )

        # Try FA2 first, fall back to SDPA, then eager (no attn kwarg).
        for attn_impl in ["flash_attention_2", "sdpa", None]:
            kwargs = dict(common)
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            try:
                marlin = AutoModelForCausalLM.from_pretrained(_MODEL_ID, **kwargs)
                marlin.eval()
                _ = marlin.processor   # pre-warm lazy-loaded processor
                used = attn_impl or "eager"
                print(f"Marlin: loaded (SDNQ int8 + {used})")
                return {"model": marlin}
            except OSError as e:
                if local:
                    raise OSError(
                        "Weights missing. Uncheck 'Use Local Files Only' in Preferences to download."
                    ) from e
                print(f"Marlin: load attempt failed ({e}), trying next config…")
            except Exception as e:
                print(f"Marlin: load attempt failed ({e}), trying next config…")

        print("Marlin: all load attempts failed.")
        return {"model": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "marlin_mode", expand=True)
        if scene.marlin_mode == "FIND":
            col.prop(scene, "marlin_find_query", text="Find")
        return False

    def generate(self, pipe, inputs: ModelInputs, scene, prefs):
        import bpy
        import torch

        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if ver < (2, 11):
            print(f"Marlin: WARNING — torch >= 2.11.0 recommended; installed: {torch.__version__}")

        mode  = getattr(scene, "marlin_mode",       "CAPTION")
        query = getattr(scene, "marlin_find_query", "").strip()

        # ── locate MOVIE strip ────────────────────────────────────────────
        seq_editor = scene.sequence_editor
        if not seq_editor:
            print("Marlin: No sequence editor found.")
            return None

        _scene_fps = scene.render.fps / scene.render.fps_base
        fps = _scene_fps

        # When called from the render queue, inputs.video_path contains the
        # resolved path of the specific source strip assigned to this job,
        # and inputs.insert_frame_start records where that strip begins on
        # the timeline.  Use these directly to avoid searching bpy and to
        # correctly handle multiple parallel jobs targeting different strips.
        if inputs.video_path and os.path.isfile(inputs.video_path):
            video_path        = inputs.video_path
            strip_start_frame = inputs.insert_frame_start
            trim_start_s      = 0.0
            # Derive duration from the file itself via a quick container probe.
            try:
                import av as _av
                with _av.open(video_path) as _c:
                    _v = next((s for s in _c.streams if s.type == "video"), None)
                    trim_dur_s = float(_v.duration * _v.time_base) if _v and _v.duration else 0.0
            except Exception:
                trim_dur_s = 0.0
            if trim_dur_s <= 0.0:
                # Fallback: derive from the frame range stored in the snapshot
                trim_dur_s = max(1.0, scene.frame_end - strip_start_frame) / fps
            trim_end_s = trim_start_s + trim_dur_s
            print(f"Marlin: source → {video_path}  (queue mode, start_frame={strip_start_frame})")
        else:
            # Interactive / fallback: search the sequence editor for the strip.
            def _usable_movie(s):
                """Readable MOVIE strip whose visible duration is at least 1 second."""
                if s.type != "MOVIE":
                    return False
                p = bpy.path.abspath(s.filepath)
                return bool(p) and os.path.isfile(p) and (s.frame_final_duration / _scene_fps) >= 1.0

            def _is_source(s):
                """Usable MOVIE strip that is not a Pallaidium-generated output."""
                if not _usable_movie(s):
                    return False
                return "Pallaidium_Media" not in bpy.path.abspath(s.filepath)

            # Use active strip if it is usable; otherwise find the best source strip.
            candidate = seq_editor.active_strip
            if not candidate or not _usable_movie(candidate):
                ref_frame = int(candidate.frame_final_start) if candidate else int(scene.frame_current)
                if candidate:
                    print(f"Marlin: Active strip {candidate.name!r} is not usable "
                          f"(type={candidate.type}, dur={int(candidate.frame_final_duration)} frame(s), "
                          f"pos={ref_frame}) — searching for a source strip.")
                sources = [s for s in seq_editor.strips_all if _is_source(s)]
                if not sources:
                    print("Marlin: No usable source MOVIE strip found (≥1s, not a Pallaidium output). "
                          "Add a source video to the VSE.")
                    return None
                # Prefer a strip whose timeline range contains where the active strip sits,
                # then fall back to the longest available strip.
                overlapping = [s for s in sources
                               if s.frame_final_start <= ref_frame
                               < s.frame_final_start + s.frame_final_duration]
                candidate = max(overlapping if overlapping else sources,
                                key=lambda s: s.frame_final_duration)
                print(f"Marlin: Using strip {candidate.name!r} instead.")

            active     = candidate
            video_path = bpy.path.abspath(active.filepath)
            strip_start_frame = int(active.frame_final_start)
            trim_start_s      = getattr(active, "frame_offset_start", 0) / fps
            trim_dur_s        = active.frame_final_duration / fps
            trim_end_s        = trim_start_s + trim_dur_s

            print(f"Marlin: strip={active.name!r}  ch={active.channel}  "
                  f"timeline_start={strip_start_frame}  dur_frames={int(active.frame_final_duration)}  "
                  f"trim=[{trim_start_s:.2f}s, {trim_end_s:.2f}s]  fps={fps:.3f}")
            print(f"Marlin: source → {video_path}")

        # ── find mode: skip if query unchanged ────────────────────────────
        if mode == "FIND":
            if not query:
                print("Marlin: Enter a search query in Find mode.")
                return None
            if query == getattr(scene, "marlin_last_query", ""):
                print(f"Marlin: Markers already current for query {query!r}.")
                return None

        # ── use cached model ──────────────────────────────────────────────
        marlin = pipe.get("model") if pipe else None
        if marlin is None:
            print("Marlin: Model not loaded — click Install / reload the add-on.")
            return None

        # ── downscaled preview clip via PyAV → MKV ───────────────────────
        # Extracts the trimmed window at 640 px wide so GPU prefill is fast.
        # MKV avoids the strict DTS-monotonicity check that causes MP4 to fail.
        import tempfile
        _tmp_path  = None
        _tmp_path  = None
        clip_path  = video_path
        clip_start = trim_start_s

        if trim_start_s > 0.01:
            # Strip has a trim offset — extract only the visible window so Marlin
            # starts at t=0 and events align correctly.
            try:
                with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as _f:
                    _tmp_path = _f.name
                _make_preview_clip(video_path, _tmp_path, trim_start_s, trim_end_s, width=320)
                _sz = os.path.getsize(_tmp_path) if os.path.isfile(_tmp_path) else 0
                print(f"Marlin: temp clip (trimmed) → {_tmp_path}  size={_sz:,} bytes")
                if _sz == 0:
                    raise RuntimeError("temp clip is empty (0 bytes)")
                clip_path  = _tmp_path
                clip_start = 0.0
            except Exception as _pe:
                print(f"Marlin: temp clip failed ({_pe}), using original file with offset.")
                _tmp_path  = None
                clip_path  = video_path
                clip_start = trim_start_s
        else:
            # No trim offset — pass the original file directly.
            # Marlin reads it natively; FPS_MAX_FRAMES limits processing to the strip duration.
            clip_start = 0.0
            print(f"Marlin: no trim — passing original file directly.")

        # [clip_start, clip_start + trim_dur_s) is the valid event window in clip time
        clip_end = clip_start + trim_dur_s

        # FPS/VIDEO_MAX_PIXELS/etc. are fixed in load() before transformers import.
        # Only FPS_MAX_FRAMES is clip-duration-specific and must be set per-generate.
        max_frames  = str(int(trim_dur_s * _SAMPLE_FPS) + 4)
        cap_new_tok = max(768, int(trim_dur_s * 15))
        print(f"Marlin: {trim_dur_s:.2f}s  → ~{int(trim_dur_s * _SAMPLE_FPS)} frames at {_SAMPLE_FPS}fps  "
              f"FPS_MAX_FRAMES={max_frames}  max_new_tokens={cap_new_tok}")

        _prev_max_frames = os.environ.get("FPS_MAX_FRAMES")
        os.environ["FPS_MAX_FRAMES"] = max_frames

        try:
            # ══ Find mode ════════════════════════════════════════════════
            if mode == "FIND":
                self.set_phase(inputs, "Finding events")
                old = [m for m in scene.timeline_markers if m.name.startswith("MARLIN:")]
                for m in old:
                    scene.timeline_markers.remove(m)

                print(f"── Marlin: Find {query!r}  file={clip_path!r}  "
                      f"clip_start={clip_start:.2f}s  strip_start_frame={strip_start_frame}  "
                      f"(fps={_SAMPLE_FPS}, max_frames={max_frames}, max_new_tokens=64) ──")
                find_result = marlin.find(clip_path, event=query, max_new_tokens=64)
                print(f"Marlin find result: {find_result}")

                span = find_result.get("span") if find_result else None
                if not span or not find_result.get("format_ok"):
                    raw = (find_result.get("raw") or "") if find_result else ""
                    print(f"Marlin: No span found for {query!r}. Raw: {raw!r}")
                    scene.marlin_last_query = query
                    return None

                t_start = float(span[0])
                frame   = strip_start_frame + int((t_start - clip_start) * fps)
                frame   = max(frame, strip_start_frame)
                scene.timeline_markers.new(name=f"MARLIN: {query[:30]}", frame=frame)
                scene.frame_current = frame
                scene.marlin_last_query = query
                print(f"Marlin: Marker added for {query!r} → frame {frame}.")
                return None

            # ══ Caption mode ═════════════════════════════════════════════
            self.set_phase(inputs, "Captioning video")
            print(f"── Marlin: Caption  file={clip_path!r}  "
                  f"clip_start={clip_start:.2f}s  strip_start_frame={strip_start_frame}  "
                  f"(fps={_SAMPLE_FPS}, max_frames={max_frames}, strip={trim_dur_s:.1f}s, "
                  f"max_new_tokens={cap_new_tok}) ──")
            result = marlin.caption(clip_path, max_new_tokens=cap_new_tok,
                                    do_sample=False, temperature=0.0)
            print(f"Marlin caption result: {result}")

        finally:
            if _prev_max_frames is None:
                os.environ.pop("FPS_MAX_FRAMES", None)
            else:
                os.environ["FPS_MAX_FRAMES"] = _prev_max_frames
            if _tmp_path:
                try:
                    os.remove(_tmp_path)
                except Exception:
                    pass

        # ── Schedule all bpy mutations on the main thread ─────────────────
        # generate() runs in a worker thread; bpy writes must happen via timers.
        import bpy as _bpy

        if mode == "FIND":
            _query_snap        = query
            _scene_name        = scene.name
            _find_result_snap  = find_result
            _strip_start_snap  = strip_start_frame
            _clip_start_snap   = clip_start
            _fps_snap          = fps

            def _apply_find():
                _s = _bpy.data.scenes.get(_scene_name)
                if _s is None:
                    return None
                span = _find_result_snap.get("span") if _find_result_snap else None
                if not span or not _find_result_snap.get("format_ok"):
                    raw = (_find_result_snap.get("raw") or "") if _find_result_snap else ""
                    print(f"Marlin: No span found for {_query_snap!r}. Raw: {raw!r}")
                    _s.marlin_last_query = _query_snap
                    return None
                t_start = float(span[0])
                frame   = _strip_start_snap + int((t_start - _clip_start_snap) * _fps_snap)
                frame   = max(frame, _strip_start_snap)
                _s.timeline_markers.new(name=f"MARLIN: {_query_snap[:30]}", frame=frame)
                _s.frame_current = frame
                _s.marlin_last_query = _query_snap
                print(f"Marlin: Marker added for {_query_snap!r} → frame {frame}.")
                return None

            _bpy.app.timers.register(_apply_find, first_interval=0.1)
            return None

        # Caption mode — build plain-data snapshot, no bpy references
        scene_text = (result.get("scene") or "").strip()
        events     = result.get("events") or []
        events     = [ev for ev in events if ev["start"] < clip_end and ev["end"] > clip_start]

        if not events:
            print("Marlin: No events returned within the strip duration.")
            return None

        _scene_name       = scene.name
        _scene_text_snap  = scene_text
        _events_snap      = [dict(ev) for ev in events]   # plain dicts, no bpy refs
        _strip_start_snap = strip_start_frame
        _trim_dur_snap    = trim_dur_s
        _clip_start_snap  = clip_start
        _fps_snap         = fps
        _ch_hint_snap     = inputs.insert_channel if inputs.insert_channel > 0 else 1

        def _apply_caption():
            _s = _bpy.data.scenes.get(_scene_name)
            if _s is None or not _s.sequence_editor:
                return None
            _seq          = _s.sequence_editor
            _clip_end_fr  = _strip_start_snap + int(_trim_dur_snap * _fps_snap)
            _need_ch      = 2 if _scene_text_snap else 1
            _channels     = _find_free_channels(_seq, _strip_start_snap, _clip_end_fr + 2,
                                                _need_ch, start_ch=_ch_hint_snap)
            _events_ch    = _channels[0]
            _scene_ch     = _channels[1] if _need_ch == 2 else None

            if _scene_ch and _scene_text_snap:
                ov = _seq.strips.new_effect(
                    name=_scene_text_snap[:63],
                    type="TEXT",
                    frame_start=_strip_start_snap,
                    length=max(1, int(_clip_end_fr - _strip_start_snap)),
                    channel=_scene_ch,
                )
                ov.text = _format_two_lines(_scene_text_snap)
                _apply_text_strip_style(ov, font_size=12, y=0.95)
                ov.box_color = (0, 0, 0, 0.6)

            _created = 0
            for ev in _events_snap:
                ev_start = _strip_start_snap + int((ev["start"] - _clip_start_snap) * _fps_snap)
                ev_end   = _strip_start_snap + int((ev["end"]   - _clip_start_snap) * _fps_snap)
                ev_start = max(ev_start, _strip_start_snap)
                ev_end   = min(ev_end,   int(_clip_end_fr))
                length   = max(1, ev_end - ev_start)
                disp     = _format_two_lines(ev["description"])
                s = _seq.strips.new_effect(
                    name=disp[:63], type="TEXT",
                    frame_start=ev_start, length=length, channel=_events_ch,
                )
                try:
                    s.frame_final_end = ev_end
                except Exception:
                    pass
                s.text = disp
                _apply_text_strip_style(s)
                _created += 1

            print(f"Marlin: Done — {_created} event strip(s) on channel {_events_ch}.")
            return None

        _bpy.app.timers.register(_apply_caption, first_interval=0.1)
        return None
