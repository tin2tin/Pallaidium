"""
Marlin Video Captions - dense video captioning and event search directly into Blender.

Caption mode  : runs marlin.caption() - inserts a Scene overview strip and one
                TEXT strip per event on the VSE timeline.
Find mode     : runs marlin.caption(), filters events by query string, and adds
                timeline markers (prefixed "MARLIN:") for every match.  Markers
                are kept until the query string changes.

Model : tintwotin/Marlin-2B-SDNQ-int8
New dep: qwen-vl-utils, sdnq
"""

import gc
import os
import time

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs

_MODEL_ID = "tintwotin/Marlin-2B-SDNQ-int8"

# - Scene props registered at import time -------------------------------------
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
            description="Event to search for - e.g. 'person enters room'",
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


# - Helpers (adapted from hviske_subtitles) -----------------------------------

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
    """Extract [start_s, end_s), resize to width x h, write H.264 MKV."""
    import av
    with av.open(src_path) as inp:
        v_in = next((s for s in inp.streams if s.type == "video"), None)
        if v_in is None:
            raise ValueError("no video stream found")
        src_w, src_h = v_in.width, v_in.height
        if not src_w or not src_h:
            raise ValueError(f"unknown source dimensions {src_w} x {src_h}")
        out_h   = max(2, (round(width * src_h / src_w) // 2) * 2)
        src_fps = float(v_in.average_rate or v_in.guessed_rate or 25)
        if start_s > 0:
            inp.seek(int(start_s * 1000000))
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
                frame.pts = None
                for pkt in v_out.encode(frame):
                    outp.mux(pkt)
            for pkt in v_out.encode(None):
                outp.mux(pkt)


# Sampling rate used in both load() (env var) and generate() (FPS_MAX_FRAMES).
_SAMPLE_FPS = 1.0

# - Plugin --------------------------------------------------------------------

class MarlinVideoCaptionsPlugin(ModelPlugin):
    MODEL_ID     = _MODEL_ID
    DISPLAY_NAME = "Marlin: Video Captions (SDNQ)"
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
    requires_main_thread = True

    def load(self, prefs, scene, **kw) -> dict:
        import torch
        import os as _os
        _os.environ["FPS"]                       = str(_SAMPLE_FPS)
        _os.environ["VIDEO_MAX_PIXELS"]          = "100352"
        _os.environ["FPS_MIN_FRAMES"]            = "4"
        _os.environ["FORCE_QWENVL_VIDEO_READER"] = "cv2"
        _os.environ["TOKENIZERS_PARALLELISM"]    = "false"

        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from sdnq import SDNQConfig

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

        local     = getattr(prefs, "local_files_only", False)
        cache_dir = getattr(prefs, "hf_cache_dir", None) or None

        triton_available = False
        try:
            import triton
            triton_available = True
        except ImportError:
            triton_available = False

        # Build SDNQ configuration to handle 3D tensors in the checkpoint.
        try:
            sdnq_config = SDNQConfig(
                weights_dtype="int8",
                group_size=128,
                use_quantized_matmul=triton_available,
                quantization_device="cuda",
                return_device="cuda",
                modules_to_not_convert=["vision_tower", "visual", "lm_head", "embedding_projection"],
            )
        except Exception as e:
            print(f"Marlin: SDNQ Triton check failed ({e}) - forcing Eager dequantization.")
            triton_available = False
            sdnq_config = SDNQConfig(
                weights_dtype="int8",
                group_size=128,
                use_quantized_matmul=False,
                quantization_device="cuda",
                return_device="cuda",
                modules_to_not_convert=["vision_tower", "visual", "lm_head", "embedding_projection"],
            )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Loading loop prioritizing SDNQ and SDPA for Windows/Blender stability.
        for attempt_kwargs in [
            dict(trust_remote_code=True, quantization_config=sdnq_config,
                 torch_dtype=torch.float16, attn_implementation="sdpa",
                 device_map={"": "cuda"}, low_cpu_mem_usage=True,
                 local_files_only=local, cache_dir=cache_dir),
            dict(trust_remote_code=True, quantization_config=sdnq_config,
                 torch_dtype=torch.float16, attn_implementation="eager",
                 device_map={"": "cuda"}, low_cpu_mem_usage=True,
                 local_files_only=local, cache_dir=cache_dir),
            dict(trust_remote_code=True, quantization_config=bnb,
                 attn_implementation="sdpa",
                 device_map={"": "cuda"}, low_cpu_mem_usage=True,
                 local_files_only=local, cache_dir=cache_dir),
        ]:
            try:
                marlin = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, **attempt_kwargs)
                marlin.eval()

                try:
                    from sdnq.loader import apply_sdnq_options_to_model
                    marlin = apply_sdnq_options_to_model(marlin, use_quantized_matmul=triton_available)
                except ImportError:
                    pass

                _ = marlin.processor
                sdnq_used = "quantization_config" in attempt_kwargs and isinstance(attempt_kwargs["quantization_config"], SDNQConfig)
                used = f"SDPA+{'SDNQ-int8' if sdnq_used else 'fallback'}"
                print(f"Marlin: loaded ({used})")
                return {"model": marlin}
            except Exception as e:
                print(f"Marlin: load attempt failed ({e}), trying next config...")

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
            print(f"Marlin: WARNING - torch >= 2.11.0 recommended; installed: {torch.__version__}")

        mode  = getattr(scene, "marlin_mode",       "CAPTION")
        query = getattr(scene, "marlin_find_query", "").strip()

        # - Step 1: locate MOVIE strip ------------------------------------
        self.set_phase(inputs, "Step 1: Locating video strip")
        seq_editor = scene.sequence_editor
        if not seq_editor:
            print("Marlin: No sequence editor found.")
            return None

        _scene_fps = scene.render.fps / scene.render.fps_base
        fps = _scene_fps

        if inputs.video_path and os.path.isfile(inputs.video_path):
            video_path        = inputs.video_path
            strip_start_frame = inputs.insert_frame_start
            trim_start_s      = 0.0
            try:
                import av as _av
                with _av.open(video_path) as _c:
                    _v = next((s for s in _c.streams if s.type == "video"), None)
                    trim_dur_s = float(_v.duration * _v.time_base) if _v and _v.duration else 0.0
            except Exception:
                trim_dur_s = 0.0
            if trim_dur_s <= 0.0:
                trim_dur_s = max(1.0, scene.frame_end - strip_start_frame) / fps
            trim_end_s = trim_start_s + trim_dur_s
            print(f"Marlin: source - {video_path} (queue mode, start_frame={strip_start_frame})")
        else:
            def _usable_movie(s):
                if s.type != "MOVIE":
                    return False
                p = bpy.path.abspath(s.filepath)
                return bool(p) and os.path.isfile(p) and (s.frame_final_duration / _scene_fps) >= 1.0

            def _is_source(s):
                if not _usable_movie(s):
                    return False
                return "Pallaidium_Media" not in bpy.path.abspath(s.filepath)

            candidate = seq_editor.active_strip
            if not candidate or not _usable_movie(candidate):
                ref_frame = int(candidate.frame_final_start) if candidate else int(scene.frame_current)
                sources = [s for s in seq_editor.strips_all if _is_source(s)]
                if not sources:
                    print("Marlin: No usable source MOVIE strip found.")
                    return None
                overlapping = [s for s in sources
                               if s.frame_final_start <= ref_frame
                               < s.frame_final_start + s.frame_final_duration]
                candidate = max(overlapping if overlapping else sources,
                                key=lambda s: s.frame_final_duration)

            active     = candidate
            video_path = bpy.path.abspath(active.filepath)
            strip_start_frame = int(active.frame_final_start)
            trim_start_s      = getattr(active, "frame_offset_start", 0) / fps
            trim_dur_s        = active.frame_final_duration / fps
            trim_end_s        = trim_start_s + trim_dur_s

        if mode == "FIND":
            if not query:
                print("Marlin: Enter a search query in Find mode.")
                return None
            if query == getattr(scene, "marlin_last_query", ""):
                print(f"Marlin: Markers already current for query {query!r}.")
                return None

        marlin = pipe.get("model") if pipe else None
        if marlin is None:
            print("Marlin: Model not loaded.")
            return None

        # - Step 2: extract trimmed window --------------------------------
        self.set_phase(inputs, "Step 2: Extracting video window")
        import tempfile
        _tmp_path  = None
        clip_path  = video_path
        clip_start = trim_start_s

        if trim_start_s > 0.01:
            try:
                with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as _f:
                    _tmp_path = _f.name
                _make_preview_clip(video_path, _tmp_path, trim_start_s, trim_end_s, width=320)
                clip_path  = _tmp_path
                clip_start = 0.0
            except Exception as _pe:
                print(f"Marlin: temp clip failed ({_pe}), using original file with offset.")
                _tmp_path  = None
                clip_path  = video_path
                clip_start = trim_start_s
        else:
            clip_start = 0.0

        clip_end = clip_start + trim_dur_s
        max_frames  = str(int(trim_dur_s * _SAMPLE_FPS) + 4)
        cap_new_tok = max(768, int(trim_dur_s * 15))

        _prev_max_frames = os.environ.get("FPS_MAX_FRAMES")
        os.environ["FPS_MAX_FRAMES"] = max_frames

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        try:
            if mode == "FIND":
                self.set_phase(inputs, f"Step 3: Finding — {query}")
                old = [m for m in scene.timeline_markers if m.name.startswith("MARLIN:")]
                for m in old:
                    scene.timeline_markers.remove(m)

                print(f"Marlin: Find {query!r}  file={clip_path!r}")
                t0 = time.time()
                find_result = marlin.find(clip_path, event=query, max_new_tokens=64)
                print(f"Marlin: find completed in {time.time() - t0:.1f}s  result={find_result}")
                span = find_result.get("span") if find_result else None
                if not span or not find_result.get("format_ok"):
                    scene.marlin_last_query = query
                    return None

                t_start = float(span[0])
                frame   = strip_start_frame + int((t_start - clip_start) * fps)
                frame   = max(frame, strip_start_frame)
                scene.timeline_markers.new(name=f"MARLIN: {query[:30]}", frame=frame)
                scene.marlin_last_query = query
                return None

            else:
                self.set_phase(inputs, "Step 3: Captioning video")
                print(f"Marlin: Caption  file={clip_path!r}  "
                      f"dur={trim_dur_s:.1f}s  max_new_tokens={cap_new_tok}")
                t0 = time.time()
                result = marlin.caption(clip_path, max_new_tokens=cap_new_tok,
                                        do_sample=False, temperature=0.0)
                print(f"Marlin: captioning completed in {time.time() - t0:.1f}s")

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
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # - Step 4: insert VSE strips -------------------------------------
        self.set_phase(inputs, "Step 4: Inserting VSE strips")
        scene_text = (result.get("scene") or "").strip()
        events     = result.get("events") or []
        events = [ev for ev in events if ev["start"] < clip_end and ev["end"] > clip_start]

        if not events:
            print("Marlin: No events returned within the strip duration.")
            return None

        clip_end_frame = strip_start_frame + int(trim_dur_s * fps)
        need_ch  = 2 if scene_text else 1
        _ch_hint = inputs.insert_channel if inputs.insert_channel > 0 else 1
        channels = _find_free_channels(seq_editor, strip_start_frame, clip_end_frame + 2,
                                       need_ch, start_ch=_ch_hint)
        events_ch = channels[0]
        scene_ch  = channels[1] if need_ch == 2 else None

        if scene_ch and scene_text:
            ov = seq_editor.strips.new_effect(
                name=scene_text[:63],
                type="TEXT",
                frame_start=strip_start_frame,
                length=max(1, int(clip_end_frame - strip_start_frame)),
                channel=scene_ch,
            )
            ov.text = _format_two_lines(scene_text)
            _apply_text_strip_style(ov, font_size=12, y=0.95)

        created = 0
        for ev in events:
            ev_start = strip_start_frame + int((ev["start"] - clip_start) * fps)
            ev_end   = strip_start_frame + int((ev["end"]   - clip_start) * fps)
            ev_start = max(ev_start, strip_start_frame)
            ev_end   = min(ev_end,   int(clip_end_frame))
            disp     = _format_two_lines(ev["description"])

            s = seq_editor.strips.new_effect(
                name=disp[:63],
                type="TEXT",
                frame_start=ev_start,
                length=max(1, ev_end - ev_start),
                channel=events_ch,
            )
            s.text = disp
            _apply_text_strip_style(s)
            created += 1

        print(f"Marlin: Done — {created} event strip(s) on channel {events_ch}.")
        return None