"""Async render queue for Pallaidium Generative AI.

Jobs are enqueued from the VSE panel, executed one at a time in a background
thread, and strips are inserted back into the VSE timeline from the main
thread via bpy.app.timers (runs on the main thread, no context required).

Thread safety contract:
  - _worker_thread, _cancel_event, _result_queue, _progress_store are only
    mutated from the main thread EXCEPT:
      • _progress_store[job_id] is written by the worker (float assignment
        under the GIL is atomic enough for progress polling)
      • _result_queue.put() is called by the worker (thread-safe by design)
  - All bpy.types / bpy.context access happens in the main thread only.
  - The worker receives a plain dict snapshot; it never touches bpy.
"""
from __future__ import annotations

import gc
import json
import os
import random
import threading
import traceback
import types
import queue as _stdqueue
from datetime import date

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator, PropertyGroup

from ..utils.helpers import (
    ADDON_ID,
    clean_filename,
    clear_cuda_cache,
    closest_divisible_32,
    find_first_empty_channel,
    load_first_frame,
    release_model_cache,
    set_ai_metadata_from_dict,
    set_system_console_topmost,
    show_system_console,
    style_prompt,
)

# ---------------------------------------------------------------------------
# Module-level thread state  (main thread writes, worker reads via closure)
# ---------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_cancel_event = threading.Event()
_result_queue = _stdqueue.Queue()
_progress_store: dict = {}  # job_id → {"progress": float, "phase": str, "step": int, "total": int}
_queue_paused: bool = False  # True = timer stays alive but won't start new jobs
_active_scene = None  # direct bpy.types.Scene reference — avoids bpy.data in timer


# ---------------------------------------------------------------------------
# RenderQueueJob  —  one entry per queued job
# ---------------------------------------------------------------------------

class RenderQueueJob(PropertyGroup):
    """Stores a complete, self-contained snapshot of one generation job."""

    job_id:      StringProperty()
    output_type: StringProperty()   # "image" | "movie" | "audio" | "text"
    model_card:  StringProperty()

    status: EnumProperty(
        items=[
            ("PENDING",    "Pending",    "", "STRIP_COLOR_03", 0),
            ("RUNNING",    "Running",    "", "STRIP_COLOR_05", 1),
            ("COMPLETED",  "Done",       "", "STRIP_COLOR_04", 2),
            ("FAILED",     "Failed",     "", "STRIP_COLOR_01", 3),
            ("CANCELLED",  "Cancelled",  "", "STRIP_COLOR_09", 4),
            ("CANCELLING", "Stopping",   "", "STRIP_COLOR_02", 5),
        ],
        default="PENDING",
    )
    progress:             FloatProperty(min=0.0, max=1.0, default=0.0)
    phase:                StringProperty(default="")
    current_step:         IntProperty(default=0)
    total_steps:          IntProperty(default=0)
    sequencer_scene_name: StringProperty(default="")
    error_message:        StringProperty()
    error_traceback:      StringProperty()
    output_path:          StringProperty()

    # Insertion point — calculated at add-time so user actions cannot
    # displace the intended location.
    insert_frame_start: IntProperty()
    insert_frame_end:   IntProperty()
    insert_channel:     IntProperty()
    insert_duration:    IntProperty()

    # Generation params snapshot
    prompt:       StringProperty()
    neg_prompt:   StringProperty()
    steps:        IntProperty()
    guidance:     FloatProperty()
    seed:         IntProperty()
    use_random:   BoolProperty()
    width:        IntProperty()
    height:       IntProperty()
    frames:       IntProperty()
    image_power:  FloatProperty()
    mode:         StringProperty()  # "txt2img" | "img2img" | "inpaint" | …

    use_lcm:           BoolProperty()
    refine_sd:         BoolProperty()
    adetailer:         BoolProperty()
    aurasr:            BoolProperty()
    remove_silence:    BoolProperty()
    audio_length:      FloatProperty()
    audio_speed_tts:   FloatProperty()
    chat_exaggeration: FloatProperty()
    chat_pace:         FloatProperty()
    chat_temperature:  FloatProperty()
    fps:               FloatProperty()
    music_bpm:         IntProperty()
    music_lyrics:      StringProperty()
    music_key_scale:   StringProperty()
    music_time_signature: StringProperty()

    # Resolved input file paths (resolved from strips at add-time)
    image_path:  StringProperty()
    movie_path:  StringProperty()
    sound_path:  StringProperty()
    audio_path:  StringProperty()
    audio_text:  StringProperty()

    # Prefs snapshot
    hugginface_token: StringProperty()
    local_files_only: BoolProperty()
    generator_ai:     StringProperty()
    hf_cache_dir:     StringProperty()

    # LoRA files — JSON list of {"name": path, "weight": float, "enabled": bool}
    lora_files_json: StringProperty()
    lora_folder:     StringProperty()   # absolute path to the LoRA folder

    # JoyAI spatial editing snapshot (captured at queue-add time)
    joyimage_spatial_mode: StringProperty(default="general")
    joyimage_object:       StringProperty(default="object")
    joyimage_rotate_view:  StringProperty(default="front")
    joyimage_yaw:          FloatProperty(default=0.0)
    joyimage_pitch:        FloatProperty(default=0.0)
    joyimage_zoom:         StringProperty(default="unchanged")

    # Resolved strip paths for model-specific inputs (captured at enqueue time)
    kontext_strip_1_path: StringProperty()   # FLUX Kontext reference / fallback image
    inpaint_mask_path:    StringProperty()   # inpaint mask image path
    qwen_strip_1_path:    StringProperty()   # Qwen Image Edit reference image 1
    qwen_strip_2_path:    StringProperty()   # Qwen Image Edit reference image 2
    qwen_strip_3_path:    StringProperty()   # Qwen Image Edit reference image 3
    klein_strip_1_path:   StringProperty()   # Klein reference image 1
    klein_strip_2_path:   StringProperty()   # Klein reference image 2
    klein_strip_3_path:   StringProperty()   # Klein reference image 3

    # OmniVoice
    omnivoice_instruct:    StringProperty(default="")
    omnivoice_language:    StringProperty(default="")
    omnivoice_preprocess:  BoolProperty(default=True)
    omnivoice_denoise:     BoolProperty(default=True)
    omnivoice_postprocess: BoolProperty(default=True)

    # Stem Splitter
    stem_split_model:  StringProperty(default="htdemucs_ft")
    stem_split_vocals: BoolProperty(default=True)
    stem_split_drums:  BoolProperty(default=True)
    stem_split_bass:   BoolProperty(default=True)
    stem_split_other:  BoolProperty(default=True)
    stem_split_guitar: BoolProperty(default=False)
    stem_split_piano:  BoolProperty(default=False)

    # Klein Schematic LoRA plugin
    klein_schematic_mode:   StringProperty(default="DEPTH")
    klein_schematic_target: StringProperty(default="person")

    # VRAM management — set at run-time based on the next queued job
    should_unload: BoolProperty(default=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_job(scene, job_id: str):
    for job in scene.render_queue:
        if job.job_id == job_id:
            return job
    return None


def _find_next_pending(scene):
    for job in scene.render_queue:
        if job.status == "PENDING":
            return job
    return None


def _find_running_job(scene):
    for job in scene.render_queue:
        if job.status in ("RUNNING", "CANCELLING"):
            return job
    return None


def _find_free_channel(scene, frame_start: int, frame_end: int, preferred: int) -> int:
    """Scan upward from preferred_channel until a free slot is found.

    Called at insert-time so it reflects whatever has been added since the
    job was enqueued.
    """
    channel = max(1, preferred)
    for _ in range(128):
        occupied = any(
            s.channel == channel
            and s.frame_final_start < frame_end
            and (s.frame_final_start + s.frame_final_duration) > frame_start
            for s in scene.sequence_editor.strips_all
        )
        if not occupied:
            return channel
        channel += 1
    return max(1, preferred)


def _queue_solve_path(filename: str, generator_ai_dir: str) -> str:
    """Thread-safe output path resolver — does NOT use bpy.context."""
    out_dir = os.path.join(generator_ai_dir, str(date.today()))
    os.makedirs(out_dir, exist_ok=True)
    base = clean_filename(os.path.splitext(os.path.basename(filename))[0])
    ext = os.path.splitext(filename)[1] or ".png"
    out = os.path.join(out_dir, base + ext)
    i = 1
    while os.path.exists(out):
        out = os.path.join(out_dir, f"{base}({i}){ext}")
        i += 1
    return out


# ---------------------------------------------------------------------------
# Background worker — no bpy access allowed
# ---------------------------------------------------------------------------

def _run_job(snapshot: dict, result_queue, cancel_event, progress_store) -> None:
    """Runs in a background thread.

    Loads the model, runs inference, puts a result dict into result_queue,
    then exits.  Never touches bpy.context; uses SimpleNamespace proxies so
    plugins that read scene/prefs attributes still work.
    """
    job_id = snapshot["job_id"]
    progress_store[job_id] = {"progress": 0.0, "phase": "Queued", "step": 0, "total": 0}
    pipe_obj = None
    model_cache = None  # shared cache dict from main_ops; set once otype is known

    try:
        from ..models import get_plugin
        from ..models.base import ModelInputs

        plugin = get_plugin(snapshot["model_card"])
        if plugin is None:
            raise RuntimeError(f"Plugin not found: {snapshot['model_card']!r}")

        otype = snapshot["output_type"]
        model_card = snapshot["model_card"]

        # ---- Grab the shared model cache for this output type -----------
        from ..operators.main_ops import (
            _pallaidium_model_cache,
            _pallaidium_movie_model_cache,
            _pallaidium_audio_model_cache,
            _pallaidium_text_model_cache,
        )
        model_cache = {
            "image": _pallaidium_model_cache,
            "movie": _pallaidium_movie_model_cache,
            "audio": _pallaidium_audio_model_cache,
            "text":  _pallaidium_text_model_cache,
        }.get(otype)

        # ---- Reconstruct LoRA list first so scene_proxy can reference it ---
        try:
            lora_raw = json.loads(snapshot.get("lora_files_json", "[]"))
        except Exception:
            lora_raw = []
        enabled_items = [
            types.SimpleNamespace(
                name         = item["name"],
                weight_value = item.get("weight", 1.0),
                enabled      = True,
            )
            for item in lora_raw
            if item.get("enabled")
        ]

        # ---- Proxy objects — every field comes from the snapshot ----------
        prefs_proxy = types.SimpleNamespace(
            hugginface_token = snapshot.get("hugginface_token", ""),
            local_files_only = snapshot.get("local_files_only", False),
            generator_ai     = snapshot.get("generator_ai", ""),
            hf_cache_dir     = snapshot.get("hf_cache_dir", ""),
            image_model_card = model_card if otype == "image" else "",
            movie_model_card = model_card if otype == "movie" else "",
            audio_model_card = model_card if otype == "audio" else "",
            text_model_card  = model_card if otype == "text"  else "",
        )
        scene_proxy = types.SimpleNamespace(
            generate_movie_prompt          = snapshot["prompt"],
            generate_movie_negative_prompt = snapshot["neg_prompt"],
            generate_movie_x               = snapshot["width"],
            generate_movie_y               = snapshot["height"],
            generate_movie_frames          = snapshot["frames"],
            movie_num_inference_steps      = snapshot["steps"],
            movie_num_guidance             = snapshot["guidance"],
            movie_num_seed                 = snapshot["seed"],
            movie_use_random               = snapshot["use_random"],
            image_power                    = snapshot["image_power"],
            use_lcm                        = snapshot["use_lcm"],
            refine_sd                      = snapshot["refine_sd"],
            adetailer                      = snapshot["adetailer"],
            aurasr                         = snapshot["aurasr"],
            remove_silence                 = snapshot["remove_silence"],
            audio_length_in_f              = int(snapshot["audio_length"]),
            audio_speed_tts                = snapshot["audio_speed_tts"],
            chat_exaggeration              = snapshot["chat_exaggeration"],
            chat_pace                      = snapshot["chat_pace"],
            chat_temperature               = snapshot["chat_temperature"],
            input_strips                   = (
                "input_strips"
                if (snapshot.get("image_path") or snapshot.get("movie_path"))
                else "input_prompt"
            ),
            image_path                     = snapshot.get("image_path", ""),
            movie_path                     = snapshot.get("movie_path", ""),
            sound_path                     = snapshot.get("sound_path", ""),
            audio_path                     = snapshot.get("audio_path", ""),
            audio_text                     = snapshot.get("audio_text", ""),
            inpaint_selected_strip         = "",
            kontext_strip_1                = "",
            kontext_strip_1_path           = snapshot.get("kontext_strip_1_path", ""),
            qwen_strip_1_path              = snapshot.get("qwen_strip_1_path", ""),
            qwen_strip_2_path              = snapshot.get("qwen_strip_2_path", ""),
            qwen_strip_3_path              = snapshot.get("qwen_strip_3_path", ""),
            music_bpm                      = snapshot["music_bpm"],
            music_lyrics                   = snapshot["music_lyrics"],
            music_key_scale                = snapshot["music_key_scale"],
            music_time_signature           = snapshot["music_time_signature"],
            ip_adapter_face_folder         = "",
            ip_adapter_style_folder        = "",
            svd_decode_chunk_size          = 2,
            svd_motion_bucket_id           = 1,
            img_guidance_scale             = 1.6,
            lora_files                     = enabled_items,
            lora_folder                    = snapshot.get("lora_folder", ""),
            render                         = types.SimpleNamespace(
                fps=round(snapshot.get("fps", 24.0)), fps_base=1.0
            ),
            stem_split_model  = snapshot.get("stem_split_model",  "htdemucs_ft"),
            stem_split_vocals = snapshot.get("stem_split_vocals", True),
            stem_split_drums  = snapshot.get("stem_split_drums",  True),
            stem_split_bass   = snapshot.get("stem_split_bass",   True),
            stem_split_other  = snapshot.get("stem_split_other",  True),
            stem_split_guitar  = snapshot.get("stem_split_guitar",  False),
            stem_split_piano   = snapshot.get("stem_split_piano",   False),
            omnivoice_instruct    = snapshot.get("omnivoice_instruct",    ""),
            omnivoice_language    = snapshot.get("omnivoice_language",    ""),
            omnivoice_preprocess  = snapshot.get("omnivoice_preprocess",  True),
            omnivoice_denoise     = snapshot.get("omnivoice_denoise",     True),
            omnivoice_postprocess = snapshot.get("omnivoice_postprocess", True),
            klein_schematic_mode   = snapshot.get("klein_schematic_mode",   "DEPTH"),
            klein_schematic_target = snapshot.get("klein_schematic_target", "person"),
            klein_strip_1_path     = snapshot.get("klein_strip_1_path",     ""),
            klein_strip_2_path     = snapshot.get("klein_strip_2_path",     ""),
            klein_strip_3_path     = snapshot.get("klein_strip_3_path",     ""),
        )

        mode = snapshot["mode"]

        if cancel_event.is_set():
            result_queue.put({"job_id": job_id, "status": "CANCELLED"})
            return

        # ---- Load model (or reuse from cache) ---------------------------
        progress_store[job_id] = {"progress": 0.02, "phase": "Loading model", "step": 0, "total": 0}
        show_system_console(True)
        set_system_console_topmost(True)

        # Ensure self-registering packages (e.g. sdnq) are imported before
        # generate() so their side-effects (registering quantization backends,
        # etc.) are visible to the loaded model weights.
        import importlib as _importlib
        for _pkg in getattr(plugin, "REQUIRED_PACKAGES", []):
            _pkg_name = _pkg if isinstance(_pkg, str) else _pkg[0]
            try:
                _importlib.import_module(_pkg_name)
            except (ImportError, PermissionError, OSError):
                pass

        # Many plugins load different pipeline objects per mode (e.g. ZImage
        # loads `pipe` for txt2img but `converter` for img2img — never both).
        # A cache built for one mode cannot serve another mode of the same
        # model, so we include `last_mode` in the hit check.
        _schematic_mode = snapshot.get("klein_schematic_mode", "")
        _cache_skip = {"last_model_card", "last_mode", "last_schematic_mode"}
        cache_hit = (
            model_cache is not None
            and model_cache.get("last_model_card") == model_card
            and model_cache.get("last_mode") == mode
            and model_cache.get("last_schematic_mode", "") == _schematic_mode
            and any(v is not None for k, v in model_cache.items() if k not in _cache_skip)
        )

        if cache_hit:
            pipe_obj = model_cache
            print(f"[Queue] Reusing cached model: {model_card} ({mode}) schematic={_schematic_mode or '-'}")
        else:
            # Release a different model (or same model with incompatible mode)
            if (model_cache is not None
                    and (model_cache.get("last_model_card") != model_card
                         or model_cache.get("last_mode") != mode
                         or model_cache.get("last_schematic_mode", "") != _schematic_mode)):
                release_model_cache(model_cache)
            clear_cuda_cache()
            if prefs_proxy.hf_cache_dir:
                os.environ["HF_HUB_CACHE"] = prefs_proxy.hf_cache_dir

            # Set initial state before load
            progress_store[job_id] = {
                "progress": 0.0,
                "phase":    "Loading model",
                "step":     0,
                "total":    0,
            }

            # Patch tqdm.std.tqdm.__init__ and .update directly on the base class.
            # This works regardless of when huggingface_hub imported tqdm, because
            # huggingface_hub.utils.tqdm.tqdm inherits from tqdm.std.tqdm and does
            # not override update() — so our patch is inherited by every instance.
            # Running in the download thread means KeyboardInterrupt here genuinely
            # aborts the HTTP transfer.
            try:
                import tqdm.std as _tqdm_std
            except Exception:
                _tqdm_std = None

            _active_bars: dict = {}   # id(bar) → [bytes_done, bytes_total]

            if _tqdm_std is not None:
                _orig_tqdm_init   = _tqdm_std.tqdm.__init__
                _orig_tqdm_update = _tqdm_std.tqdm.update

                def _patched_tqdm_init(tqdm_self, *a, **kw):
                    _orig_tqdm_init(tqdm_self, *a, **kw)
                    if not getattr(tqdm_self, "disable", False):
                        _active_bars[id(tqdm_self)] = [tqdm_self.n or 0, tqdm_self.total or 0]

                def _patched_tqdm_update(tqdm_self, n=1):
                    if cancel_event.is_set():
                        raise KeyboardInterrupt("Queue job cancelled during download")
                    result = _orig_tqdm_update(tqdm_self, n)
                    entry = _active_bars.get(id(tqdm_self))
                    if entry is not None:
                        entry[0] = tqdm_self.n or 0
                        entry[1] = tqdm_self.total or 0
                    total_b = sum(v[1] for v in _active_bars.values() if v[1] > 0)
                    done_b  = sum(v[0] for v in _active_bars.values())
                    dl_frac = (done_b / total_b) if total_b > 0 else 0.0
                    progress_store[job_id] = {
                        "progress": dl_frac,
                        "phase":    "Downloading model",
                        "step":     max(0, int(done_b  / 1_048_576)),
                        "total":    max(1, int(total_b / 1_048_576)),
                    }
                    return result

                _tqdm_std.tqdm.__init__ = _patched_tqdm_init
                _tqdm_std.tqdm.update   = _patched_tqdm_update

            if getattr(plugin, "requires_main_thread_for_load", False):
                # Plugin's load() crashes in worker threads on Windows
                # (e.g. PyTorch CPU ops via safetensors). Schedule it on
                # the main thread via a one-shot timer; worker blocks here
                # until done, then continues generate() in the thread.
                _load_done   = threading.Event()
                _load_result = {}

                def _main_thread_load():
                    try:
                        _load_result["loaded"] = plugin.load(
                            prefs_proxy, scene_proxy,
                            mode=mode, enabled_items=enabled_items,
                            use_lcm=snapshot["use_lcm"],
                            use_refine=snapshot["refine_sd"],
                            ip_adapter_face_folder="",
                            ip_adapter_style_folder="",
                            local_files_only=snapshot["local_files_only"],
                        )
                    except Exception as _exc:
                        _load_result["error"] = _exc
                    finally:
                        if _tqdm_std is not None:
                            _tqdm_std.tqdm.__init__ = _orig_tqdm_init
                            _tqdm_std.tqdm.update   = _orig_tqdm_update
                        _active_bars.clear()
                        _load_done.set()
                    return None  # don't re-register the timer

                bpy.app.timers.register(_main_thread_load, first_interval=0)
                _load_done.wait()

                if "error" in _load_result:
                    raise _load_result["error"]
                loaded = _load_result["loaded"]
            else:
                try:
                    loaded = plugin.load(
                        prefs_proxy,
                        scene_proxy,
                        mode=mode,
                        enabled_items=enabled_items,
                        use_lcm=snapshot["use_lcm"],
                        use_refine=snapshot["refine_sd"],
                        ip_adapter_face_folder="",
                        ip_adapter_style_folder="",
                        local_files_only=snapshot["local_files_only"],
                    )
                finally:
                    if _tqdm_std is not None:
                        _tqdm_std.tqdm.__init__ = _orig_tqdm_init
                        _tqdm_std.tqdm.update   = _orig_tqdm_update
                    _active_bars.clear()

            # If cancelled during load, stop here
            if cancel_event.is_set():
                result_queue.put({"job_id": job_id, "status": "CANCELLED"})
                return

            progress_store[job_id] = {
                "progress": 0.10, "phase": "Loaded", "step": 0, "total": 0,
            }

            if model_cache is not None and isinstance(loaded, dict):
                model_cache.update(loaded)
                model_cache["last_model_card"] = model_card
                model_cache["last_mode"] = mode
                model_cache["last_schematic_mode"] = _schematic_mode
                pipe_obj = model_cache
            else:
                pipe_obj = loaded

        if cancel_event.is_set():
            result_queue.put({"job_id": job_id, "status": "CANCELLED"})
            return

        # ---- Build inputs -----------------------------------------------
        progress_store[job_id] = {"progress": 0.10, "phase": "Preparing", "step": 0, "total": 0}

        _preserve_dims = getattr(plugin, "preserve_image_dimensions", False)

        init_image = None
        img_path = snapshot.get("image_path", "")
        vid_path = snapshot.get("movie_path", "")
        if img_path and os.path.isfile(img_path):
            init_image = load_first_frame(img_path)
            if init_image and not _preserve_dims:
                init_image = init_image.resize((snapshot["width"], snapshot["height"]))
        if init_image is None and vid_path and os.path.isfile(vid_path):
            init_image = load_first_frame(vid_path)
            if init_image and not _preserve_dims:
                init_image = init_image.resize((snapshot["width"], snapshot["height"]))

        inpaint_mask = None
        mask_path = snapshot.get("inpaint_mask_path", "")
        if mask_path and os.path.isfile(mask_path):
            inpaint_mask = load_first_frame(mask_path)
            if inpaint_mask:
                inpaint_mask = inpaint_mask.resize((snapshot["width"], snapshot["height"]))

        def _phase_fn(label: str) -> None:
            store = progress_store.get(job_id)
            if store is not None:
                store["phase"] = label

        def _progress_fn(step: int, total: int) -> None:
            # Maps denoising steps to the 10 %–99 % band; 0–10 % = load,
            # 100 % = done.
            cur_phase = progress_store.get(job_id, {}).get("phase", "Generating")
            progress_store[job_id] = {
                "progress": 0.10 + 0.89 * (step / max(1, total)),
                "phase":    cur_phase,
                "step":     step,
                "total":    total,
            }

        # audio_length is stored in frames; models expect seconds
        _fps = snapshot.get("fps", 24.0) or 24.0
        _audio_length_in_s = snapshot["audio_length"] / _fps

        _infer_w, _infer_h = snapshot["width"], snapshot["height"]
        if _preserve_dims and init_image is not None:
            _infer_w, _infer_h = init_image.size

        inputs = ModelInputs(
            prompt         = snapshot["prompt"],
            neg_prompt     = snapshot["neg_prompt"],
            mode           = mode,
            image          = init_image,
            inpaint_mask   = inpaint_mask,
            width          = _infer_w,
            height         = _infer_h,
            frames         = snapshot["frames"],
            fps            = snapshot.get("fps", 24.0),
            steps          = snapshot["steps"],
            guidance       = snapshot["guidance"],
            strength       = snapshot["image_power"],
            seed           = snapshot["seed"],
            audio_ref      = snapshot.get("audio_path") or snapshot.get("sound_path") or None,
            text_ref       = snapshot.get("audio_text", ""),
            video_path     = vid_path or None,
            audio_length   = _audio_length_in_s,
            speed          = snapshot["audio_speed_tts"],
            exaggeration   = snapshot["chat_exaggeration"],
            pace           = snapshot["chat_pace"],
            temperature    = snapshot["chat_temperature"],
            remove_silence = snapshot["remove_silence"],
            use_lcm        = snapshot["use_lcm"],
            use_adetailer  = snapshot["adetailer"],
            use_upscale    = snapshot["aurasr"],
            bpm            = snapshot["music_bpm"],
            lyrics         = snapshot["music_lyrics"],
            key_scale      = snapshot["music_key_scale"],
            time_signature = snapshot["music_time_signature"],
            lora_files     = [(i.name, i.weight_value) for i in enabled_items],
            progress_fn    = _progress_fn,
            phase_fn       = _phase_fn,
        )

        # ---- Generate ---------------------------------------------------
        result = plugin.generate(pipe_obj, inputs, scene_proxy, prefs_proxy)

        if cancel_event.is_set():
            result_queue.put({"job_id": job_id, "status": "CANCELLED"})
            return

        # ---- Save output (image plugins → PIL; video/audio → file path) --
        generator_ai = snapshot.get("generator_ai", "") or os.path.join(
            os.path.expanduser("~"), "Pallaidium Media"
        )

        out_path = None
        text_content = None
        if isinstance(result, str) and result.startswith("MULTI_STEM:"):
            out_path = result  # pass through; _queue_insert_strip handles multi-stem
        elif isinstance(result, str) and os.path.isfile(result):
            out_path = result
        elif isinstance(result, str) and result.strip():
            # Plain text result (caption, rewritten prompt) — save to .txt and
            # pass content directly so _queue_insert_strip can create a TEXT strip.
            text_content = result
            fname = clean_filename(f"{snapshot['seed']}_{snapshot['prompt'][:40]}")
            out_path = _queue_solve_path(fname + ".txt", generator_ai)
            with open(out_path, "w", encoding="utf-8") as _fh:
                _fh.write(result)
        elif result is not None and not isinstance(result, str):
            fname = clean_filename(f"{snapshot['seed']}_{snapshot['prompt']}")
            out_path = _queue_solve_path(fname + ".png", generator_ai)
            result.save(out_path)

        progress_store[job_id] = {"progress": 1.0, "phase": "Done", "step": 0, "total": 0}
        result_queue.put({
            "job_id":               job_id,
            "status":               "COMPLETED",
            "output_path":          out_path or "",
            "text_content":         text_content or "",
            "output_type":          otype,
            "frame_start":          snapshot["insert_frame_start"],
            "frame_end":            snapshot["insert_frame_end"],
            "channel":              snapshot["insert_channel"],
            "duration":             snapshot["insert_duration"],
            "sequencer_scene_name": snapshot.get("sequencer_scene_name", ""),
            "prompt":      snapshot["prompt"],
            "neg_prompt":  snapshot["neg_prompt"],
            "seed":        snapshot["seed"],
            "model_card":  snapshot["model_card"],
            "mode":        snapshot["mode"],
            "steps":       snapshot["steps"],
            "guidance":    snapshot["guidance"],
            "width":       snapshot["width"],
            "height":      snapshot["height"],
            "frames":      snapshot["frames"],
        })

    except KeyboardInterrupt:
        # User cancelled during download — report as CANCELLED, not FAILED
        result_queue.put({
            "job_id":    job_id,
            "status":    "CANCELLED",
        })
    except Exception as exc:
        result_queue.put({
            "job_id":    job_id,
            "status":    "FAILED",
            "error":     str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        pipe_obj = None
        gc.collect()
        if snapshot.get("should_unload", True):
            if model_cache is not None:
                release_model_cache(model_cache)
            clear_cuda_cache()
        _progress_store.pop(job_id, None)


# ---------------------------------------------------------------------------
# Operator: Add to Queue
# ---------------------------------------------------------------------------

class SEQUENCER_OT_add_to_queue(Operator):
    """Add current settings as a job in the render queue"""

    bl_idname = "sequencer.add_to_queue"
    bl_label = "Add to Queue"
    bl_description = "Enqueue the current generation settings without blocking the UI"
    bl_options = {"REGISTER"}

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_named_strip_path(scene, strip_name: str) -> str:
        """Resolve a named strip to its absolute file path for queue snapshotting."""
        if not strip_name:
            return ""
        from ..utils.helpers import find_strip_by_name, get_strip_path
        strip = find_strip_by_name(scene, strip_name)
        if strip is None:
            return ""
        path = get_strip_path(strip)
        return bpy.path.abspath(path) if path else ""

    @staticmethod
    def _render_named_strip_image(context, scene, strip_name: str) -> str:
        """Resolve a named strip to an image file path suitable for model input.

        Mirrors load_strip_as_pil()'s decision tree but runs at queue-add time
        (main thread, bpy available) and returns a file path instead of a PIL image:
          - IMAGE without transforms → raw source file (no letterboxing)
          - MOVIE                    → raw source file (load_first_frame will seek)
          - IMAGE with transforms / SCENE / META / MASK / COLOR → VSE render to PNG
        """
        if not strip_name:
            return ""
        from ..utils.helpers import find_strip_by_name, get_strip_path, render_strip_to_path
        strip = find_strip_by_name(scene, strip_name)
        if strip is None:
            return ""

        if strip.type == "IMAGE":
            try:
                tx = strip.transform
                has_transform = (
                    tx.scale_x != 1.0 or tx.scale_y != 1.0
                    or tx.offset_x != 0.0 or tx.offset_y != 0.0
                    or getattr(strip, "use_crop", False)
                )
            except Exception:
                has_transform = False
            if not has_transform:
                path = get_strip_path(strip)
                return bpy.path.abspath(path) if path else ""

        if strip.type == "MOVIE":
            path = get_strip_path(strip)
            if path:
                return bpy.path.abspath(path)

        # Complex strips (META, SCENE, IMAGE with transforms, etc.) — render through VSE
        path = render_strip_to_path(context, strip, image_output=True)
        return path or ""

    @staticmethod
    def _paths_from_strip(strip):
        """Extract (image_path, movie_path, sound_path) from a VSE strip.

        Returns empty strings for strip types that carry no media (TEXT, COLOR,
        ADJUSTMENT, META, effect strips, …).  The caller uses these to drive
        mode detection: empty paths → txt2* mode, but frame alignment still
        follows the strip's timing.
        """
        image_path = movie_path = sound_path = ""
        if strip is None:
            return image_path, movie_path, sound_path
        if strip.type == "IMAGE":
            dirname = os.path.dirname(bpy.path.abspath(strip.directory))
            try:
                fname = strip.elements[0].filename
            except (IndexError, AttributeError):
                fname = ""
            image_path = os.path.join(dirname, fname) if fname else ""
        elif strip.type == "MOVIE":
            movie_path = bpy.path.abspath(strip.filepath)
        elif strip.type == "SOUND":
            # Assign to sound_path, NOT movie_path — otherwise _detect_mode
            # sees has_vid=True and incorrectly routes to vid2vid / img2vid.
            try:
                sound_path = bpy.path.abspath(strip.sound.filepath)
            except AttributeError:
                sound_path = ""
        elif strip.type == "META":
            # Decompose META children into typed paths so the job gets real
            # media references rather than empty strings.
            for child in strip.strips:
                if child.type == "IMAGE" and not image_path:
                    try:
                        dirname = os.path.dirname(bpy.path.abspath(child.directory))
                        fname = child.elements[0].filename
                        candidate = os.path.join(dirname, fname)
                        if fname and os.path.isfile(candidate):
                            image_path = candidate
                    except (IndexError, AttributeError):
                        pass
                elif child.type == "MOVIE" and not movie_path:
                    try:
                        mp = bpy.path.abspath(child.filepath)
                        if os.path.isfile(mp):
                            movie_path = mp
                    except Exception:
                        pass
                elif child.type == "SOUND" and not sound_path:
                    try:
                        sp = bpy.path.abspath(child.sound.filepath)
                        if os.path.isfile(sp):
                            sound_path = sp
                    except AttributeError:
                        pass
        # TEXT, COLOR, ADJUSTMENT, SCENE, effect strips, etc.:
        # return empty strings → mode detection falls through to txt2* path.
        return image_path, movie_path, sound_path

    @staticmethod
    def _detect_mode(otype, image_path, movie_path, inpaint_strip):
        has_img = bool(image_path)
        has_vid = bool(movie_path)
        if otype == "image":
            do_inpaint = bool(inpaint_strip) and has_img
            return "inpaint" if do_inpaint else ("img2img" if (has_img or has_vid) else "txt2img")
        elif otype == "movie":
            return "vid2vid" if has_vid else ("img2vid" if has_img else "txt2vid")
        elif otype == "audio":
            return "txt2audio"
        return "txt2text"

    def execute(self, context):
        scene = context.scene
        prefs = context.preferences.addons[ADDON_ID].preferences
        otype = scene.generatorai_typeselect

        # In Blender 5.x the workspace.sequencer_scene can differ from the
        # active scene.  Record it now so the strip lands in the right place.
        _ws_seq_scene = getattr(context.workspace, "sequencer_scene", None)
        sequencer_scene_name = (_ws_seq_scene or scene).name

        if not scene.sequence_editor:
            scene.sequence_editor_create()

        model_card = {
            "image": prefs.image_model_card,
            "movie": prefs.movie_model_card,
            "audio": prefs.audio_model_card,
            "text":  prefs.text_model_card,
        }.get(otype, "")

        if not model_card:
            self.report({"ERROR"}, "No model selected.")
            return {"CANCELLED"}

        # ---- Shared settings (same for every job in this batch) ----------
        x          = closest_divisible_32(scene.generate_movie_x)
        y          = closest_divisible_32(scene.generate_movie_y)
        # raw_frames: -1 means "match input strip duration"; 0/positive = explicit
        raw_frames = scene.generate_movie_frames
        audio_dur  = float(getattr(scene, "audio_length_in_f", 80))

        styled     = style_prompt(scene.generate_movie_prompt)
        prompt     = styled[0]

        # Let the plugin bake any scene-specific state into the prompt now so
        # the job snapshot is self-contained (e.g. JoyAI spatial mode).
        from ..models import get_plugin as _get_plugin_for_prompt
        _pi = _get_plugin_for_prompt(model_card)
        if _pi is not None and hasattr(_pi, "_build_prompt"):
            prompt = _pi._build_prompt(scene, prompt)

        neg_prompt = (
            scene.generate_movie_negative_prompt
            + ", " + styled[1]
        )
        lora_json  = json.dumps([
            {"name": f.name, "weight": getattr(f, "weight_value", 1.0), "enabled": f.enabled}
            for f in getattr(scene, "lora_files", [])
        ])
        # 'frames' and 'audio_length' are per-strip — omitted from common
        common = dict(
            output_type  = otype,
            model_card   = model_card,
            prompt       = prompt,
            neg_prompt   = neg_prompt,
            steps        = scene.movie_num_inference_steps,
            guidance     = scene.movie_num_guidance,
            use_random   = scene.movie_use_random,
            width        = x,
            height       = y,
            image_power  = scene.image_power,
            use_lcm           = getattr(scene, "use_lcm", False),
            refine_sd         = getattr(scene, "refine_sd", False),
            adetailer         = getattr(scene, "adetailer", False),
            aurasr            = getattr(scene, "aurasr", False),
            remove_silence    = getattr(scene, "remove_silence", True),
            audio_speed_tts   = getattr(scene, "audio_speed_tts", 1.0),
            chat_exaggeration = getattr(scene, "chat_exaggeration", 0.5),
            chat_pace         = getattr(scene, "chat_pace", 0.5),
            chat_temperature  = getattr(scene, "chat_temperature", 0.8),
            fps               = round(scene.render.fps / scene.render.fps_base, 3),
            music_bpm         = getattr(scene, "music_bpm", 0),
            music_lyrics      = getattr(scene, "music_lyrics", ""),
            music_key_scale   = getattr(scene, "music_key_scale", ""),
            music_time_signature = getattr(scene, "music_time_signature", ""),
            audio_path        = bpy.path.abspath(getattr(scene, "audio_path", "") or ""),
            audio_text        = getattr(scene, "audio_text", ""),
            hugginface_token  = getattr(prefs, "hugginface_token", ""),
            local_files_only  = getattr(prefs, "local_files_only", False),
            generator_ai      = getattr(prefs, "generator_ai", "") or os.path.join(
                bpy.utils.user_resource("DATAFILES"), "Pallaidium Media"
            ),
            hf_cache_dir      = getattr(prefs, "hf_cache_dir", ""),
            lora_files_json      = lora_json,
            lora_folder          = bpy.path.abspath(getattr(scene, "lora_folder", "") or ""),
            sequencer_scene_name = sequencer_scene_name,
            joyimage_spatial_mode = getattr(scene, "joyimage_spatial_mode", "general"),
            joyimage_object       = getattr(scene, "joyimage_object", "object"),
            joyimage_rotate_view  = getattr(scene, "joyimage_rotate_view", "front"),
            joyimage_yaw          = getattr(scene, "joyimage_yaw", 0.0),
            joyimage_pitch        = getattr(scene, "joyimage_pitch", 0.0),
            joyimage_zoom         = getattr(scene, "joyimage_zoom", "unchanged"),
            kontext_strip_1_path  = self._resolve_named_strip_path(scene, getattr(scene, "kontext_strip_1", "")),
            inpaint_mask_path     = self._resolve_named_strip_path(scene, getattr(scene, "inpaint_selected_strip", "")),
            qwen_strip_1_path     = self._render_named_strip_image(context, scene, getattr(scene, "qwen_strip_1", "")),
            qwen_strip_2_path     = self._render_named_strip_image(context, scene, getattr(scene, "qwen_strip_2", "")),
            qwen_strip_3_path     = self._render_named_strip_image(context, scene, getattr(scene, "qwen_strip_3", "")),
            klein_strip_1_path    = self._render_named_strip_image(context, scene, getattr(scene, "klein_strip_1", "")),
            klein_strip_2_path    = self._render_named_strip_image(context, scene, getattr(scene, "klein_strip_2", "")),
            klein_strip_3_path    = self._render_named_strip_image(context, scene, getattr(scene, "klein_strip_3", "")),
            stem_split_model  = getattr(scene, "stem_split_model",  "htdemucs_ft"),
            stem_split_vocals = getattr(scene, "stem_split_vocals", True),
            stem_split_drums  = getattr(scene, "stem_split_drums",  True),
            stem_split_bass   = getattr(scene, "stem_split_bass",   True),
            stem_split_other  = getattr(scene, "stem_split_other",  True),
            stem_split_guitar  = getattr(scene, "stem_split_guitar",  False),
            stem_split_piano   = getattr(scene, "stem_split_piano",   False),
            omnivoice_instruct    = getattr(scene, "omnivoice_instruct",    ""),
            omnivoice_language    = getattr(scene, "omnivoice_language",    ""),
            omnivoice_preprocess  = getattr(scene, "omnivoice_preprocess",  True),
            omnivoice_denoise     = getattr(scene, "omnivoice_denoise",     True),
            omnivoice_postprocess = getattr(scene, "omnivoice_postprocess", True),
            klein_schematic_mode   = getattr(scene, "klein_schematic_mode",   "DEPTH"),
            klein_schematic_target = getattr(scene, "klein_schematic_target", "person"),
        )

        # ---- Decide which strips to iterate over -------------------------
        # Only content-bearing strip types are meaningful as queue inputs.
        # Effect strips (SPEED, GLOW, GAUSSIAN_BLUR, TRANSFORM, WIPE, …) are
        # automatically selected alongside their parent strip in Blender, which
        # would create spurious extra jobs — exclude them with a whitelist.
        _CONTENT_TYPES = {
            "IMAGE", "MOVIE", "SOUND", "TEXT",
            "COLOR", "META", "SCENE", "MOVIECLIP", "MASK", "ADJUSTMENT",
        }
        input_mode = scene.input_strips
        # context.selected_strips is the authoritative source in Blender 5.x —
        # it works correctly from the N-panel (same as strip_to_generatorAI).
        # Fall back to se.strips + s.select for non-VSE invocations (keybinds,
        # Python calls) where context may not carry the sequence editor state.
        se = scene.sequence_editor
        _ctx_sel = list(context.selected_strips) if context.selected_strips else []
        if _ctx_sel:
            selected = sorted(
                (s for s in _ctx_sel if s.type in _CONTENT_TYPES),
                key=lambda s: s.frame_final_start,
            )
        else:
            # Fallback: iterate top-level strips and check s.select directly.
            top_level = list(se.strips) if se else []
            selected = sorted(
                (s for s in top_level if s.select and s.type in _CONTENT_TYPES),
                key=lambda s: s.frame_final_start,
            )
        if input_mode == "input_strips" and not selected:
            self.report({"WARNING"}, "No strips selected. Select one or more strips before adding to the queue.")
            return {"CANCELLED"}
        strip_list = selected if (input_mode == "input_strips" and selected) else [None]

        inpaint_strip = getattr(scene, "inpaint_selected_strip", "")
        batch_count = max(1, getattr(scene, "movie_num_batch", 1))
        added = 0

        for strip in strip_list:
            if strip is not None:
                image_path, movie_path, sound_path = self._paths_from_strip(strip)

                # Align output to the input strip's in-point
                strip_frame_start = strip.frame_final_start
                # -1 (or any negative) = match the input strip's length
                strip_dur = strip.frame_final_duration
                gen_frames = strip_dur if raw_frames < 0 else max(1, abs(raw_frames))
                if otype == "audio":
                    # audio_dur < 0 is the -1 sentinel: match the input strip.
                    # Also match when the plugin requires a strip (e.g. Stem Splitter).
                    _pi_req = _pi is not None and getattr(_pi, "requires_input_strip", False)
                    insert_dur = strip_dur if (audio_dur < 0 or _pi_req) else max(1, int(audio_dur))
                else:
                    insert_dur = gen_frames
                # For SOUND strips generating audio: use the strip file as the
                # audio reference (overrides scene-level audio_path in common).
                strip_audio_path = sound_path if (otype == "audio" and sound_path) else ""
            else:
                image_path = bpy.path.abspath(getattr(scene, "image_path", "") or "")
                movie_path = bpy.path.abspath(getattr(scene, "movie_path", "") or "")
                sound_path = bpy.path.abspath(getattr(scene, "sound_path", "") or "")
                strip_audio_path = ""

                # Negative raw_frames = "auto" sentinel. In prompt mode there is no
                # input strip to derive duration from, so fall back to 100 frames.
                if raw_frames < 0:
                    gen_frames = 100
                elif raw_frames > 0:
                    gen_frames = raw_frames
                else:
                    gen_frames = 1
                if otype == "audio":
                    # audio_dur < 0 is the -1 sentinel: no strip to match, use default
                    insert_dur = 80 if audio_dur < 0 else max(1, int(audio_dur))
                else:
                    insert_dur = gen_frames

            mode = self._detect_mode(otype, image_path, movie_path,
                                     inpaint_strip if strip is None else "")

            # For strip mode: find the output channel once, then advance the
            # frame cursor after each batch copy so they form a sequence.
            if strip is not None:
                _batch_frame = strip_frame_start
                _batch_ch    = _find_free_channel(
                    scene,
                    strip_frame_start,
                    strip_frame_start + max(1, insert_dur),
                    strip.channel + 1,
                )

            for _batch_i in range(batch_count):
                seed = (
                    random.randint(-2147483647, 2147483647)
                    if scene.movie_use_random else scene.movie_num_seed
                )

                if strip is not None:
                    # Strip mode: batch copies placed consecutively in time,
                    # all in the same channel directly above the input strip.
                    frame_start = _batch_frame
                    frame_end   = frame_start + max(1, insert_dur)
                    channel     = _batch_ch
                    _batch_frame = frame_end  # advance cursor for next copy
                else:
                    # Prompt mode: each batch copy is placed consecutively in time.
                    frame_start = scene.frame_current
                    frame_end   = frame_start + max(1, insert_dur)
                    channel     = find_first_empty_channel(frame_start, frame_end)

                job = scene.render_queue.add()
                job.job_id   = f"{id(job)}_{random.randint(0, 999999)}"
                job.status   = "PENDING"
                job.progress = 0.0
                job.mode     = mode
                job.seed     = seed
                job.frames   = gen_frames
                job.audio_length = insert_dur

                job.insert_frame_start = frame_start
                job.insert_frame_end   = frame_end
                job.insert_channel     = channel
                job.insert_duration    = insert_dur

                job.image_path = image_path
                job.movie_path = movie_path
                job.sound_path = sound_path

                for attr, val in common.items():
                    setattr(job, attr, val)

                # For TEXT strips: prepend the strip's text to the prompt,
                # matching the non-queue behaviour: strip.text + ", " + base_prompt
                if strip is not None and strip.type == "TEXT":
                    strip_text = getattr(strip, "text", "").strip()
                    if strip_text:
                        job.prompt = (strip_text + ", " + job.prompt) if job.prompt else strip_text

                # For META strips: prepend TEXT children to the prompt
                if strip is not None and strip.type == "META":
                    meta_texts = [
                        c.text for c in strip.strips
                        if c.type == "TEXT" and getattr(c, "text", "").strip()
                    ]
                    if meta_texts:
                        meta_text = ", ".join(meta_texts)
                        job.prompt = (meta_text + ", " + job.prompt) if job.prompt else meta_text

                # Override audio_path from a SOUND strip (wins over scene-level value)
                if strip_audio_path:
                    job.audio_path = strip_audio_path

                # For strip-input plugins (e.g. Stem Splitter) use the input
                # filename as the queue job name instead of the text prompt.
                if strip is not None and _pi is not None and getattr(_pi, "requires_input_strip", False):
                    input_path = sound_path or movie_path or ""
                    input_name = os.path.splitext(os.path.basename(input_path))[0]
                    if input_name:
                        job.prompt = input_name

                added += 1

        if added == 0:
            self.report({"WARNING"}, "Nothing to queue.")
            return {"CANCELLED"}

        label = f"{added} job(s)" if added > 1 else (prompt[:40] + ("…" if len(prompt) > 40 else ""))
        self.report({"INFO"}, f"Queued: {label}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Timer-based queue processing  (bpy.app.timers — main thread, no context needed)
# ---------------------------------------------------------------------------

_TIMER_INTERVAL = 0.2


def _queue_get_scene():
    """Return the scene being processed (direct reference — no bpy.data lookup)."""
    return _active_scene


def _queue_start_job(scene, job) -> None:
    global _worker_thread
    print(f"[Pallaidium Queue] Starting job: {job.prompt[:60]!r}")
    job.status   = "RUNNING"
    job.progress = 0.0

    # Keep VRAM warm if the next pending job uses the same model
    next_job = next((j for j in scene.render_queue if j.status == "PENDING"), None)
    job.should_unload = (next_job is None or next_job.model_card != job.model_card)

    snapshot = {k: getattr(job, k) for k in (
        "job_id", "output_type", "model_card", "prompt", "neg_prompt",
        "steps", "guidance", "seed", "use_random", "width", "height",
        "frames", "image_power", "mode", "use_lcm", "refine_sd",
        "adetailer", "aurasr", "remove_silence", "audio_length",
        "audio_speed_tts", "chat_exaggeration", "chat_pace",
        "chat_temperature", "fps", "music_bpm", "music_lyrics",
        "music_key_scale", "music_time_signature",
        "image_path", "movie_path", "sound_path", "audio_path",
        "audio_text", "hugginface_token", "local_files_only",
        "generator_ai", "hf_cache_dir", "lora_files_json", "lora_folder",
        "insert_frame_start", "insert_frame_end",
        "insert_channel", "insert_duration",
        "sequencer_scene_name",
        "should_unload",
        "omnivoice_instruct", "omnivoice_language",
        "omnivoice_preprocess", "omnivoice_denoise", "omnivoice_postprocess",
        "stem_split_model", "stem_split_vocals", "stem_split_drums",
        "stem_split_bass", "stem_split_other", "stem_split_guitar", "stem_split_piano",
        "qwen_strip_1_path", "qwen_strip_2_path", "qwen_strip_3_path",
        "klein_strip_1_path", "klein_strip_2_path", "klein_strip_3_path",
        "klein_schematic_mode", "klein_schematic_target",
    )}
    _cancel_event.clear()
    _worker_thread = threading.Thread(
        target=_run_job,
        args=(snapshot, _result_queue, _cancel_event, _progress_store),
        daemon=True,
    )
    _worker_thread.start()


def _run_job_main_thread(scene, job) -> None:
    """Execute a requires_main_thread plugin synchronously on the main thread.

    Used by plugins (hviske_subtitles, marlin_video_captions) that must call
    bpy directly inside generate().  The UI will be unresponsive for the
    duration — same behaviour as interactive generation.
    """
    from ..models import get_plugin
    from ..models.base import ModelInputs
    from ..operators.main_ops import (
        _pallaidium_model_cache,
        _pallaidium_movie_model_cache,
        _pallaidium_audio_model_cache,
        _pallaidium_text_model_cache,
    )

    job_id = job.job_id
    print(f"[Pallaidium Queue] Starting main-thread job: {job.prompt[:60]!r}")
    job.status   = "RUNNING"
    job.progress = 0.0
    job.phase    = "Loading model"

    _progress_store[job_id] = {"progress": 0.0, "phase": "Loading model", "step": 0, "total": 0}

    # Determine whether to unload the model after this job
    following = next(
        (j for j in scene.render_queue
         if j.status == "PENDING" and j.job_id != job_id),
        None,
    )
    should_unload = following is None or following.model_card != job.model_card

    model_cache = None
    try:
        plugin = get_plugin(job.model_card)
        if plugin is None:
            raise RuntimeError(f"Plugin not found: {job.model_card!r}")

        otype = job.output_type
        model_cache = {
            "image": _pallaidium_model_cache,
            "movie": _pallaidium_movie_model_cache,
            "audio": _pallaidium_audio_model_cache,
            "text":  _pallaidium_text_model_cache,
        }.get(otype)

        mode = job.mode
        prefs = bpy.context.preferences.addons[ADDON_ID].preferences

        if prefs.hf_cache_dir:
            os.environ["HF_HUB_CACHE"] = prefs.hf_cache_dir

        # Load model (or reuse cache)
        _schematic_mode_mt = getattr(job, "klein_schematic_mode", "")
        _cache_skip = {"last_model_card", "last_mode", "last_schematic_mode"}
        cache_hit = (
            model_cache is not None
            and model_cache.get("last_model_card") == job.model_card
            and model_cache.get("last_mode") == mode
            and model_cache.get("last_schematic_mode", "") == _schematic_mode_mt
            and any(v is not None for k, v in model_cache.items() if k not in _cache_skip)
        )
        if cache_hit:
            pipe_obj = model_cache
            print(f"[Queue] Reusing cached model: {job.model_card} ({mode}) schematic={_schematic_mode_mt or '-'}")
        else:
            if (model_cache is not None
                    and (model_cache.get("last_model_card") != job.model_card
                         or model_cache.get("last_mode") != mode
                         or model_cache.get("last_schematic_mode", "") != _schematic_mode_mt)):
                release_model_cache(model_cache)
            clear_cuda_cache()
            show_system_console(True)
            set_system_console_topmost(True)

            _progress_store[job_id] = {
                "progress": 0.0,
                "phase":    "Loading model",
                "step":     0,
                "total":    0,
            }

            import tqdm.std as _tqdm_std_mt

            _active_bars_mt: dict = {}
            _orig_tqdm_init_mt   = _tqdm_std_mt.tqdm.__init__
            _orig_tqdm_update_mt = _tqdm_std_mt.tqdm.update

            def _patched_tqdm_init_mt(tqdm_self, *a, **kw):
                _orig_tqdm_init_mt(tqdm_self, *a, **kw)
                if not getattr(tqdm_self, "disable", False):
                    _active_bars_mt[id(tqdm_self)] = [tqdm_self.n or 0, tqdm_self.total or 0]

            def _patched_tqdm_update_mt(tqdm_self, n=1):
                result = _orig_tqdm_update_mt(tqdm_self, n)
                entry = _active_bars_mt.get(id(tqdm_self))
                if entry is not None:
                    entry[0] = tqdm_self.n or 0
                    entry[1] = tqdm_self.total or 0
                total_b = sum(v[1] for v in _active_bars_mt.values() if v[1] > 0)
                done_b  = sum(v[0] for v in _active_bars_mt.values())
                dl_frac = (done_b / total_b) if total_b > 0 else 0.0
                _progress_store[job_id] = {
                    "progress": dl_frac,
                    "phase":    "Downloading model",
                    "step":     max(0, int(done_b  / 1_048_576)),
                    "total":    max(1, int(total_b / 1_048_576)),
                }
                job.progress = dl_frac
                job.phase    = "Downloading model"
                return result

            _tqdm_std_mt.tqdm.__init__ = _patched_tqdm_init_mt
            _tqdm_std_mt.tqdm.update   = _patched_tqdm_update_mt

            try:
                loaded = plugin.load(
                    prefs, scene,
                    mode=mode,
                    enabled_items=[],
                    use_lcm=job.use_lcm,
                    use_refine=job.refine_sd,
                    ip_adapter_face_folder="",
                    ip_adapter_style_folder="",
                    local_files_only=job.local_files_only,
                )
            finally:
                _tqdm_std_mt.tqdm.__init__ = _orig_tqdm_init_mt
                _tqdm_std_mt.tqdm.update   = _orig_tqdm_update_mt
                _active_bars_mt.clear()

            if model_cache is not None and isinstance(loaded, dict):
                model_cache.update(loaded)
                model_cache["last_model_card"] = job.model_card
                model_cache["last_mode"] = mode
                model_cache["last_schematic_mode"] = _schematic_mode_mt
                pipe_obj = model_cache
            else:
                pipe_obj = loaded

        _progress_store[job_id] = {"progress": 0.10, "phase": "Preparing", "step": 0, "total": 0}

        def _phase_fn(label: str) -> None:
            store = _progress_store.get(job_id)
            if store is not None:
                store["phase"] = label
            job.phase = label  # direct update — we are on the main thread

        def _progress_fn(step: int, total: int) -> None:
            cur_phase = _progress_store.get(job_id, {}).get("phase", "Generating")
            _progress_store[job_id] = {
                "progress": 0.10 + 0.89 * (step / max(1, total)),
                "phase":    cur_phase,
                "step":     step,
                "total":    total,
            }

        inputs = ModelInputs(
            prompt              = job.prompt,
            neg_prompt          = job.neg_prompt,
            mode                = mode,
            steps               = job.steps,
            guidance            = job.guidance,
            strength            = job.image_power,
            seed                = job.seed,
            audio_ref           = job.audio_path or job.sound_path or None,
            video_path          = job.movie_path or None,
            insert_channel      = job.insert_channel,
            insert_frame_start  = job.insert_frame_start,
            progress_fn         = _progress_fn,
            phase_fn            = _phase_fn,
        )

        _result = plugin.generate(pipe_obj, inputs, scene, prefs)
        del _result  # None for bpy-inserting plugins; discarded intentionally

        # plugin handled its own VSE strip creation
        job.status   = "COMPLETED"
        job.progress = 1.0
        job.phase    = "Done"
        _progress_store.pop(job_id, None)

    except Exception as exc:
        job.status          = "FAILED"
        job.progress        = 0.0
        job.error_message   = str(exc)[:200]
        job.error_traceback = traceback.format_exc()[:4000]
        print("=== Pallaidium Queue Error (main-thread job) ===")
        print(traceback.format_exc())
        print("================================================")
        _progress_store.pop(job_id, None)
    finally:
        if should_unload and model_cache is not None:
            release_model_cache(model_cache)
            clear_cuda_cache()


def _find_vse_override():
    """Return (window, area, region) for the first SEQUENCE_EDITOR area, or (None, None, None).

    Including the WINDOW region is required for temp_override in timer callbacks
    on Blender 5.x — without it, strip-creation calls can silently fail.
    """
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "SEQUENCE_EDITOR":
                    for region in area.regions:
                        if region.type == "WINDOW":
                            return window, area, region
    except Exception:
        pass
    return None, None, None


def _queue_insert_strip(scene, result: dict) -> None:
    """Insert the generated strip into the VSE timeline (main thread).

    Uses the sequencer_scene stored at job-add time so the strip lands in the
    correct scene even when workspace.sequencer_scene differs from the active scene.
    Tries a direct call first; falls back to temp_override with the WINDOW
    region for timer-callback invocations where Blender 5.x requires it.
    """
    out_path     = result.get("output_path", "")
    otype        = result.get("output_type", "")
    text_content = result.get("text_content", "")

    # Multi-stem output from StemSplitterPlugin: insert one SOUND strip per stem.
    _MULTI_PREFIX = "MULTI_STEM:"
    if out_path.startswith(_MULTI_PREFIX):
        import json as _json
        stem_paths = _json.loads(out_path[len(_MULTI_PREFIX):])
        seq_scene_name = result.get("sequencer_scene_name", "")
        seq_scene = (bpy.data.scenes.get(seq_scene_name) if seq_scene_name else None) or scene
        if not seq_scene.sequence_editor:
            seq_scene.sequence_editor_create()
        ed          = seq_scene.sequence_editor
        frame_start = result["frame_start"]
        frame_end   = result["frame_end"]
        base_ch     = result["channel"]
        next_ch     = base_ch
        for stem_name, stem_path in stem_paths.items():
            if not os.path.isfile(stem_path):
                print(f"[Queue] Stem file not found: {stem_path!r}")
                continue
            ch = _find_free_channel(seq_scene, frame_start, frame_end, next_ch)
            ch = max(ch, next_ch)
            next_ch = ch + 1
            # Derive input filename from the stem file: "{stem}_{orig_base}.wav"
            file_stem = os.path.splitext(os.path.basename(stem_path))[0]
            orig_base = file_stem[len(stem_name) + 1:] if file_stem.startswith(stem_name + "_") else file_stem
            strip_name = f"{stem_name} | {orig_base}"
            snd = ed.strips.new_sound(
                name=strip_name,
                filepath=stem_path,
                channel=ch,
                frame_start=frame_start,
            )
            snd.frame_final_duration = (result["frame_end"] - result["frame_start"]) - 1
            print(f"[Queue] Inserted stem '{stem_name}' on channel {ch}")
        return

    # For text-type jobs the content may be carried directly; no file required.
    if otype == "text" and text_content:
        pass  # proceed to strip creation
    elif otype == "text" and not text_content and not out_path:
        print("[Queue] Text model produced no output — skipping strip insertion.")
        return
    elif not out_path or not os.path.isfile(out_path):
        print(f"[Queue] Output file not found: {out_path!r}")
        return

    # Resolve the target scene — use the sequencer_scene recorded at add-time.
    seq_scene_name = result.get("sequencer_scene_name", "")
    seq_scene = (bpy.data.scenes.get(seq_scene_name) if seq_scene_name else None) or scene

    if not seq_scene.sequence_editor:
        seq_scene.sequence_editor_create()

    frame_start = result["frame_start"]
    frame_end   = result["frame_end"]
    channel     = _find_free_channel(seq_scene, frame_start, frame_end, result["channel"])
    duration    = result["duration"]
    name        = f"{result['seed']}_{result['prompt'][:30]}"
    ed          = seq_scene.sequence_editor

    def _do_insert():
        nonlocal new_strip
        if otype == "image":
            new_strip = ed.strips.new_image(
                name=name,
                filepath=out_path,
                frame_start=frame_start,
                channel=channel,
                fit_method="FIT",
            )
            new_strip.frame_final_duration = duration
            new_strip.use_proxy = True

        elif otype == "movie":
            new_strip = ed.strips.new_movie(
                name=name,
                filepath=out_path,
                frame_start=frame_start,
                channel=channel,
                fit_method="FIT",
            )
            if channel > 1:
                snd_ch = _find_free_channel(seq_scene, frame_start, frame_end, channel - 1)
                snd_strip = ed.strips.new_sound(
                    name=name,
                    filepath=out_path,
                    channel=snd_ch,
                    frame_start=frame_start,
                )
                snd_strip.select = False

        elif otype == "audio":
            new_strip = ed.strips.new_sound(
                name=result["prompt"][:50],
                filepath=out_path,
                channel=channel,
                frame_start=frame_start,
            )

        elif otype == "text":
            # Build the text body: prefer the in-memory content; fall back to
            # reading the .txt file saved by the worker.
            text_body = text_content or ""
            if not text_body and out_path and out_path.endswith(".txt"):
                try:
                    with open(out_path, encoding="utf-8") as _tf:
                        text_body = _tf.read()
                except Exception:
                    pass
            if text_body:
                new_strip = ed.strips.new_effect(
                    name=text_body[:63],
                    type="TEXT",
                    frame_start=frame_start,
                    length=max(1, frame_end - frame_start),
                    channel=channel,
                )
                new_strip.text = text_body

    new_strip = None
    try:
        _do_insert()
    except Exception as exc1:
        # Direct call failed (common from timer callbacks) — retry with override.
        new_strip = None
        window, area, region = _find_vse_override()
        if window is None:
            print(f"[Queue] No SEQUENCE_EDITOR area found ({exc1}) — cannot insert strip.")
            return
        try:
            with bpy.context.temp_override(window=window, area=area, region=region):
                _do_insert()
        except Exception as exc2:
            print(f"[Queue] Strip insertion failed: {exc2}")
            traceback.print_exc()
            return

    if new_strip is not None:
        try:
            set_ai_metadata_from_dict(new_strip, {
                "model":           result.get("model_card", ""),
                "mode":            result.get("mode", ""),
                "prompt":          result.get("prompt", ""),
                "negative_prompt": result.get("neg_prompt", ""),
                "seed":            result.get("seed", 0),
                "width":           result.get("width", 0),
                "height":          result.get("height", 0),
                "frames":          result.get("frames", 0),
                "steps":           result.get("steps", 0),
                "guidance":        result.get("guidance", 0.0),
            })
            new_strip.select = False
            ed.active_strip = new_strip
        except Exception:
            pass


def _queue_stop() -> None:
    """Unregister the processing timer."""
    if bpy.app.timers.is_registered(_queue_tick):
        bpy.app.timers.unregister(_queue_tick)


def _queue_tick() -> float | None:
    """Called by bpy.app.timers every 0.2 s on the main thread.

    Returns the next interval (float) to keep running, or None to stop.
    """
    global _worker_thread

    try:
        scene = _queue_get_scene()
        if scene is None:
            _queue_stop()
            return None

        # ---- Drain one result from the worker ---------------------------
        try:
            result = _result_queue.get_nowait()
        except _stdqueue.Empty:
            result = None

        if result:
            job = _find_job(scene, result["job_id"])
            status = result["status"]

            if job:
                if job.status in ("CANCELLING", "CANCELLED"):
                    job.status   = "CANCELLED"
                    job.progress = 0.0
                elif status == "COMPLETED":
                    job.output_path = result.get("output_path", "")
                    job.status      = "COMPLETED"
                    job.progress    = 1.0
                    _queue_insert_strip(scene, result)
                elif status == "FAILED":
                    job.status          = "FAILED"
                    job.progress        = 0.0
                    job.error_message   = result.get("error", "Unknown error")[:200]
                    job.error_traceback = result.get("traceback", "")[:4000]
                    print("=== Pallaidium Queue Error ===")
                    print(result.get("traceback", ""))
                    print("==============================")
                else:
                    job.status = "CANCELLED"

            # Do NOT set _worker_thread = None here. The thread may still be
            # executing its finally block (release_model_cache, clear_cuda_cache).
            # Setting it to None too early would let the next job start while the
            # previous thread is still tearing down the model cache, causing a race
            # where the new job loads the model and the old thread then clears it.
            # The is_alive() check below is the correct gate for starting the next job.
            _cancel_event.clear()

        # ---- Sync progress for running job ------------------------------
        running = _find_running_job(scene)
        if running and running.job_id in _progress_store:
            try:
                store = _progress_store[running.job_id]
                running.progress     = store["progress"]
                running.phase        = store["phase"]
                running.current_step = store["step"]
                running.total_steps  = store["total"]
            except Exception:
                pass  # stale or unexpected store format; skip this tick

        # ---- Force panel redraw so slider animates without mouse movement ---
        try:
            for _win in bpy.context.window_manager.windows:
                for _area in _win.screen.areas:
                    if _area.type == "SEQUENCE_EDITOR":
                        _area.tag_redraw()
        except Exception:
            pass

        # ---- Start next job if worker is idle ---------------------------
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = None
            if running and running.status == "RUNNING":
                running.status        = "FAILED"
                running.error_message = "Worker thread died unexpectedly."
                running = None
            if not _queue_paused:
                next_job = _find_next_pending(scene)
                if next_job:
                    from ..models import get_plugin as _get_plugin
                    _next_plugin = _get_plugin(next_job.model_card)
                    if _next_plugin and getattr(_next_plugin, "requires_main_thread", False):
                        # Run synchronously on the main thread (UI will freeze).
                        _run_job_main_thread(scene, next_job)
                    else:
                        _queue_start_job(scene, next_job)

        # ---- Stop conditions --------------------------------------------
        still_active = any(
            j.status in ("PENDING", "RUNNING", "CANCELLING")
            for j in scene.render_queue
        )
        worker_idle = _worker_thread is None or not _worker_thread.is_alive()

        if (not still_active and worker_idle) or (_queue_paused and worker_idle):
            _queue_stop()
            return None

        return _TIMER_INTERVAL

    except BaseException:
        traceback.print_exc()
        _queue_stop()
        return None


# ---------------------------------------------------------------------------
# Operator: Queue runner  (simple execute — starts the app timer)
# ---------------------------------------------------------------------------

class SEQUENCER_OT_queue_runner(Operator):
    """Start processing the render queue"""

    bl_idname = "sequencer.queue_runner"
    bl_label  = "Start Queue"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _queue_paused, _active_scene
        _queue_paused = False
        _active_scene = context.scene   # direct reference — no bpy.data needed in timer
        # Always (re-)register to clear any stale timer from a previous crash
        if bpy.app.timers.is_registered(_queue_tick):
            bpy.app.timers.unregister(_queue_tick)
        bpy.app.timers.register(_queue_tick, first_interval=_TIMER_INTERVAL)
        print("[Pallaidium Queue] Started — scene:", context.scene.name)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Cancel a job
# ---------------------------------------------------------------------------

class SEQUENCER_OT_cancel_queue_job(Operator):
    """Cancel a pending or running job"""

    bl_idname = "sequencer.cancel_queue_job"
    bl_label  = "Cancel Job"

    job_id: StringProperty()

    def execute(self, context):
        job = _find_job(context.scene, self.job_id)
        if job is None:
            return {"CANCELLED"}
        if job.status == "PENDING":
            job.status = "CANCELLED"
        elif job.status == "RUNNING":
            # Signal the worker; it will stop at the next cancellation checkpoint
            _cancel_event.set()
            job.status = "CANCELLING"
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Remove a single job from the list
# ---------------------------------------------------------------------------

class SEQUENCER_OT_remove_queue_job(Operator):
    """Remove a finished or cancelled job from the list"""

    bl_idname = "sequencer.remove_queue_job"
    bl_label  = "Remove Job"

    job_id: StringProperty()

    def execute(self, context):
        queue = context.scene.render_queue
        for i, job in enumerate(queue):
            if job.job_id == self.job_id:
                if job.status == "RUNNING":
                    self.report({"WARNING"}, "Cannot remove a running job — cancel it first.")
                    return {"CANCELLED"}
                queue.remove(i)
                return {"FINISHED"}
        return {"CANCELLED"}


# ---------------------------------------------------------------------------
# Operator: Clear all finished jobs
# ---------------------------------------------------------------------------

class SEQUENCER_OT_clear_queue(Operator):
    """Remove all jobs that are not actively running"""

    bl_idname = "sequencer.clear_queue"
    bl_label  = "Clear Queue"

    def execute(self, context):
        queue = context.scene.render_queue
        indices = [
            i for i, j in enumerate(queue)
            if j.status != "RUNNING"
        ]
        for i in reversed(indices):
            queue.remove(i)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Show full error traceback in a popup
# ---------------------------------------------------------------------------

class SEQUENCER_OT_show_queue_error(Operator):
    """Show the full error traceback for a failed job"""

    bl_idname = "sequencer.show_queue_error"
    bl_label  = "Job Error Details"

    job_id: StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=620)

    def draw(self, context):
        job = _find_job(context.scene, self.job_id)
        layout = self.layout
        if job is None:
            layout.label(text="Job not found.")
            return

        col = layout.column(align=True)
        col.label(text=f"Failed job: {job.prompt[:60]}", icon="ERROR")
        col.label(text=f"Model: {job.model_card}")
        col.separator()

        box = col.box()
        box.label(text=f"Error: {job.error_message}", icon="CANCEL")
        col.separator()

        col.label(text="Traceback (for bug reports):")
        tb_box = col.box()
        for line in (job.error_traceback or "(no traceback)").splitlines():
            tb_box.label(text=line[:95])

        col.separator()
        col.label(text="Full traceback also printed to the System Console.", icon="INFO")

    def execute(self, context):
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator: Stop queue after current job (keeps pending jobs for later restart)
# ---------------------------------------------------------------------------

class SEQUENCER_OT_stop_queue(Operator):
    """Stop the queue after the current job finishes (pending jobs are kept)"""

    bl_idname = "sequencer.stop_queue"
    bl_label  = "Stop Queue"

    def execute(self, context):
        global _queue_paused
        _queue_paused = True
        # If no job is running right now, stop the timer immediately;
        # otherwise let the current job finish and the timer will stop itself.
        worker_idle = _worker_thread is None or not _worker_thread.is_alive()
        if worker_idle:
            _queue_stop()
            self.report({"INFO"}, "Queue stopped.")
        else:
            self.report({"INFO"}, "Queue will stop after the current job.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Exported list (used in __init__.py's classes tuple)
# ---------------------------------------------------------------------------

queue_classes = (
    RenderQueueJob,
    SEQUENCER_OT_add_to_queue,
    SEQUENCER_OT_queue_runner,
    SEQUENCER_OT_stop_queue,
    SEQUENCER_OT_cancel_queue_job,
    SEQUENCER_OT_remove_queue_job,
    SEQUENCER_OT_clear_queue,
    SEQUENCER_OT_show_queue_error,
)
