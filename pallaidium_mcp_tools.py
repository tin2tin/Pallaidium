"""
Pallaidium MCP Tools
====================
Helper functions for controlling Pallaidium Generative AI from a Claude agent
via the Blender MCP server (execute_blender_code).

Usage — load once per Blender session:

    exec(open(r"C:\...\pallaidium_generative_ai\pallaidium_mcp_tools.py").read())

Then call any function below, e.g.:

    print(list_image_models())
    set_model("image", "diffusers/FLUX.2-dev-bnb-4bit")
    result = generate_image("a sunset over golden mountains", width=1920, height=1080)
    print(result)

All generate_* functions return a dict:
    {"status": "ok", "strips": [{"name": ..., "type": ..., "channel": ...,
                                  "frame_start": ..., "frame_end": ...}]}
or on failure:
    {"status": "error", "message": "..."}
"""

from __future__ import annotations

import random
import sys

import bpy

# ---------------------------------------------------------------------------
# Resolve the Pallaidium package name dynamically (handles bl_ext.* prefix)
# ---------------------------------------------------------------------------

_ADDON_PKG: str | None = None
for _k in sys.modules:
    if _k.endswith("pallaidium_generative_ai") and "models" not in _k:
        _ADDON_PKG = _k
        break

if _ADDON_PKG is None:
    raise RuntimeError(
        "Pallaidium Generative AI extension not found in sys.modules. "
        "Make sure the extension is enabled in Blender."
    )

_ADDON_ID = _ADDON_PKG  # used for bpy.context.preferences.addons lookup


def _prefs():
    return bpy.context.preferences.addons[_ADDON_ID].preferences


def _scene():
    return bpy.context.scene


def _registry():
    mod = sys.modules.get(f"{_ADDON_PKG}.models")
    if mod is None:
        raise RuntimeError("Pallaidium models module not loaded.")
    return mod


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_models(media_type: str) -> list[dict]:
    """Return all registered plugins for media_type ('image'|'video'|'audio'|'text')."""
    reg = _registry()
    result = []
    for model_id, plugin in reg.PLUGIN_REGISTRY.items():
        if plugin.MODEL_TYPE == media_type:
            entry = {
                "model_id": model_id,
                "display_name": plugin.DISPLAY_NAME,
                "description": getattr(plugin, "DESCRIPTION", ""),
            }
            if media_type == "image":
                entry["supports_inpaint"] = getattr(plugin, "supports_inpaint", False)
                entry["supports_img2img"] = getattr(plugin, "supports_img2img", False)
            elif media_type == "video":
                entry["supports_img2vid"] = getattr(plugin, "supports_img2img", False)
            result.append(entry)
    return result


def list_image_models() -> list[dict]:
    return list_models("image")


def list_video_models() -> list[dict]:
    return list_models("video")


def list_audio_models() -> list[dict]:
    return list_models("audio")


def list_text_models() -> list[dict]:
    return list_models("text")


def list_styles() -> list[str]:
    """Return style names available in the style selector (from styles.json)."""
    prop = bpy.types.Scene.bl_rna.properties.get("generatorai_styles")
    if prop is None:
        return ["no_style"]
    return [item.identifier for item in prop.enum_items]


def get_current_model(media_type: str) -> str:
    """Return the MODEL_ID currently selected for media_type."""
    prefs = _prefs()
    attr = {
        "image": "image_model_card_id",
        "video": "movie_model_card_id",
        "audio": "audio_model_card_id",
        "text":  "text_model_card_id",
    }.get(media_type)
    if attr is None:
        raise ValueError(f"Unknown media_type: {media_type!r}")
    return getattr(prefs, attr, "")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def set_model(media_type: str, model_id: str) -> None:
    """Select a model for the given media type."""
    prefs = _prefs()
    id_attr, card_attr = {
        "image": ("image_model_card_id", "image_model_card"),
        "video": ("movie_model_card_id", "movie_model_card"),
        "audio": ("audio_model_card_id", "audio_model_card"),
        "text":  ("text_model_card_id",  "text_model_card"),
    }.get(media_type, (None, None))
    if id_attr is None:
        raise ValueError(f"Unknown media_type: {media_type!r}")
    setattr(prefs, id_attr, model_id)
    try:
        setattr(prefs, card_attr, model_id)
    except Exception:
        pass  # enum may not match yet if model was set programmatically


def set_style(style_name: str) -> None:
    """Set the active style (use a name from list_styles())."""
    _scene().generatorai_styles = style_name


def set_input_image(path: str) -> None:
    """Set the input image path for img2img / img2vid operations."""
    _scene().image_path = path


def set_input_video(path: str) -> None:
    """Set the input video path for vid2vid operations."""
    _scene().movie_path = path


def set_input_audio(path: str) -> None:
    """Set the input audio path for voice-cloning / audio operations."""
    _scene().sound_path = path


def set_ref_audio(path: str, transcription: str = "") -> None:
    """Set voice-cloning reference audio (and optional transcription for Qwen3-TTS)."""
    _scene().ref_audio_path = path
    if transcription:
        _scene().ref_text = transcription


def clear_inputs() -> None:
    """Clear all file path inputs."""
    scene = _scene()
    for attr in ("image_path", "movie_path", "sound_path", "ref_audio_path", "ref_text",
                 "inpaint_selected_strip"):
        try:
            setattr(scene, attr, "")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_common_params(scene, prompt, negative_prompt, width, height,
                          steps, guidance, seed, style, batch):
    scene.generate_movie_prompt = prompt
    if negative_prompt is not None:
        scene.generate_movie_negative_prompt = negative_prompt
    if width is not None:
        scene.generate_movie_x = width
    if height is not None:
        scene.generate_movie_y = height
    if steps is not None:
        scene.movie_num_inference_steps = steps
    if guidance is not None:
        scene.movie_num_guidance = guidance
    if seed is None:
        scene.movie_use_random = True
    else:
        scene.movie_use_random = False
        scene.movie_num_seed = seed
    if style is not None:
        scene.generatorai_styles = style
    if batch is not None:
        scene.movie_num_batch = batch


def _collect_new_strips(before: set[str]) -> list[dict]:
    """Return strips added to the VSE since before was snapshotted."""
    seq_ed = _scene().sequence_editor
    if seq_ed is None:
        return []
    result = []
    for s in seq_ed.strips_all:
        if s.name not in before:
            result.append({
                "name": s.name,
                "type": s.type,
                "channel": s.channel,
                "frame_start": s.frame_start,
                "frame_final_end": s.frame_final_end,
            })
    return result


def _strip_names_before() -> set[str]:
    seq_ed = _scene().sequence_editor
    if seq_ed is None:
        return set()
    return {s.name for s in seq_ed.strips_all}


# ---------------------------------------------------------------------------
# Generation — blocking (inline, may take minutes)
# ---------------------------------------------------------------------------

def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    guidance: float = 7.5,
    seed: int | None = None,
    style: str = "no_style",
    batch: int = 1,
) -> dict:
    """Generate image(s) inline (blocks until done). Returns strip info."""
    try:
        scene = _scene()
        _apply_common_params(scene, prompt, negative_prompt, width, height,
                              steps, guidance, seed, style, batch)
        scene.generatorai_typeselect = "image"
        before = _strip_names_before()
        bpy.ops.sequencer.generate_image()
        return {"status": "ok", "strips": _collect_new_strips(before)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_video(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 576,
    frames: int = 49,
    steps: int = 25,
    guidance: float = 7.5,
    seed: int | None = None,
    style: str = "no_style",
) -> dict:
    """Generate a video clip inline (blocks until done). Returns strip info."""
    try:
        scene = _scene()
        _apply_common_params(scene, prompt, negative_prompt, width, height,
                              steps, guidance, seed, style, 1)
        scene.generate_movie_frames = frames
        scene.generatorai_typeselect = "movie"
        before = _strip_names_before()
        bpy.ops.sequencer.generate_movie()
        return {"status": "ok", "strips": _collect_new_strips(before)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_audio(
    prompt: str,
    duration_seconds: float = 10.0,
    seed: int | None = None,
    speed: float = 1.0,
) -> dict:
    """Generate audio inline (blocks until done). Returns strip info."""
    try:
        scene = _scene()
        scene.generate_movie_prompt = prompt
        fps = scene.render.fps
        scene.audio_length_in_f = int(duration_seconds * fps)
        scene.audio_speed_tts = speed
        if seed is None:
            scene.movie_use_random = True
        else:
            scene.movie_use_random = False
            scene.movie_num_seed = seed
        scene.generatorai_typeselect = "audio"
        before = _strip_names_before()
        bpy.ops.sequencer.generate_audio()
        return {"status": "ok", "strips": _collect_new_strips(before)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_text(prompt: str = "Describe this image") -> dict:
    """Run image captioning / VQA inline. Returns strip info.
    Requires an input image or video to be set first via set_input_image()."""
    try:
        scene = _scene()
        scene.generate_movie_prompt = prompt
        scene.generatorai_typeselect = "text"
        before = _strip_names_before()
        bpy.ops.sequencer.generate_text()
        return {"status": "ok", "strips": _collect_new_strips(before)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Queue — async (non-blocking, background thread)
# ---------------------------------------------------------------------------

def queue_generate(
    media_type: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    style: str = "no_style",
    batch: int = 1,
    duration_seconds: float | None = None,
) -> str:
    """Enqueue a generation job. Returns the job_id. Non-blocking."""
    scene = _scene()
    scene.generatorai_typeselect = media_type if media_type != "video" else "movie"
    _apply_common_params(scene, prompt, negative_prompt, width, height,
                          steps, guidance, seed, style, batch)
    if frames is not None:
        scene.generate_movie_frames = frames
    if duration_seconds is not None:
        fps = scene.render.fps
        scene.audio_length_in_f = int(duration_seconds * fps)

    before_ids = {j.job_id for j in scene.render_queue}
    bpy.ops.sequencer.add_to_queue()

    # find the newly added job_id
    for job in scene.render_queue:
        if job.job_id not in before_ids:
            return job.job_id
    return "unknown"


def get_queue_status() -> list[dict]:
    """Return status of all jobs in the render queue."""
    return [_job_to_dict(j) for j in _scene().render_queue]


def get_job_status(job_id: str) -> dict:
    """Return status dict for a specific job, or {'status': 'not_found'}."""
    for job in _scene().render_queue:
        if job.job_id == job_id:
            return _job_to_dict(job)
    return {"job_id": job_id, "status": "not_found"}


def cancel_job(job_id: str) -> bool:
    """Cancel a queued or running job. Returns True if found."""
    for i, job in enumerate(_scene().render_queue):
        if job.job_id == job_id:
            bpy.ops.sequencer.cancel_queue_job(job_index=i)
            return True
    return False


def clear_queue() -> int:
    """Clear all pending/completed/failed jobs. Returns count removed."""
    before = len(_scene().render_queue)
    bpy.ops.sequencer.clear_queue()
    return before - len(_scene().render_queue)


def _job_to_dict(job) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": round(job.progress * 100, 1),
        "phase": job.phase,
        "output_type": job.output_type,
        "model_card": job.model_card,
        "prompt": job.prompt,
        "output_path": job.output_path,
        "error_message": job.error_message,
    }


# ---------------------------------------------------------------------------
# VSE helpers
# ---------------------------------------------------------------------------

def get_selected_strips() -> list[dict]:
    """Return info about currently selected strips in the VSE."""
    seq_ed = _scene().sequence_editor
    if seq_ed is None:
        return []
    return [
        {
            "name": s.name,
            "type": s.type,
            "channel": s.channel,
            "frame_start": s.frame_start,
            "frame_final_end": s.frame_final_end,
            "select": s.select,
        }
        for s in seq_ed.strips_all
        if s.select
    ]


def get_all_strips() -> list[dict]:
    """Return info about all strips in the VSE."""
    seq_ed = _scene().sequence_editor
    if seq_ed is None:
        return []
    return [
        {
            "name": s.name,
            "type": s.type,
            "channel": s.channel,
            "frame_start": s.frame_start,
            "frame_final_end": s.frame_final_end,
        }
        for s in seq_ed.strips_all
    ]


def set_active_strip(name: str) -> bool:
    """Set a strip as active (and selected) by name. Returns True if found."""
    seq_ed = _scene().sequence_editor
    if seq_ed is None:
        return False
    for s in seq_ed.strips_all:
        if s.name == name:
            seq_ed.active_strip = s
            s.select = True
            return True
    return False


def get_current_frame() -> int:
    return _scene().frame_current


def set_current_frame(frame: int) -> None:
    _scene().frame_current = frame


# ---------------------------------------------------------------------------
# Convenience: high-level workflows
# ---------------------------------------------------------------------------

def generate_variations(
    prompt: str,
    count: int = 4,
    media_type: str = "image",
    **kwargs,
) -> list[dict]:
    """Queue multiple variations of a prompt with different seeds. Returns job_ids."""
    jobs = []
    for _ in range(count):
        job_id = queue_generate(media_type, prompt, seed=random.randint(0, 2**31), **kwargs)
        jobs.append({"job_id": job_id})
    return jobs


def style_compare(
    prompt: str,
    styles: list[str],
    media_type: str = "image",
    **kwargs,
) -> list[dict]:
    """Queue one job per style for easy comparison. Returns job_ids."""
    jobs = []
    for style in styles:
        job_id = queue_generate(media_type, prompt, style=style, **kwargs)
        jobs.append({"job_id": job_id, "style": style})
    return jobs


# ---------------------------------------------------------------------------
# Self-test / info
# ---------------------------------------------------------------------------

def pallaidium_info() -> dict:
    """Return a summary of current Pallaidium state."""
    prefs = _prefs()
    return {
        "addon_id": _ADDON_ID,
        "current_image_model": getattr(prefs, "image_model_card_id", ""),
        "current_video_model": getattr(prefs, "movie_model_card_id", ""),
        "current_audio_model": getattr(prefs, "audio_model_card_id", ""),
        "current_text_model":  getattr(prefs, "text_model_card_id", ""),
        "image_models_count": len(list_image_models()),
        "video_models_count": len(list_video_models()),
        "audio_models_count": len(list_audio_models()),
        "text_models_count":  len(list_text_models()),
        "styles_count": len(list_styles()),
        "queue_jobs": len(_scene().render_queue),
        "vse_strips": len(get_all_strips()),
    }


print("[Pallaidium MCP Tools] Loaded. Call pallaidium_info() to get started.")
