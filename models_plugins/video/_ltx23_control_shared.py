"""Shared IC-LoRA control/reference-strip helpers for the LTX-2.3 staged plugins.

Used by both ltx23_multi_ic_lora.py and ltx23_multi.py so the LoRA-loading /
control-resolution logic has a single source of truth. Leading underscore
keeps this out of the plugin auto-discovery scan.
"""

import os

_IC_LORA_FALLBACK = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"


def apply_loras(pipe, lora_folder, enabled_loras):
    """Load and activate each enabled LoRA file onto pipe. Returns adapter names loaded."""
    import warnings as _warnings
    from ...utils.helpers import clean_filename

    names, weights = [], []
    for item in enabled_loras:
        name = clean_filename(item.name).replace(".", "")
        try:
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message="Already found a `peft_config` attribute",
                )
                pipe.load_lora_weights(
                    lora_folder,
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
        except Exception as e:
            print(f"  LoRA '{item.name}': load error — {e}")
            continue
        loaded = {a for v in pipe.get_list_adapters().values() for a in v}
        if name in loaded:
            names.append(name)
            weights.append(getattr(item, "weight_value", 1.0))
            print(f"  LoRA '{item.name}': loaded (weight={weights[-1]})")
        else:
            print(f"  LoRA '{item.name}': no matching keys for LTX-2.3, skipped.")
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
        print(f"  Active LoRAs: {names}")
    else:
        print("  No compatible LoRAs applied.")
    return names


def ensure_ic_lora(pipe, lora_folder, enabled_loras, cache_dir, lfo, *, enable_fallback=True):
    """Load enabled LoRAs; if none took and enable_fallback, auto-download the IC-LoRA fallback."""
    names = []
    if enabled_loras and lora_folder:
        names = apply_loras(pipe, lora_folder, enabled_loras)

    if not names and enable_fallback:
        loaded = {a for v in pipe.get_list_adapters().values() for a in v}
        if not loaded:
            print(f"[LTX23Control] No LoRA loaded — auto-loading fallback: {_IC_LORA_FALLBACK}")
            try:
                pipe.load_lora_weights(
                    _IC_LORA_FALLBACK,
                    weight_name="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
                    adapter_name="ic_lora_fallback",
                    cache_dir=cache_dir,
                    local_files_only=lfo,
                )
                pipe.set_adapters(["ic_lora_fallback"], adapter_weights=[1.0])
                print("[LTX23Control] Fallback IC-LoRA loaded.")
                names = ["ic_lora_fallback"]
            except Exception as _e:
                print(f"[LTX23Control] WARNING: Fallback IC-LoRA load failed ({_e}). Proceeding without IC-LoRA.")
    return names


def resolve_control_inputs(image_input, vid_path, ref_image_path, ctrl_video_path, load_first_frame, load_video):
    """Resolve 3DREAL ref-image override + control-video frames.

    Returns (image_input, video_is_control, control_video_frames).

    - image_input: possibly overridden with the ref-image's first frame when
      a 3DREAL ref image is set (ref_image_path present).
    - video_is_control: True when a ref image AND a main input video are both
      present, meaning the main video drives control_video instead of being
      used as a first-frame image condition.
    - control_video_frames: loaded frames from the control source (explicit
      IC-LoRA control MOVIE, else the main input video in 3DREAL mode), or
      None if no control video source is available.
    """
    # 3DREAL mode: the "ref strip" dropdown holds a still image. Use it as the
    # frame-0 appearance reference and let the MAIN input video drive
    # control_video (matches fal/LTX-2.3-3DREAL: video_url + image_url).
    if ref_image_path and os.path.isfile(ref_image_path):
        try:
            image_input = load_first_frame(ref_image_path)
            print(f"[LTX23Control] Ref image loaded: {ref_image_path!r}")
        except Exception as _e:
            print(f"[LTX23Control] WARNING: failed to load ref image ({_e})")

    video_is_control = bool(ref_image_path) and bool(vid_path)

    control_video_frames = None
    control_src = ctrl_video_path if (ctrl_video_path and os.path.isfile(ctrl_video_path)) else None
    if control_src is None and video_is_control and vid_path and os.path.isfile(vid_path):
        control_src = vid_path
    if control_src:
        try:
            control_video_frames = load_video(control_src)
            print(f"[LTX23Control] Control video loaded: {len(control_video_frames)} frames from {control_src!r}")
        except Exception as _e:
            print(f"[LTX23Control] WARNING: failed to load control video ({_e})")

    return image_input, video_is_control, control_video_frames


def print_input_summary(tag, *, image_input, vid_path, sound_path, ref_image_path,
                         ctrl_video_path, ctrl_audio_path, video_is_control,
                         control_video_frames, control_active, ctrl_strength,
                         ctrl_downscale, ctrl_audio_str, identity_guid,
                         stage_mode, lora_folder, enabled_loras):
    """Print every resolved input for this job — one line per input type.

    Run with a plain "no control" job to see the baseline, then compare
    against a job that sets a Ref Strip to see exactly which fields change.
    """
    _names = [getattr(i, "name", "?") for i in (enabled_loras or [])]
    print(f"[{tag}] ── Resolved inputs ─────────────────────────────────")
    print(f"[{tag}]   image_input (first-frame/appearance cond.) : "
          f"{'set (' + type(image_input).__name__ + ')' if image_input is not None else 'None'}")
    print(f"[{tag}]   main video   (vid_path)                    : {vid_path or 'None'}")
    print(f"[{tag}]   main audio   (sound_path)                  : {sound_path or 'None'}")
    print(f"[{tag}]   ref image    (ref_image_path, 3DREAL)      : {ref_image_path or 'None'}")
    print(f"[{tag}]   control video (ctrl_video_path, explicit)  : {ctrl_video_path or 'None'}")
    print(f"[{tag}]   control audio (ctrl_audio_path)            : {ctrl_audio_path or 'None'}")
    print(f"[{tag}]   video_is_control (3DREAL engaged: main video drives control_video): {video_is_control}")
    print(f"[{tag}]   control_video_frames                       : "
          f"{'None' if control_video_frames is None else f'{len(control_video_frames)} frames loaded'}")
    print(f"[{tag}]   control_active (gates LoRA auto-fallback)  : {control_active}")
    print(f"[{tag}]   control_strength={ctrl_strength} control_downscale={ctrl_downscale} "
          f"control_audio_strength={ctrl_audio_str} identity_guidance={identity_guid}")
    print(f"[{tag}]   stage_mode                                 : {stage_mode}")
    print(f"[{tag}]   lora_folder                                : {lora_folder or 'None'}")
    print(f"[{tag}]   enabled_loras                               : {_names or 'none'}")
    if not video_is_control and control_video_frames is None:
        print(f"[{tag}]   NOTE: no control video is engaged for this job — the main "
              f"video (if any) only contributes its first frame as an image condition. "
              f"Set a Ref Strip (MOVIE, SCENE, or META) to drive full control_video conditioning.")
    print(f"[{tag}] ─────────────────────────────────────────────────────")
