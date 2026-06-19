"""Video upscaling via NVIDIA Maxine Video Super Resolution."""

import os
import shutil
import subprocess

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename, load_video_as_np_array


def _np_frame_to_chw(frame):
    """Convert a single HWC uint8 RGB numpy frame to CHW float32 CUDA tensor."""
    import torch

    return (
        torch.from_numpy(frame)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
        .to(device="cuda")
    )


def _source_has_audio(path):
    try:
        import av
        with av.open(path) as container:
            return any(s.type == "audio" for s in container.streams)
    except Exception:
        return False


def _mux_audio(video_path, audio_source, output_path):
    """Copy audio from audio_source into video_path, writing to output_path."""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_source,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
        return os.path.isfile(output_path)
    except Exception:
        return False


class MaxineVSRVideoPlugin(ModelPlugin):
    MODEL_ID     = "nvidia/maxine-vsr-video"
    DISPLAY_NAME = "Video: Maxine Super Resolution"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "AI video super-resolution + denoise + deblur via NVIDIA Maxine"

    INPUTS      = InputSpec.VIDEO
    UI_SECTIONS = [UISection.VIDEO_STRIP, UISection.RESOLUTION, UISection.SEED]
    PARAMS      = ParamSpec(width=1920, height=1080)

    REQUIRED_PACKAGES          = ["torch", "nvvfx"]
    supports_inpaint           = False
    supports_img2img           = False
    requires_input_strip       = True
    uses_standard_input_strip  = False
    show_enhance               = False
    supports_batch             = False

    def is_available(self):
        try:
            import nvvfx  # noqa: F401
        except ImportError:
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                return False
        except ImportError:
            return False
        return True

    def draw_post_seed_ui(self, col, context):
        col.prop(context.scene, "maxine_quality")

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "converter": None, "refiner": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import cv2
        from nvvfx import VideoSuperRes
        from nvvfx.effects.video_super_res import QualityLevel

        vid_path = inputs.video_path
        if not vid_path:
            raise ValueError("Maxine VSR requires an input video strip.")

        self.set_phase(inputs, "Reading video")
        frames = load_video_as_np_array(vid_path)
        total = len(frames)
        if total == 0:
            raise ValueError("Input video has no frames.")

        cap = cv2.VideoCapture(vid_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or inputs.fps
        cap.release()

        out_w, out_h = inputs.width, inputs.height

        quality_name = getattr(scene, "maxine_quality", "HIGH")
        quality = getattr(QualityLevel, quality_name, QualityLevel.HIGH)

        self.set_phase(inputs, "Upscaling")
        upscaled = []

        with VideoSuperRes(quality=quality) as sr:
            sr.output_width = out_w
            sr.output_height = out_h
            sr.load()

            for i, frame in enumerate(frames):
                tensor_in = _np_frame_to_chw(frame)
                result = sr.run(tensor_in)
                tensor_out = torch.from_dlpack(result.image).clone()

                out_frame = (
                    tensor_out.clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                upscaled.append(out_frame)

                if inputs.progress_fn and total > 0:
                    inputs.progress_fn(i + 1, total)

        self.set_phase(inputs, "Saving")

        dst_path = solve_path(
            clean_filename(str(inputs.seed) + "_maxine_vsr") + ".mp4"
        )

        try:
            from diffusers.utils import export_to_video
            from PIL import Image

            pil_frames = [Image.fromarray(f) for f in upscaled]
            tmp_path = export_to_video(pil_frames, fps=src_fps)
            shutil.move(tmp_path, dst_path)
        except ImportError:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(dst_path, fourcc, src_fps, (out_w, out_h))
            for f in upscaled:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()

        if _source_has_audio(vid_path):
            self.set_phase(inputs, "Muxing audio")
            muxed_path = dst_path.replace(".mp4", "_muxed.mp4")
            if _mux_audio(dst_path, vid_path, muxed_path):
                os.replace(muxed_path, dst_path)

        return dst_path
