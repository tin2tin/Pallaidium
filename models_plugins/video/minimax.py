"""Cloud video generation via MiniMax API (txt2vid, img2vid, subject2vid)."""

import os
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import (
    solve_path, clean_filename, find_strip_by_name,
    invoke_video_generation, query_video_generation, fetch_video_result, minimax_validate_image,
)


class _MiniMaxBase(ModelPlugin):
    MODEL_TYPE              = "video"
    INPUTS                  = InputSpec.PROMPT | InputSpec.API_KEY
    UI_SECTIONS             = [UISection.PROMPT, UISection.FRAMES, UISection.SEED]
    PARAMS                  = ParamSpec(steps=1, guidance=1.0)
    REQUIRED_PACKAGES       = []
    uses_standard_input_strip = False
    supports_batch          = False  # external API single-shot generation

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

    def _get_api_key(self, prefs):
        import bpy
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ops_dir = os.path.join(current_dir, "..", "..", "operators")
        key_path = os.path.join(ops_dir, "MiniMax_API.txt")
        if os.path.isfile(key_path):
            with open(key_path) as f:
                return f.read().strip()
        return ""

    def _poll_and_download(self, task_id: str, api_key: str, dst_path: str) -> str:
        while True:
            file_id, status = query_video_generation(task_id, api_key)
            if file_id:
                result = fetch_video_result(file_id, api_key, dst_path)
                if os.path.exists(result):
                    return result
                raise RuntimeError("MiniMax video download failed.")
            if status in ("Fail", "Unknown"):
                raise RuntimeError(f"MiniMax generation failed: status={status}")


class MiniMaxTxt2VidPlugin(_MiniMaxBase):
    MODEL_ID     = "Hailuo/MiniMax/txt2vid"
    DISPLAY_NAME = "Video: MiniMax txt2vid (cloud)"
    DESCRIPTION  = "Cloud text-to-video via MiniMax Hailuo API"

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        api_key = self._get_api_key(prefs)
        if not api_key:
            raise RuntimeError("MiniMax API key is missing.")
        self.set_phase(inputs, "Submitting to cloud")
        dst_path = solve_path(clean_filename(inputs.prompt[:20]) + ".mp4")
        task_id = invoke_video_generation(inputs.prompt[:2000], api_key, None, self.MODEL_ID)
        print(f"MiniMax task submitted: {task_id}")
        self.set_phase(inputs, "Waiting for cloud")
        return self._poll_and_download(task_id, api_key, dst_path)


class MiniMaxImg2VidPlugin(_MiniMaxBase):
    MODEL_ID     = "Hailuo/MiniMax/img2vid"
    DISPLAY_NAME = "Video: MiniMax img2vid (cloud)"
    DESCRIPTION  = "Cloud image-to-video via MiniMax Hailuo API"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.API_KEY

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import bpy

        api_key = self._get_api_key(prefs)
        if not api_key:
            raise RuntimeError("MiniMax API key is missing.")

        image_path = bpy.path.abspath(scene.image_path) if getattr(scene, "image_path", "") else None
        if not image_path or not minimax_validate_image(image_path):
            raise RuntimeError("MiniMax img2vid requires a valid image strip.")

        self.set_phase(inputs, "Submitting to cloud")
        dst_path = solve_path(clean_filename(inputs.prompt[:20]) + ".mp4")
        task_id = invoke_video_generation(inputs.prompt[:2000], api_key, image_path, self.MODEL_ID)
        print(f"MiniMax task submitted: {task_id}")
        self.set_phase(inputs, "Waiting for cloud")
        result = self._poll_and_download(task_id, api_key, dst_path)
        self.set_phase(inputs, "Saving")
        return result


class MiniMaxSubject2VidPlugin(_MiniMaxBase):
    MODEL_ID     = "Hailuo/MiniMax/subject2vid"
    DISPLAY_NAME = "Video: MiniMax subject2vid (cloud)"
    DESCRIPTION  = "Cloud subject-to-video via MiniMax Hailuo API"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.API_KEY

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        if scene.sequence_editor is None:
            return False
        row = col.row(align=True)
        row.prop_search(
            scene, "minimax_subject", scene.sequence_editor, "strips",
            text="Subject", icon="USER",
        )
        row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "minimax_select"
        return False

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import bpy, os

        api_key = self._get_api_key(prefs)
        if not api_key:
            raise RuntimeError("MiniMax API key is missing.")

        minimax_subject = getattr(scene, "minimax_subject", "")
        if not minimax_subject:
            raise RuntimeError("MiniMax subject2vid requires a subject strip to be selected.")

        strip = find_strip_by_name(scene, minimax_subject)
        if not strip or strip.type != "IMAGE":
            raise RuntimeError("MiniMax subject2vid: selected subject is not an IMAGE strip.")

        image_path = bpy.path.abspath(
            os.path.join(strip.directory, strip.elements[0].filename)
        )
        if not minimax_validate_image(image_path):
            raise RuntimeError(f"MiniMax subject2vid: image validation failed for {image_path!r}.")

        self.set_phase(inputs, "Submitting to cloud")
        dst_path = solve_path(clean_filename(inputs.prompt[:20]) + ".mp4")
        task_id = invoke_video_generation(inputs.prompt[:2000], api_key, image_path, self.MODEL_ID)
        print(f"MiniMax task submitted: {task_id}")
        self.set_phase(inputs, "Waiting for cloud")
        return self._poll_and_download(task_id, api_key, dst_path)
