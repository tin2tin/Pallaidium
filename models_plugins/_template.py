"""
Pallaidium Model Plugin Template
=================================
Copy this file to models_plugins/{type}/your_model_name.py, fill in the
class attributes and implement load() + generate(). Restart Blender — done.

Naming convention for the file: lowercase, underscores, no spaces.
  video/      → models that produce .mp4
  image/      → models that produce .png / .jpg
  audio/      → models that produce .wav
  text/       → models that produce a text strip

Files and directories whose names start with _ are ignored by the loader.
"""

# The base classes live two levels up in models/base.py.
# Use a relative import like this:
from ...models.base import (
    ModelPlugin,
    InputSpec,
    UISection,
    ParamSpec,
    ModelInputs,
)


class MyModelPlugin(ModelPlugin):
    # ------------------------------------------------------------------
    # Identity  (required — all three must be non-empty)
    # ------------------------------------------------------------------

    MODEL_ID     = "author/my-model-name"      # HuggingFace repo or unique key
    DISPLAY_NAME = "My Model (resolution)"     # shown in the dropdown
    MODEL_TYPE   = "image"                     # "video" | "image" | "audio" | "text"
    DESCRIPTION  = "Short tooltip description"

    # ------------------------------------------------------------------
    # Inputs  (bitflags — OR together what you need)
    # ------------------------------------------------------------------
    #
    # InputSpec.PROMPT          text prompt
    # InputSpec.NEG_PROMPT      negative prompt
    # InputSpec.IMAGE           one image from selected strip
    # InputSpec.MULTI_IMAGE     up to PARAMS.max_multi_images dynamic image strips
    # InputSpec.TRIPLE_IMAGE    exactly 3 fixed image pickers
    # InputSpec.AUDIO_REF       speaker reference audio (optional)
    # InputSpec.AUDIO_REF_REQ   speaker reference audio (required)
    # InputSpec.TEXT_REF        reference transcription text
    # InputSpec.VIDEO           video strip input
    # InputSpec.FACE_FOLDER     IP Adapter face folder
    # InputSpec.STYLE_FOLDER    IP Adapter style folder
    # InputSpec.LORA            LoRA files with weights
    # InputSpec.API_KEY         external API key
    #
    INPUTS = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE

    # ------------------------------------------------------------------
    # UI sections  (listed in the order they appear in the panel)
    # ------------------------------------------------------------------
    #
    # UISection.PROMPT            text prompt textarea
    # UISection.NEG_PROMPT        negative prompt textarea
    # UISection.IMAGE_STRIP       single image eyedropper
    # UISection.MULTI_IMAGES      dynamic add/remove image strips
    # UISection.TRIPLE_IMAGE      three fixed image pickers
    # UISection.TRIPLE_PROMPT_IMG three (prompt + image) pairs  (OmniGen)
    # UISection.AUDIO_REF         speaker reference file picker
    # UISection.TEXT_REF          reference text input
    # UISection.VIDEO_STRIP       video strip eyedropper
    # UISection.RESOLUTION        width × height dropdowns
    # UISection.FRAMES            frame count slider
    # UISection.STEPS             inference steps slider
    # UISection.GUIDANCE          guidance / word-power slider
    # UISection.IMAGE_STRENGTH    img2img / inpaint strength slider
    # UISection.SEED              seed + randomise toggle
    # UISection.LORA              LoRA folder + file list
    # UISection.IP_ADAPTER        face + style folder pickers
    # UISection.AUDIO_DURATION    audio duration slider
    # UISection.SPEED             audio speed slider
    # UISection.CHAT_PARAMS       exaggeration + pace + temperature
    # UISection.ILLUMINATION      lighting style + direction dropdowns
    # UISection.POSE_TOGGLE       "Read as OpenPose Rig" checkbox
    # UISection.SCRIBBLE_TOGGLE   "Read as Scribble" checkbox
    # UISection.ENHANCE           Quality / Speed / Faces / Upscale toggles
    #
    UI_SECTIONS = [
        UISection.PROMPT,
        UISection.NEG_PROMPT,
        UISection.IMAGE_STRIP,
        UISection.STEPS,
        UISection.GUIDANCE,
        UISection.SEED,
    ]

    # ------------------------------------------------------------------
    # Parameter defaults  (override only what differs from generic)
    # ------------------------------------------------------------------

    PARAMS = ParamSpec(
        width=1024,
        height=1024,
        steps=20,
        guidance=5.0,
    )

    # ------------------------------------------------------------------
    # Required packages  (checked by is_available())
    # ------------------------------------------------------------------

    REQUIRED_PACKAGES = ["diffusers", "torch", "PIL"]

    # ------------------------------------------------------------------
    # load()  —  called once; return value is cached by MODEL_ID
    # ------------------------------------------------------------------

    def load(self, prefs, scene, **kwargs):
        """Load the pipeline. Called once per Blender session (result is cached)."""
        import torch
        from diffusers import AutoPipelineForImage2Image  # replace with your pipeline

        _cache_dir = prefs.hf_cache_dir or None
        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=_cache_dir,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    # ------------------------------------------------------------------
    # generate()  —  called each time the user hits Generate
    # ------------------------------------------------------------------

    def generate(self, pipe, inputs: ModelInputs, scene, prefs) -> str:
        """Run inference. Return the absolute path to the output file."""
        import torch
        from PIL import Image

        # Signal phase transitions so the queue panel shows what's happening.
        # set_phase() is a no-op when running outside the queue (inputs.phase_fn is None).
        self.set_phase(inputs, "Generating")

        # inputs fields are populated from whatever you declared in INPUTS.
        # Unpopulated fields stay at their ModelInputs defaults.
        result = pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            image=inputs.image,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            generator=torch.Generator().manual_seed(inputs.seed),
            # Wire the step counter so the panel shows "Step N/total".
            # step_callback() returns None when running outside the queue.
            callback_on_step_end=self.step_callback(inputs),
        )

        self.set_phase(inputs, "Saving")
        # Save and return the output path.
        # Use the helpers already available in helpers.py if needed.
        output_path = _save_image(result.images[0], scene)
        return output_path

    # ------------------------------------------------------------------
    # draw_custom_ui()  —  optional escape hatch for non-standard UI
    # ------------------------------------------------------------------

    def draw_custom_ui(self, layout, context):
        """Render UI elements that don't fit any standard UISection.

        Called after all UI_SECTIONS have been rendered.
        Leave this method out entirely if you don't need it.
        """
        # Example: a custom checkbox
        # scene = context.scene
        # layout.prop(scene, "my_custom_property", text="My Option")
        pass


# ------------------------------------------------------------------
# Helper (example only — remove or replace)
# ------------------------------------------------------------------

def _save_image(img, scene):
    """Placeholder — replace with actual save logic or use helpers.py."""
    import os
    import bpy
    output_dir = bpy.context.preferences.addons[
        __package__.split(".")[0]
    ].preferences.generator_ai
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "output.png")
    img.save(path)
    return path
