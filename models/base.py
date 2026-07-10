"""
Plugin base classes for Pallaidium Generative AI.

Every AI model is a ModelPlugin subclass living in models_plugins/{type}/*.py.
The framework discovers, loads, and dispatches through the registry automatically.
"""

import importlib
from dataclasses import dataclass, field
from enum import Flag, auto, Enum
from typing import Optional, Any


# ---------------------------------------------------------------------------
# InputSpec — declare what data the model needs from the scene
# ---------------------------------------------------------------------------

class InputSpec(Flag):
    """Bitflags declaring which inputs a model consumes.

    The framework collects exactly these fields from the Blender scene/strips
    and passes them as a ModelInputs instance to generate().
    """
    PROMPT        = auto()   # text prompt
    NEG_PROMPT    = auto()   # negative prompt
    IMAGE         = auto()   # single image from selected strip
    MULTI_IMAGE   = auto()   # up to PARAMS.max_multi_images dynamic image strips
    TRIPLE_IMAGE  = auto()   # exactly 3 fixed image pickers (e.g. Qwen-Edit)
    AUDIO_REF     = auto()   # speaker reference audio (optional)
    AUDIO_REF_REQ = auto()   # speaker reference audio (required, e.g. F5-TTS)
    TEXT_REF      = auto()   # reference transcription text (e.g. Qwen3-TTS)
    VIDEO         = auto()   # video strip input (e.g. MMAudio)
    FACE_FOLDER   = auto()   # IP Adapter face image folder
    STYLE_FOLDER  = auto()   # IP Adapter style image folder
    LORA          = auto()   # LoRA files with per-file weights
    API_KEY       = auto()   # external runtime API key (e.g. MiniMax)
    HF_TOKEN      = auto()   # HuggingFace token (gated-model download)
    MUSIC_PARAMS  = auto()   # BPM, key, time signature, lyrics for music generation


# ---------------------------------------------------------------------------
# UISection — declare which UI sections to render and in what order
# ---------------------------------------------------------------------------

class UISection(str, Enum):
    """Ordered list of UI sections a plugin opts into.

    panels.py iterates plugin.UI_SECTIONS and calls the matching renderer
    for each entry. Use draw_custom_ui() for anything not listed here.
    """

    # --- Prompt inputs ---
    PROMPT            = "prompt"           # main text prompt textarea
    NEG_PROMPT        = "neg_prompt"       # negative prompt textarea

    # --- Image/video/audio strip inputs ---
    IMAGE_STRIP       = "image_strip"      # single image eyedropper + strip name
    MULTI_IMAGES      = "multi_images"     # dynamic add/remove image strips
                                           # (count limited by PARAMS.max_multi_images)
    TRIPLE_IMAGE      = "triple_image"     # three fixed image strip pickers
    TRIPLE_PROMPT_IMG = "triple_prompt_img"# three (prompt textarea + image picker) pairs
                                           # used by OmniGen
    AUDIO_REF         = "audio_ref"        # speaker reference file picker
    TEXT_REF          = "text_ref"         # reference text input (Qwen3-TTS)
    VIDEO_STRIP       = "video_strip"      # video strip eyedropper

    # --- Generation parameters ---
    RESOLUTION        = "resolution"       # width × height dropdowns
    FRAMES            = "frames"           # frame count slider
    STEPS             = "steps"            # inference steps slider
    GUIDANCE          = "guidance"         # guidance / word-power slider
    IMAGE_STRENGTH    = "image_strength"   # img2img / inpaint strength slider
    SEED              = "seed"             # seed input + randomise toggle

    # --- LoRA / IP Adapter ---
    LORA              = "lora"             # LoRA folder + file list with enable/weight
    IP_ADAPTER        = "ip_adapter"       # face folder + style folder pickers

    # --- Audio-specific ---
    AUDIO_DURATION    = "audio_duration"   # duration in frames / seconds
    SPEED             = "speed"            # audio playback-speed / CPS slider
    CHAT_PARAMS       = "chat_params"      # exaggeration + pace + temperature
    AUDIO_OUTPUT      = "audio_output"     # toggle: generate synchronized audio for a video output

    # --- Image-specific extras ---
    ILLUMINATION      = "illumination"     # lighting style + light direction dropdowns
    POSE_TOGGLE       = "pose_toggle"      # "Read as OpenPose Rig Image" checkbox
    SCRIBBLE_TOGGLE   = "scribble_toggle"  # "Read as Scribble Image" checkbox
    ENHANCE           = "enhance"          # Quality / Speed / Faces / Upscale toggles
    MUSIC_PARAMS      = "music_params"     # BPM, key, time signature, and lyrics inputs


# ---------------------------------------------------------------------------
# ParamSpec — per-model defaults and constraints
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """Default values and constraints for generation parameters.

    Override only the fields that differ from the generic defaults.
    """
    width:  int   = 1024
    height: int   = 576
    frames: int   = 49
    steps:  int   = 25
    guidance: float = 7.5
    strength: float = 0.8        # img2img / inpaint strength
    audio_length: float = 5.0    # seconds
    max_multi_images: int = 1    # FLUX.2-dev sets this to 9
    audio_ref_required: bool = False  # True → UI shows "required" label


# ---------------------------------------------------------------------------
# ModelInputs — what generate() receives
# ---------------------------------------------------------------------------

@dataclass
class ModelInputs:
    """Collected inputs passed to ModelPlugin.generate().

    The framework populates only the fields listed in plugin.INPUTS;
    all others remain at their default. Plugins should not read fields
    they did not declare in INPUTS.
    """
    # Text
    prompt:       str = ""
    neg_prompt:   str = ""
    text_ref:     str = ""      # Qwen3-TTS reference transcription

    # Mode — set by dispatcher before calling load()/generate()
    mode:         str  = "txt2img"   # "txt2img" | "img2img" | "inpaint"

    # Images  (populated based on InputSpec flags)
    image:        Optional[Any] = None           # PIL.Image — single image (init/source)
    inpaint_mask: Optional[Any] = None           # PIL.Image — inpaint mask (white = paint here)
    images:       list = field(default_factory=list)   # list of PIL.Image
    image_prompts: list = field(default_factory=list)  # per-image prompts (OmniGen)
    last_image:   Optional[Any] = None           # PIL.Image — last-frame condition (FLF / last-frame-only mode)
    middle_images_paths: list = field(default_factory=list)  # [(path_str, fraction_float), ...] for N-anchor (LTX Multi)

    # Audio / video paths
    audio_ref:    Optional[str] = None           # path to speaker reference .wav/.mp3
    video_path:   Optional[str] = None           # path to input video

    # IP Adapter folder paths
    face_folder:  Optional[str] = None
    style_folder: Optional[str] = None

    # LoRA — list of (path, weight) tuples
    lora_files:   list = field(default_factory=list)

    # Generation parameters
    width:        int   = 1024
    height:       int   = 576
    frames:       int   = 49
    fps:          float = 24.0
    steps:        int   = 25
    guidance:     float = 7.5
    strength:     float = 0.8
    seed:         int   = 0
    batch:        int   = 1

    # Audio parameters
    audio_length: float = 5.0
    speed:        float = 1.0
    exaggeration: float = 0.5
    pace:         float = 0.5
    temperature:  float = 0.8

    # Toggles / extras
    use_lcm:           bool = False
    use_adetailer:     bool = False
    use_upscale:       bool = False
    openpose_use_bones: bool = False
    use_scribble:      bool = False
    remove_silence:    bool = False
    is_voice_clone:    bool = False   # Chatterbox VC mode (input is a SOUND strip)

    # Lighting (Kontext Relight)
    illumination_style: str = ""
    light_direction:    str = ""

    # Music generation (ACE-Step etc.)
    bpm:              int = 0    # 0 = model estimates
    lyrics:           str = ""
    key_scale:        str = ""   # e.g. "C major", "A minor"
    time_signature:   str = ""   # e.g. "4" for 4/4, "3" for 3/4

    # Queue insertion hints — set by the queue for requires_main_thread plugins.
    # 0 means "not set" (plugin should use its own auto-detection logic).
    insert_channel:     int = 0   # target VSE channel for output strips
    insert_frame_start: int = 0   # timeline frame where the source strip starts

    # Progress reporting — set by the queue worker; None when running interactively.
    # Signature: progress_fn(step: int, total_steps: int) → None
    progress_fn:      Optional[Any] = None
    # Phase reporting — set by the queue worker; None when running interactively.
    # Signature: phase_fn(label: str) → None
    phase_fn:         Optional[Any] = None
    # Cooperative cancellation — set by the queue worker; None when running
    # interactively. Signature: should_cancel() → bool. Long-running plugins
    # (e.g. remote backends polling a job) may check this and abort.
    should_cancel:    Optional[Any] = None

    # Result note — a short, human-readable string a plugin may set during
    # generate() (e.g. token usage / cost).  Surfaced on the completed queue job.
    usage_note:       str = ""


# ---------------------------------------------------------------------------
# ModelPlugin — base class every plugin must subclass
# ---------------------------------------------------------------------------

class ModelPlugin:
    """Base class for all Pallaidium model plugins.

    Subclass this in a file under models_plugins/{type}/your_model.py.
    Set the class attributes, implement load() and generate(), done.
    """

    # ---- Required class attributes ----------------------------------------

    MODEL_ID:     str = ""   # unique identifier / HuggingFace repo ID
    DISPLAY_NAME: str = ""   # shown in the UI dropdown
    MODEL_TYPE:   str = ""   # "video" | "image" | "audio" | "text" | "3d"
    DESCRIPTION:  str = ""   # tooltip text shown in the dropdown

    # ---- What data to collect from the scene ------------------------------

    INPUTS: InputSpec = InputSpec.PROMPT

    # ---- Which UI sections to render (in order) ---------------------------

    UI_SECTIONS: list = [UISection.PROMPT, UISection.STEPS, UISection.SEED]

    # ---- Parameter defaults / constraints ---------------------------------

    PARAMS: ParamSpec = field(default_factory=ParamSpec) if False else ParamSpec()

    # ---- Python packages needed (for availability check) ------------------

    REQUIRED_PACKAGES: list = []

    # ---- Capability flags (read by the dispatcher and UI) -----------------

    supports_inpaint:          bool = True   # False → inpaint mode never activated
    supports_img2img:          bool = True   # False → img2img conversion never activated
    requires_input_strip:      bool = False  # True  → generation always requires a selected strip
    uses_standard_input_strip: bool = True   # False → plugin provides its own strip input UI
    uses_strip_power:          bool = True   # False → hide the "Strip Power" (image_power) slider
    strip_power_inpaint_only:  bool = False  # True  → hide "Strip Power" unless an inpaint mask is
                                             #         selected (plugin only wires strength into its
                                             #         inpaint pipeline; it's a no-op for txt2img/img2img)
    show_enhance:              bool = True   # False → hide Quality/Speed/Upscale enhance row
    requires_main_thread:      bool = False  # True  → run generate() on main thread (bpy access needed)
    supports_batch:            bool = True   # False → hide "Batch Count" (deterministic single-output
                                             #         models: captioning, transcription, stem split, …)
    supports_input_downscale:  bool = False  # True  → queue honours ltx23ic_input_downscale_pct when
                                             #         pre-rendering a MOVIE/SCENE main-input strip

    # -----------------------------------------------------------------------

    @staticmethod
    def step_callback(inputs: "ModelInputs"):
        """Return a diffusers-compatible callback_on_step_end, or None.

        Pass the return value directly to the pipeline's callback_on_step_end
        parameter.  When inputs.progress_fn is None (interactive mode) this
        returns None so the pipeline runs unchanged.

        Usage in any plugin::

            pipe_result = pipe(
                ...,
                callback_on_step_end=self.step_callback(inputs),
            )
        """
        if inputs.progress_fn is None:
            return None
        fn = inputs.progress_fn
        total = max(1, inputs.steps)
        def _cb(pipe, step: int, timestep, callback_kwargs: dict):
            fn(step + 1, total)
            return callback_kwargs
        return _cb

    @staticmethod
    def set_phase(inputs: "ModelInputs", label: str) -> None:
        """Signal a phase transition; no-op when running outside the queue."""
        if inputs.phase_fn is not None:
            inputs.phase_fn(label)

    def is_available(self) -> bool:
        """Return False if any required package is missing.

        Override for more complex checks (e.g. platform-specific packages).
        """
        for pkg in self.REQUIRED_PACKAGES:
            try:
                importlib.import_module(pkg)
            except ImportError:
                return False
        return True

    def load(self, prefs, scene, **kwargs) -> Any:
        """Load and return the model pipeline.

        Called once; the framework caches the return value by MODEL_ID.
        Receive prefs (addon preferences) and scene (bpy scene) for settings.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.load() is not implemented")

    def generate(self, pipe: Any, inputs: ModelInputs, scene, prefs) -> str:
        """Run inference and return the output file path.

        Args:
            pipe:   The pipeline object returned by load().
            inputs: Collected ModelInputs, populated per plugin.INPUTS.
            scene:  bpy.context.scene
            prefs:  Add-on preferences

        Returns:
            Absolute path to the generated file (video/image/audio).
        """
        raise NotImplementedError(f"{self.__class__.__name__}.generate() is not implemented")

    def draw_custom_ui(self, col, context) -> bool:
        """Optional: draw model-specific UI inside the input selector column.

        Return True  → completely replaced the standard input_strips dropdown.
        Return False → either added extra controls, or did nothing.
        """
        return False

    def draw_post_enhance_ui(self, col, context) -> None:
        """Optional: draw model-specific UI after the Enhance row."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.MODEL_ID!r} type={self.MODEL_TYPE!r}>"
