"""RemoteModelPlugin — run generation on an OpenAI-/v1-dialect backend.

A drop-in ModelPlugin whose load()/generate() talk HTTP to a remote service
(see utils/remote_backend.py and the Backend Contract) instead of running a
local torch pipeline.

Two ways to get a remote plugin:
  * Subclass RemoteModelPlugin and override build_payload() (hand-written, rare).
  * Call make_remote_plugin(entry) with a /v1/models discovery entry — the
    factory derives INPUTS / UI_SECTIONS / endpoint / payload mapping from the
    model's type + modes + capability hints. This is what the "Refresh Remote
    Models" operator uses, so a backend's models appear without per-model code.

Because this lives under models/ (not models_plugins/) the plugin loader never
auto-registers it; remote plugins are injected into the registry at refresh time.
"""

import os

from .base import (
    ModelPlugin, ModelInputs, InputSpec, UISection, ParamSpec,
)
from ..utils.remote_backend import client_from_prefs, RemoteBackendError


# Contract endpoint + output extension per media type.
_ENDPOINTS = {
    "video": ("/v1/videos", ".mp4"),
    "image": ("/v1/images/generations", ".png"),
    "audio": ("/v1/audio/speech", ".wav"),
    "text":  ("/v1/audio/transcriptions", ".txt"),
}


class RemoteModelPlugin(ModelPlugin):
    """Base for backend-driven plugins.

    Subclasses (or the factory) set:
        MODEL_ID, DISPLAY_NAME, MODEL_TYPE, DESCRIPTION
        ENDPOINT          -- contract path, e.g. "/v1/images/generations"
        OUTPUT_EXT        -- ".png" | ".mp4" | ".wav" | ".txt"
        REMOTE_MODEL_NAME -- the backend's model id (defaults to MODEL_ID)

    build_payload()/upload_inputs() have generic per-type implementations here;
    override build_payload() for a bespoke mapping.
    """

    # No local packages required — only stdlib urllib.
    REQUIRED_PACKAGES: list = []

    # Contract specifics (override / set by factory).
    ENDPOINT: str = ""
    OUTPUT_EXT: str = ".png"
    REMOTE_MODEL_NAME: str = ""

    # Remote backends are single-shot; no local batch loop.
    supports_batch: bool = False

    # Capability hints (set by the factory; safe defaults for hand-written subs).
    _max_ref_images: int = 0
    _needs_speaker_ref: bool = False
    _needs_ref_text: bool = False
    _control_types: list = []

    # ---- identity helpers -------------------------------------------------

    def remote_model_name(self) -> str:
        return self.REMOTE_MODEL_NAME or self.MODEL_ID

    # ---- payload + upload (generic; override build_payload to customise) ---

    def build_payload(self, inputs: ModelInputs, scene, prefs) -> dict:
        t = self.MODEL_TYPE
        if t == "image":
            return {
                "prompt": inputs.prompt,
                "negative_prompt": inputs.neg_prompt,
                "width": inputs.width,
                "height": inputs.height,
                "num_inference_steps": inputs.steps,
                "guidance_scale": inputs.guidance,
                "seed": inputs.seed,
            }
        if t == "video":
            p = {
                "prompt": inputs.prompt,
                "negative_prompt": inputs.neg_prompt,
                "width": inputs.width,
                "height": inputs.height,
                "num_frames": inputs.frames,
                "fps": inputs.fps,
                "seed": inputs.seed,
                "strength": inputs.strength,
            }
            if getattr(self, "_supports_audio_output", False):
                p["generate_audio"] = bool(getattr(scene, "remote_generate_audio", True))
            return p
        if t == "audio":
            return {
                "input": inputs.prompt,
                "voice": "default",
                "response_format": "wav",
                "speed": inputs.speed,
            }
        # text/transcription is handled directly in generate()
        return {}

    def upload_inputs(self, client, inputs: ModelInputs, payload: dict) -> None:
        """Upload every reference the model declared (gated by INPUTS), set ids.

        Uses the contract's POST /v1/files mechanism plus the additive extension
        fields documented in BACKEND_CONTRACT_EXTENSIONS.md.
        """
        flags = self.INPUTS

        # Single init image (img2img / img2vid).
        if (InputSpec.IMAGE in flags) and inputs.image is not None:
            payload["image_file_id"] = self._upload_pil(client, inputs.image)
            if self.MODEL_TYPE == "video":
                payload["image_b64"] = self.image_to_b64(inputs.image)

        # Multiple reference images (e.g. FLUX.2/Klein) + per-image prompts.
        if (InputSpec.MULTI_IMAGE in flags) and inputs.images:
            ids = [self._upload_pil(client, im) for im in inputs.images if im is not None]
            if ids:
                payload["reference_file_ids"] = ids
            if inputs.image_prompts:
                payload["reference_prompts"] = list(inputs.image_prompts)

        # Triple fixed images (e.g. Qwen-Edit).
        if (InputSpec.TRIPLE_IMAGE in flags) and inputs.images:
            ids = [self._upload_pil(client, im) for im in inputs.images if im is not None]
            if ids:
                payload["reference_file_ids"] = ids

        # Last-frame (FLF) and middle anchors (video).
        if getattr(inputs, "last_image", None) is not None and self.MODEL_TYPE == "video":
            payload["last_image_file_id"] = self._upload_pil(client, inputs.last_image)
        if getattr(inputs, "middle_images_paths", None) and self.MODEL_TYPE == "video":
            anchors = []
            for p, frac in inputs.middle_images_paths:
                if p and os.path.isfile(p):
                    anchors.append({
                        "file_id": client.upload_file(p, "reference"),
                        "fraction": frac,
                    })
            if anchors:
                payload["anchor_file_ids"] = anchors

        # Audio reference. For audio models this is a voice-clone speaker ref;
        # for video models (Seedance reference-to-video) it is a reference audio
        # track that fal exposes as audio_urls.
        if (InputSpec.AUDIO_REF in flags or InputSpec.AUDIO_REF_REQ in flags):
            ref = inputs.audio_ref
            if ref and os.path.isfile(ref):
                if self.MODEL_TYPE == "video":
                    payload["reference_audio_ids"] = [client.upload_file(ref, "reference")]
                else:
                    payload["speaker_reference_id"] = client.upload_file(ref, "speaker_reference")
        if (InputSpec.TEXT_REF in flags) and inputs.text_ref:
            payload["speaker_reference_text"] = inputs.text_ref

        # Video control input (canny/depth/pose) for video models.
        if (InputSpec.VIDEO in flags) and self.MODEL_TYPE == "video":
            vp = inputs.video_path
            if vp and os.path.isfile(vp):
                payload["control_file_id"] = client.upload_file(vp, "control")
                if self._control_types:
                    payload.setdefault("control_type", self._control_types[0])
                payload.setdefault("control_strength", inputs.strength)

        # IP-Adapter face / style folders.
        if (InputSpec.FACE_FOLDER in flags) and inputs.face_folder:
            ids = self._upload_folder(client, inputs.face_folder)
            if ids:
                payload["ip_face_file_ids"] = ids
        if (InputSpec.STYLE_FOLDER in flags) and inputs.style_folder:
            ids = self._upload_folder(client, inputs.style_folder)
            if ids:
                payload["ip_style_file_ids"] = ids

    # ---- ModelPlugin interface -------------------------------------------

    def load(self, prefs, scene, **kw):
        """'Loading' a remote model = verifying the backend is reachable."""
        client = client_from_prefs(prefs)
        try:
            client.check_compatible()
        except RemoteBackendError as e:
            raise RuntimeError(f"Remote backend unavailable: {e}") from e
        return {"client": client}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        # Rebuild from prefs so a changed URL/key always wins over the cached one.
        client = client_from_prefs(prefs)

        # Transcription returns text, not a media file.
        if self.MODEL_TYPE == "text":
            return self._transcribe(client, inputs)

        from ..utils.helpers import solve_path, clean_filename

        self.set_phase(inputs, "Submitting to backend")
        # Gather multi-reference images from the flux-style strip pickers.
        if InputSpec.MULTI_IMAGE in self.INPUTS:
            self._collect_multi_images(inputs, scene)
        payload = self.build_payload(inputs, scene, prefs)
        payload.setdefault("model", self.remote_model_name())
        try:
            self.upload_inputs(client, inputs, payload)
        except RemoteBackendError as e:
            raise RuntimeError(str(e)) from e

        print(f"[Remote] {self.MODEL_ID} → {self.ENDPOINT} payload keys: "
              f"{sorted(payload.keys())}")

        stem = clean_filename((inputs.prompt or self.MODEL_ID)[:20]) or "remote"
        dst_path = solve_path(stem + self.OUTPUT_EXT)

        self.set_phase(inputs, "Waiting for backend")
        try:
            return client.run(
                self.ENDPOINT, payload, dst_path,
                phase_fn=inputs.phase_fn,
                progress_fn=inputs.progress_fn,
                should_cancel=inputs.should_cancel,
            )
        except RemoteBackendError as e:
            raise RuntimeError(str(e)) from e

    def _transcribe(self, client, inputs: ModelInputs):
        src = inputs.video_path or inputs.audio_ref
        if not src or not os.path.isfile(src):
            raise RuntimeError("Transcription requires a selected audio/video strip.")
        self.set_phase(inputs, "Transcribing")
        try:
            resp = client.transcribe(src, self.remote_model_name())
        except RemoteBackendError as e:
            raise RuntimeError(str(e)) from e
        return resp.get("text", "")

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _upload_pil(client, image, purpose: str = "reference") -> str:
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return client.upload_bytes("image.png", buf.getvalue(), purpose)

    @staticmethod
    def _upload_folder(client, folder: str) -> list:
        ids = []
        if folder and os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    ids.append(client.upload_file(os.path.join(folder, f), "reference"))
        return ids

    @staticmethod
    def image_to_b64(image) -> "str | None":
        """Encode a PIL image as a base64 PNG string for image_b64 fields."""
        if image is None:
            return None
        import io
        import base64
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ---- custom input UI (multi-image refs + video audio reference) -------

    def draw_custom_ui(self, col, context) -> bool:
        """Draw multi-reference-image pickers and/or a video audio reference.

        Reuses the existing flux-style strip pickers (scene.flux_strip_N +
        flux_visible_strips, the strip_picker / flux_add_strip / flux_hide_strip
        operators). Returns True only when it took over the input area; otherwise
        False so the standard single-strip selector is drawn by the panel.
        """
        flags = self.INPUTS
        has_multi = InputSpec.MULTI_IMAGE in flags
        has_audio_ref = (InputSpec.AUDIO_REF in flags) and self.MODEL_TYPE == "video"
        if not (has_multi or has_audio_ref):
            return False

        scene = context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:  # noqa: BLE001
            pass

        if has_multi and scene.sequence_editor is not None:
            n = max(1, min(9, int(getattr(self.PARAMS, "max_multi_images", 1))))
            vis = max(1, min(int(getattr(scene, "flux_visible_strips", 1)), n))
            for i in range(1, vis + 1):
                row = col.row(align=True)
                row.prop_search(scene, f"flux_strip_{i}", scene.sequence_editor,
                                "strips", text="Ref.", icon="FILE_IMAGE")
                row.operator("sequencer.strip_picker", text="",
                             icon="EYEDROPPER").action = f"flux_select{i}"
                if i == vis and vis < n:
                    if vis > 1:
                        row.operator("object.flux_hide_strip", text="",
                                     icon="REMOVE").strip_index = i
                    row.operator("object.flux_add_strip", text="", icon="ADD")

        if has_audio_ref:
            row = col.row(align=True)
            row.prop(scene, "ref_audio_path", text="Audio Ref.")
            row.operator("sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER")

        return True

    @staticmethod
    def _collect_multi_images(inputs: ModelInputs, scene) -> None:
        """Populate inputs.images from the selected flux_strip_N reference strips."""
        from ..utils.helpers import find_strip_by_name, load_strip_as_pil
        imgs = []
        if inputs.image is not None:
            imgs.append(inputs.image)
        for i in range(1, 10):
            name = getattr(scene, f"flux_strip_{i}", "") or ""
            if not name:
                continue
            strip = find_strip_by_name(scene, name)
            if strip is None:
                continue
            try:
                pil = load_strip_as_pil(strip)
            except Exception as e:  # noqa: BLE001
                print(f"[Remote] could not load reference strip {name!r}: {e}")
                continue
            if pil is not None:
                imgs.append(pil)
        if imgs:
            inputs.images = imgs


# ---------------------------------------------------------------------------
# Factory — build a synthetic plugin from a /v1/models discovery entry
# ---------------------------------------------------------------------------

def _derive_ui(mtype: str, modes: list, entry: dict):
    """Return (INPUTS, UI_SECTIONS) for a model from its type + modes + hints."""
    modes = modes or []
    max_ref = int(entry.get("max_ref_images", 0))

    if mtype == "image":
        inputs = InputSpec.PROMPT | InputSpec.NEG_PROMPT
        sections = [UISection.PROMPT, UISection.NEG_PROMPT, UISection.RESOLUTION,
                    UISection.STEPS, UISection.GUIDANCE, UISection.SEED]
        if "i2i" in modes or max_ref >= 1:
            inputs |= InputSpec.IMAGE
            sections.insert(2, UISection.IMAGE_STRIP)
            sections.append(UISection.IMAGE_STRENGTH)
        if max_ref > 1:
            inputs |= InputSpec.MULTI_IMAGE
            sections.insert(3, UISection.MULTI_IMAGES)
        return inputs, sections

    if mtype == "video":
        inputs = InputSpec.PROMPT | InputSpec.NEG_PROMPT
        sections = [UISection.PROMPT, UISection.NEG_PROMPT, UISection.RESOLUTION,
                    UISection.FRAMES, UISection.SEED]
        if "i2v" in modes:
            inputs |= InputSpec.IMAGE
            sections.insert(2, UISection.IMAGE_STRIP)
        if max_ref > 1:
            # Multiple reference images (e.g. Seedance reference-to-video):
            # drawn with the flux-style pickers in draw_custom_ui.
            inputs |= InputSpec.MULTI_IMAGE
            sections.insert(3, UISection.MULTI_IMAGES)
        if "control" in modes:
            inputs |= InputSpec.VIDEO
            sections.insert(3, UISection.VIDEO_STRIP)
        if entry.get("needs_audio_ref"):
            inputs |= InputSpec.AUDIO_REF        # reference audio (audio_urls)
        if entry.get("supports_audio_output"):
            sections.append(UISection.AUDIO_OUTPUT)
        return inputs, sections

    if mtype == "audio":
        inputs = InputSpec.PROMPT
        sections = [UISection.PROMPT, UISection.SPEED, UISection.SEED]
        if entry.get("needs_speaker_ref"):
            inputs |= InputSpec.AUDIO_REF
            sections.insert(1, UISection.AUDIO_REF)
        if entry.get("needs_ref_text"):
            inputs |= InputSpec.TEXT_REF
            sections.insert(2, UISection.TEXT_REF)
        return inputs, sections

    # text / transcription: takes an audio/video strip, returns text.
    return InputSpec.VIDEO, [UISection.VIDEO_STRIP]


def _derive_params(mtype: str, entry: dict) -> ParamSpec:
    """Seed ParamSpec defaults from optional hints, falling back to sane values."""
    kw = {}
    if "default_steps" in entry:
        kw["steps"] = int(entry["default_steps"])
    if "default_guidance" in entry:
        kw["guidance"] = float(entry["default_guidance"])
    if "max_width" in entry:
        kw["width"] = int(entry["max_width"])
    if "max_height" in entry:
        kw["height"] = int(entry["max_height"])
    if "max_ref_images" in entry:
        kw["max_multi_images"] = max(1, int(entry["max_ref_images"]))
    return ParamSpec(**kw)


def make_remote_plugin(entry: dict) -> "RemoteModelPlugin":
    """Build a configured RemoteModelPlugin instance from a /v1/models entry.

    entry = {"id", "type", optional "modes", "display_name", "description",
             and optional capability hints: max_ref_images, needs_speaker_ref,
             needs_ref_text, control_types, default_steps, ...}
    """
    mid = entry.get("id")
    if not mid:
        raise ValueError("model entry missing 'id'")
    mtype = entry.get("type", "image")
    if mtype not in _ENDPOINTS:
        mtype = "image"
    modes = entry.get("modes") or []
    endpoint, ext = _ENDPOINTS[mtype]
    inputs, sections = _derive_ui(mtype, modes, entry)

    inst = RemoteModelPlugin()
    inst.MODEL_ID = f"remote:{mid}"
    inst.REMOTE_MODEL_NAME = mid
    inst.DISPLAY_NAME = f"[Remote] {entry.get('display_name', mid)}"
    inst.MODEL_TYPE = mtype
    inst.DESCRIPTION = entry.get("description", f"Remote {mtype} model '{mid}'")
    inst.ENDPOINT = endpoint
    inst.OUTPUT_EXT = ext
    inst.INPUTS = inputs
    inst.UI_SECTIONS = sections
    inst.PARAMS = _derive_params(mtype, entry)

    # Capability hints used by upload_inputs / draw_custom_ui.
    inst._max_ref_images = int(entry.get("max_ref_images", 0))
    inst._needs_speaker_ref = bool(entry.get("needs_speaker_ref", False))
    inst._needs_ref_text = bool(entry.get("needs_ref_text", False))
    inst._control_types = list(entry.get("control_types", []))
    inst._supports_audio_output = bool(entry.get("supports_audio_output", False))

    # Models with multiple reference images, or a video-side audio reference,
    # draw their input area in draw_custom_ui (flux-style pickers) rather than
    # the single standard input-strip selector.
    needs_custom = (InputSpec.MULTI_IMAGE in inputs) or (
        (InputSpec.AUDIO_REF in inputs) and mtype == "video")
    inst.uses_standard_input_strip = (not needs_custom) and bool(
        (InputSpec.IMAGE in inputs) or (InputSpec.VIDEO in inputs)
    )
    inst.requires_input_strip = (mtype == "text")
    return inst
