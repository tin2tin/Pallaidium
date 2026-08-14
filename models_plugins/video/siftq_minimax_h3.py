"""SiftQ MiniMax-H3 V2 cloud video generation.

Provider identity, transport, configuration, request mapping, polling and tests
are intentionally independent from pre-existing cloud-provider plugins.
``MiniMax-H3`` is retained only because it is protocol data required by the
SiftQ-compatible upstream contract.
"""

from __future__ import annotations

import os
import time

from ...models.base import ModelPlugin, ModelInputs, InputSpec, UISection, ParamSpec
from ...utils.siftq_h3_v2 import (
    DEFAULT_BASE_URL,
    MODEL_ID as SIFTQ_API_MODEL_ID,
    SiftQClient,
    SiftQError,
    SiftQValidationError,
    build_video_payload,
    usage_note,
)


SIFTQ_PROVIDER_SLUG = "siftq/minimax-h3"
SIFTQ_MAX_REFERENCE_IMAGES = 9
_REF_ATTRS = [f"siftq_ref_strip_{index}" for index in range(1, SIFTQ_MAX_REFERENCE_IMAGES + 1)]


def _media_item(kind: str, url: str, role: str) -> dict:
    item_type = f"{kind}_url"
    return {"type": item_type, item_type: {"url": url}, "role": role}


class SiftQMiniMaxH3Plugin(ModelPlugin):
    MODEL_ID = SIFTQ_PROVIDER_SLUG
    DISPLAY_NAME = "SiftQ MiniMax H3 (cloud)"
    MODEL_TYPE = "video"
    DESCRIPTION = (
        "SiftQ MiniMax-H3 V2: text, first-frame, first+last-frame and reference video generation"
    )

    INPUTS = (
        InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.VIDEO |
        InputSpec.AUDIO_REF | InputSpec.API_KEY
    )
    UI_SECTIONS = [UISection.PROMPT]
    PARAMS = ParamSpec(width=1024, height=576, frames=120, steps=1, guidance=1.0)
    # The plugin itself is available without heavyweight add-on dependencies.
    # PyAV is used opportunistically for deep timed-media inspection; the
    # transport has a stdlib fallback for valid MP4/MOV/WAV/MP3 references.
    REQUIRED_PACKAGES = []

    supports_inpaint = False
    supports_img2img = True
    uses_standard_input_strip = False
    uses_strip_power = False
    show_enhance = False
    supports_batch = False
    preserve_image_dimensions = True

    def load(self, prefs, scene, **kw):
        # The client is rebuilt in generate() so runtime environment changes win.
        return {"pipe": None, "last_model_card": self.MODEL_ID}

    @staticmethod
    def _client_from_env() -> SiftQClient:
        try:
            import bpy
        except ImportError:
            bpy = None
        if bpy is not None and hasattr(bpy.app, "online_access") and not bpy.app.online_access:
            raise SiftQValidationError(
                "Blender Online Access is disabled; enable it before using SiftQ."
            )
        key = os.environ.get("SIFTQ_API_KEY", "").strip()
        base_url = os.environ.get("SIFTQ_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        request_timeout = SiftQMiniMaxH3Plugin._poll_setting(
            "SIFTQ_REQUEST_TIMEOUT_SECONDS", 180.0, minimum=1.0,
        )
        return SiftQClient(api_key=key, base_url=base_url, timeout=request_timeout)

    @staticmethod
    def _poll_setting(name: str, default: float, *, minimum: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError as exc:
            raise SiftQValidationError(f"{name} must be numeric.") from exc
        if value < minimum:
            raise SiftQValidationError(f"{name} must be at least {minimum:g}.")
        return value

    # ---- UI -------------------------------------------------------------
    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        mode = getattr(scene, "siftq_mode", "TEXT")
        if mode == "TEXT":
            return False
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene

        if mode in {"FIRST_FRAME", "FIRST_LAST"}:
            count = 2 if mode == "FIRST_LAST" else 1
            labels = ("First Frame", "Last Frame")
            if vse_scene.sequence_editor is not None:
                for index, attr in enumerate(_REF_ATTRS[:count], 1):
                    row = col.row(align=True)
                    row.prop_search(
                        vse_scene, attr, vse_scene.sequence_editor, "strips",
                        text=labels[index - 1], icon="FILE_IMAGE",
                    )
                    row.operator(
                        "sequencer.strip_picker", text="", icon="EYEDROPPER"
                    ).action = f"siftq_select{index}"
            else:
                for index, attr in enumerate(_REF_ATTRS[:count]):
                    col.prop(vse_scene, attr, text=labels[index])
            col.label(text="Pick image strips above, or use the selected timeline strip", icon="INFO")
            return False

        if mode == "REFERENCE":
            col.prop(scene, "siftq_ref_count")
            if vse_scene.sequence_editor is None:
                return False
            count = max(1, min(int(getattr(scene, "siftq_ref_count", 3)), len(_REF_ATTRS)))
            for index, attr in enumerate(_REF_ATTRS[:count], 1):
                row = col.row(align=True)
                row.prop_search(
                    vse_scene, attr, vse_scene.sequence_editor, "strips",
                    text="Ref. Image", icon="FILE_IMAGE",
                )
                row.operator(
                    "sequencer.strip_picker", text="", icon="EYEDROPPER"
                ).action = f"siftq_select{index}"
            col.label(text="Selected main IMAGE + picker images must total at most 9", icon="INFO")
            col.label(text="For reference video: set Input to Strips and select one MOVIE", icon="INFO")
        row = col.row(align=True)
        row.prop(scene, "ref_audio_path", text="Ref. Audio")
        row.operator("sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER")
        return False

    def draw_post_seed_ui(self, col, context):
        scene = context.scene
        mode = getattr(scene, "siftq_mode", "TEXT")
        col.prop(scene, "siftq_api_key_session", text="Session API Key")
        if os.environ.get("SIFTQ_API_KEY", "").strip():
            col.label(text="SiftQ key is active for this Blender session", icon="CHECKMARK")
        else:
            col.label(text="Enter a SiftQ key to enable generation", icon="LOCKED")
        col.prop(scene, "siftq_mode")
        col.prop(scene, "siftq_resolution")
        col.prop(scene, "siftq_duration")
        if mode in {"FIRST_FRAME", "FIRST_LAST"}:
            col.label(text="Aspect: Adaptive (from frame)")
        elif mode == "REFERENCE":
            col.prop(scene, "siftq_reference_ratio")
        else:
            col.prop(scene, "siftq_ratio")
        col.label(text="The session key is not saved in .blend files", icon="INFO")

    # ---- Mapping --------------------------------------------------------
    @staticmethod
    def _reference_count(scene) -> int:
        return max(1, min(int(getattr(scene, "siftq_ref_count", 3)), SIFTQ_MAX_REFERENCE_IMAGES))

    @staticmethod
    def _picker_image(scene, attr: str, client: SiftQClient):
        """Return (data_uri, dedupe_key) for one explicitly configured picker.

        A named picker is an intent-critical input. If queue-time rendering or
        later file resolution failed, surface that failure instead of silently
        dropping the image while another reference keeps the request valid.
        """
        raw_path = getattr(scene, attr + "_path", "") or ""
        path = os.path.abspath(raw_path) if raw_path else ""
        name = getattr(scene, attr, "") or ""
        if path:
            if not os.path.isfile(path):
                raise SiftQValidationError(
                    f"SiftQ reference picker {attr!r} points to a missing file."
                )
            key = "path:" + os.path.normcase(path)
            return client.data_uri_from_file(path, "image"), key
        if not name:
            return None, None

        # Interactive fallback outside the render queue.  An IMAGE picker means
        # "use this image file", not "render this strip inside the project
        # canvas".  Prefer the source path here too, otherwise Blender's
        # auto-FIT transform can be baked into a native-size render and create
        # black padding around portrait inputs.
        try:
            from ...utils.helpers import find_strip_by_name, get_strip_path, load_strip_as_pil
            strip = find_strip_by_name(scene, name)
        except Exception:
            strip = None
        if strip is not None and strip.type == "IMAGE":
            try:
                source_path = get_strip_path(strip) or ""
                source_path = os.path.abspath(source_path) if source_path else ""
            except Exception:
                source_path = ""
            if source_path and os.path.isfile(source_path):
                key = "path:" + os.path.normcase(source_path)
                # Preserve validation errors from the SiftQ client instead of
                # hiding them behind the rendered-strip fallback.
                return client.data_uri_from_file(source_path, "image"), key
        try:
            image = load_strip_as_pil(strip) if strip is not None else None
        except Exception:
            image = None
        if image is None:
            raise SiftQValidationError(
                f"Could not resolve SiftQ reference picker {attr!r} ({name!r}) to an image."
            )
        return client.pil_image_to_data_uri(image), "strip:" + name

    def _collect_picker_images(self, scene, client: SiftQClient,
                               excluded_paths: set[str] | None = None) -> list[str]:
        refs = []
        seen_paths = {
            "path:" + os.path.normcase(os.path.abspath(path))
            for path in (excluded_paths or set()) if path
        }
        count = self._reference_count(scene)
        for attr in _REF_ATTRS[:count]:
            image_url, key = self._picker_image(scene, attr, client)
            if image_url is not None and key not in seen_paths:
                refs.append(image_url)
                seen_paths.add(key)
        return refs

    def build_request(self, inputs: ModelInputs, scene, client: SiftQClient) -> dict:
        mode = getattr(scene, "siftq_mode", "TEXT")
        resolution = getattr(scene, "siftq_resolution", "768P")
        duration = int(getattr(scene, "siftq_duration", 5))
        text_ratio = getattr(scene, "siftq_ratio", "16:9")
        reference_ratio = getattr(scene, "siftq_reference_ratio", text_ratio)
        media = []

        if mode == "TEXT":
            ratio = text_ratio
        elif mode == "FIRST_FRAME":
            picked_first, _ = self._picker_image(scene, _REF_ATTRS[0], client)
            if picked_first is not None:
                first_url = picked_first
            elif getattr(scene, "image_path", "") and os.path.isfile(scene.image_path):
                first_url = client.data_uri_from_file(scene.image_path, "image")
            elif getattr(scene, "image_path", ""):
                raise SiftQValidationError("SiftQ selected first-frame input file is missing.")
            elif inputs.image is not None:
                first_url = client.pil_image_to_data_uri(inputs.image)
            else:
                raise SiftQValidationError(
                    "SiftQ First Frame mode requires a First Frame picker or an IMAGE, MOVIE, "
                    "SCENE or META input strip."
                )
            media.append(_media_item("image", first_url, "first_frame"))
            ratio = "adaptive"
        elif mode == "FIRST_LAST":
            picked_first, _ = self._picker_image(scene, _REF_ATTRS[0], client)
            picked_last, _ = self._picker_image(scene, _REF_ATTRS[1], client)
            has_picked_first = picked_first is not None
            has_picked_last = picked_last is not None
            if has_picked_first or has_picked_last:
                if not (has_picked_first and has_picked_last):
                    raise SiftQValidationError(
                        "SiftQ First + Last Frame mode requires both picker fields."
                    )
                first_url = picked_first
                last_url = picked_last
            elif (getattr(scene, "image_path", "") and os.path.isfile(scene.image_path)
                  and getattr(scene, "last_image_path", "")
                  and os.path.isfile(scene.last_image_path)):
                first_url = client.data_uri_from_file(scene.image_path, "image")
                last_url = client.data_uri_from_file(scene.last_image_path, "image")
            elif inputs.image is not None and inputs.last_image is not None:
                first_url = client.pil_image_to_data_uri(inputs.image)
                last_url = client.pil_image_to_data_uri(inputs.last_image)
            else:
                raise SiftQValidationError(
                    "SiftQ First + Last Frame mode requires both picker fields or a META strip "
                    "containing two ordered images."
                )
            media.append(_media_item("image", first_url, "first_frame"))
            media.append(_media_item("image", last_url, "last_frame"))
            ratio = "adaptive"
        elif mode == "REFERENCE":
            # A selected MOVIE is a reference video. The queue also extracts its
            # first frame into inputs.image; do not duplicate that frame as an
            # image reference. A selected IMAGE is one reference image.
            excluded_picker_paths = set()
            main_image_path = getattr(scene, "image_path", "") or ""
            if inputs.video_path:
                media.append(_media_item(
                    "video", client.data_uri_from_file(inputs.video_path, "video"), "reference_video"
                ))
            elif main_image_path and os.path.isfile(main_image_path):
                media.append(_media_item(
                    "image", client.data_uri_from_file(main_image_path, "image"), "reference_image"
                ))
                excluded_picker_paths.add(main_image_path)
            elif main_image_path:
                raise SiftQValidationError("SiftQ selected reference image file is missing.")
            elif inputs.image is not None:
                media.append(_media_item(
                    "image", client.pil_image_to_data_uri(inputs.image), "reference_image"
                ))
            picker_images = self._collect_picker_images(
                scene, client, excluded_paths=excluded_picker_paths,
            )
            existing_image_count = sum(item.get("role") == "reference_image" for item in media)
            if existing_image_count + len(picker_images) > SIFTQ_MAX_REFERENCE_IMAGES:
                raise SiftQValidationError(
                    "SiftQ accepts at most 9 reference images total. The selected main IMAGE "
                    "counts as one; reduce the reference-image picker count to 8 or fewer."
                )
            for image_url in picker_images:
                media.append(_media_item("image", image_url, "reference_image"))
            if inputs.audio_ref:
                media.append(_media_item(
                    "audio", client.data_uri_from_file(inputs.audio_ref, "audio"), "reference_audio"
                ))
            if not media:
                raise SiftQValidationError(
                    "SiftQ Reference mode requires at least one image, video or audio reference."
                )
            ratio = reference_ratio
        else:
            raise SiftQValidationError(f"Unsupported SiftQ mode: {mode!r}.")

        return build_video_payload(
            prompt=inputs.prompt,
            media=media,
            resolution=resolution,
            duration=duration,
            ratio=ratio,
        )

    # ---- Generation -----------------------------------------------------
    @staticmethod
    def _task_snapshot(client: SiftQClient) -> set[str] | None:
        """Snapshot recent generation IDs for safe create-timeout recovery."""
        try:
            result = client.list_tasks(
                page_num=1, page_size=100, model=SIFTQ_API_MODEL_ID, task_type="generation",
            )
        except SiftQError:
            return None
        return {
            task["id"] for task in result.get("items", [])
            if isinstance(task, dict) and isinstance(task.get("id"), str) and task["id"]
        }

    def _create_video_safely(self, client: SiftQClient, payload: dict, inputs: ModelInputs) -> str:
        """Create once; recover a lost response without resubmitting billable work."""
        known_ids = self._task_snapshot(client)
        submitted_at = int(time.time())
        try:
            return client.create_video(payload)
        except SiftQError as create_error:
            if not create_error.delivery_uncertain:
                raise
            if known_ids is None:
                raise SiftQError(
                    "SiftQ submission response was lost and the pre-submit task snapshot was "
                    "unavailable. The task may still exist; check the SiftQ task list before "
                    "retrying to avoid duplicate charges."
                ) from create_error

            self.set_phase(inputs, "Recovering SiftQ submission")
            recovery_seconds = self._poll_setting(
                "SIFTQ_SUBMIT_RECOVERY_SECONDS", 120.0, minimum=1.0,
            )
            deadline = time.monotonic() + recovery_seconds
            while True:
                if inputs.should_cancel is not None and inputs.should_cancel():
                    raise KeyboardInterrupt(
                        "SiftQ submission recovery cancelled locally; the upstream task may continue."
                    )
                try:
                    result = client.list_tasks(
                        page_num=1, page_size=100, model=SIFTQ_API_MODEL_ID,
                        task_type="generation",
                    )
                except SiftQError:
                    result = {"items": []}

                now = int(time.time())
                candidates = []
                for task in result.get("items", []):
                    if not isinstance(task, dict) or task.get("id") in known_ids:
                        continue
                    created_at = task.get("created_at")
                    if (isinstance(created_at, bool) or not isinstance(created_at, int)
                            or not submitted_at - 120 <= created_at <= now + 120):
                        continue
                    if task.get("model") not in {None, "", SIFTQ_API_MODEL_ID}:
                        continue
                    if task.get("task_type") not in {None, "", "generation"}:
                        continue
                    if task.get("modality") not in {None, "", "video"}:
                        continue
                    if task.get("resolution") not in {None, "", payload.get("resolution")}:
                        continue
                    if task.get("duration") not in {None, payload.get("duration")}:
                        continue
                    candidates.append(task)

                if len(candidates) == 1:
                    task_id = candidates[0]["id"]
                    print("[SiftQ] recovered the unique task created during the timed-out submission")
                    return task_id
                if len(candidates) > 1:
                    raise SiftQError(
                        "SiftQ submission response was lost and multiple new tasks appeared. "
                        "No task was selected; check the SiftQ task list before retrying."
                    ) from create_error
                if time.monotonic() >= deadline:
                    raise SiftQError(
                        "SiftQ submission response was lost and no unique new task could be "
                        "identified. The task may still exist; check the SiftQ task list before "
                        "retrying to avoid duplicate charges."
                    ) from create_error
                time.sleep(min(2.0, max(0.0, deadline - time.monotonic())))

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        client = self._client_from_env()
        payload = self.build_request(inputs, scene, client)
        media_roles = [
            item.get("role")
            for item in payload.get("content", [])
            if isinstance(item, dict) and item.get("type") != "text"
        ]
        print(
            f"[SiftQ] request ready: mode={getattr(scene, 'siftq_mode', 'TEXT')} "
            f"media_roles={media_roles!r} resolution={payload['resolution']} "
            f"duration={payload['duration']} ratio={payload['ratio']}"
        )
        poll_timeout = self._poll_setting("SIFTQ_POLL_TIMEOUT_SECONDS", 3600.0, minimum=1.0)
        poll_interval = self._poll_setting("SIFTQ_POLL_INTERVAL_SECONDS", 2.0, minimum=0.1)

        self.set_phase(inputs, "Submitting to SiftQ")
        task_id = self._create_video_safely(client, payload, inputs)
        self.set_phase(inputs, "Waiting for SiftQ")
        task = client.poll_task(
            task_id,
            max_wait=poll_timeout,
            interval=poll_interval,
            should_cancel=inputs.should_cancel,
            phase_fn=inputs.phase_fn,
            progress_fn=inputs.progress_fn,
        )
        inputs.usage_note = usage_note(task)

        self.set_phase(inputs, "Downloading SiftQ video")
        from ...utils.helpers import clean_filename, solve_path
        stem = clean_filename("siftq_" + ((inputs.prompt or "video")[:40])) or "siftq_video"
        destination = solve_path(stem + ".mp4")
        return client.download_task_video(task_id, task, destination)
