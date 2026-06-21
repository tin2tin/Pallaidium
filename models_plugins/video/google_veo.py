"""Cloud text/image-to-video via Google Veo.

Three selectable variants (scene.veo_model):
  - veo-3.1-generate-preview       (Veo 3.1)
  - veo-3.1-fast-generate-preview  (Veo 3.1 Fast)
  - veo-3.0-generate-preview       (Veo 3.0)

The API key comes from the addon preference ``gemini_api_key`` (falls back to the
GEMINI_API_KEY environment variable).  A selected image strip is used as the first
frame; a META strip bundling two images supplies first + last frame for
interpolation (Veo 3.1).  Runs in Pallaidium's background worker thread, so
synchronous polling is safe.
"""

import os
import io
import time
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename, find_strip_by_name, load_strip_as_pil

_REF_ATTRS = ["veo_ref_strip_1", "veo_ref_strip_2", "veo_ref_strip_3"]


class GoogleVeoPlugin(ModelPlugin):
    MODEL_ID     = "google/veo"
    DISPLAY_NAME = "Video: Google Veo (cloud)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Cloud text/image-to-video via the Google Veo API"

    INPUTS       = (
        InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.API_KEY
    )
    # No SEED section: the Gemini Developer API (API-key path) rejects a seed for Veo.
    UI_SECTIONS  = [UISection.PROMPT, UISection.NEG_PROMPT]
    PARAMS       = ParamSpec(width=1280, height=720, steps=1, guidance=1.0)
    REQUIRED_PACKAGES = ["google.genai"]

    supports_inpaint = False
    supports_img2img = True    # first-frame conditioning when a strip is supplied
    uses_strip_power = False    # Veo has no img2img "strength" parameter
    supports_batch   = True     # batch count → multiple separate API calls

    # Optional config fields safe to silently drop-and-retry when a given
    # model/mode rejects them (pure tuning knobs).  Intent-critical inputs such
    # as ``reference_images``/``last_frame`` are deliberately excluded: if the
    # server refuses them we surface the real error instead of quietly producing
    # output that ignores what the user asked for.
    _DROPPABLE = ("person_generation", "negative_prompt", "enhance_prompt")

    # ---- UI ---------------------------------------------------------------
    @staticmethod
    def _supports_reference_images(model: str) -> bool:
        # Reference "ingredients" images are a Veo 3.1 feature; 3.0 cannot use them.
        return model.startswith("veo-3.1")

    def draw_custom_ui(self, col, context) -> bool:
        # Klein-style reference-image pickers (Veo 3.1 "ingredients", used in
        # Reference image mode).  Returns False so the standard input strip UI
        # (first/last frame via timeline selection) still renders.  Hidden for
        # Veo 3.0, which does not support reference images.
        scene = context.scene
        if scene.sequence_editor is None:
            return False
        if not self._supports_reference_images(
            getattr(scene, "veo_model", "veo-3.1-fast-generate-preview")
        ):
            return False
        for i, attr in enumerate(_REF_ATTRS, 1):
            row = col.row(align=True)
            row.prop_search(scene, attr, scene.sequence_editor, "strips",
                            text="Ref.", icon="FILE_IMAGE")
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"veo_select{i}"
        return False

    def draw_post_seed_ui(self, col, context):
        scene = context.scene
        col.prop(scene, "veo_model")
        col.prop(scene, "veo_image_mode")
        col.prop(scene, "veo_aspect")
        col.prop(scene, "veo_resolution")
        col.prop(scene, "veo_duration")
        col.prop(scene, "veo_person_generation")

    # ---- Lifecycle --------------------------------------------------------
    def load(self, prefs, scene, **kw):
        return {"pipe": None, "last_model_card": self.MODEL_ID}

    def _get_api_key(self, prefs):
        key = getattr(prefs, "gemini_api_key", "") or ""
        key = key.strip()
        if not key:
            key = os.environ.get("GEMINI_API_KEY", "").strip()
        return key

    @staticmethod
    def _friendly_error(e) -> RuntimeError:
        """Translate a google.genai API error into a concise, actionable message."""
        code = getattr(e, "code", None)
        msg = (getattr(e, "message", None) or str(e)).strip()
        if code == 429:
            return RuntimeError(
                "Google API quota exceeded (429). Veo is not on the free tier (limit 0) "
                "or you hit a rate limit — enable billing on your Google AI Studio / Cloud "
                "project, or wait and retry. " + msg[:200]
            )
        if code in (401, 403):
            return RuntimeError(
                f"Google API auth error ({code}). Check the Gemini API key in the "
                "add-on preferences (and that the project has the API enabled). " + msg[:160]
            )
        if code:
            return RuntimeError(f"Google API error {code}: {msg[:300]}")
        return RuntimeError(str(e)[:400])

    @staticmethod
    def _match_optional_field(opts, msg) -> str | None:
        """Return an optional-config key named in an error message, else None.

        Strips all non-alphanumerics from both sides so a key like
        'negative_prompt' matches 'Negative prompt is not supported…' and
        'enhance_prompt' matches '`enhancePrompt` isn't supported…'.
        """
        import re
        norm = re.sub(r"[^a-z0-9]", "", str(msg).lower())
        return next(
            (k for k in opts if re.sub(r"[^a-z0-9]", "", k.lower()) in norm), None
        )

    @staticmethod
    def _pil_to_image(types, pil_image):
        """Convert a PIL image to a google.genai types.Image (PNG bytes)."""
        if pil_image is None:
            return None
        buf = io.BytesIO()
        pil_image.convert("RGB").save(buf, format="PNG")
        return types.Image(image_bytes=buf.getvalue(), mime_type="image/png")

    def _collect_ref_pils(self, scene):
        """PIL reference images from the picker strips (path → name fallback)."""
        from PIL import Image
        refs = []
        for attr in _REF_ATTRS:
            img = None
            path = getattr(scene, attr + "_path", "") or ""
            if path and os.path.isfile(path):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    print(f"[Google Veo] Could not open reference {path!r}: {e}")
            if img is None:
                name = getattr(scene, attr, "") or ""
                if name:
                    strip = find_strip_by_name(scene, name)
                    if strip:
                        try:
                            img = load_strip_as_pil(strip)
                        except Exception as e:
                            print(f"[Google Veo] Could not load reference strip {name!r}: {e}")
            if img is not None:
                refs.append(img)
        return refs

    # ---- Generation -------------------------------------------------------
    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        from google import genai
        from google.genai import types
        from google.genai import errors as genai_errors

        api_key = self._get_api_key(prefs)
        if not api_key:
            raise RuntimeError(
                "Google Gemini API key is missing. Set it in the Pallaidium add-on "
                "preferences (Google Gemini API Key) or the GEMINI_API_KEY env var."
            )

        model        = getattr(scene, "veo_model", "veo-3.1-fast-generate-preview")
        aspect       = getattr(scene, "veo_aspect", "16:9")
        resolution   = getattr(scene, "veo_resolution", "720p")
        duration     = int(getattr(scene, "veo_duration", "8"))
        person_gen   = getattr(scene, "veo_person_generation", "allow_adult")
        image_mode   = getattr(scene, "veo_image_mode", "AUTO")

        # 1080p is only valid for 16:9 — fall back to 720p for portrait.
        if resolution == "1080p" and aspect != "16:9":
            print("[Google Veo] 1080p is only supported for 16:9; using 720p.")
            resolution = "720p"

        # ── Resolve effective image mode ───────────────────────────────────
        refs_ok = self._supports_reference_images(model)
        ref_pils = (
            self._collect_ref_pils(scene)
            if refs_ok and image_mode in ("AUTO", "REFERENCE") else []
        )
        if image_mode == "REFERENCE" and not refs_ok:
            print(f"[Google Veo] {model} does not support reference images; "
                  "falling back to text/first-frame.")
            image_mode = "AUTO"
        if image_mode == "AUTO":
            if ref_pils:
                image_mode = "REFERENCE"
            elif inputs.last_image is not None:
                image_mode = "INTERPOLATE"
            elif inputs.image is not None:
                image_mode = "FIRST"
            else:
                image_mode = "TEXT"
        print(f"[Google Veo] image mode: {image_mode}")

        client = genai.Client(api_key=api_key)

        # ── Build optional config fields ──────────────────────────────────
        # The Gemini Developer API (the API-key path used here) rejects several
        # Vertex/Enterprise-only fields (seed, generate_audio, enhance_prompt),
        # so they are not sent. Veo 3 generates audio by default on this API.
        optional = {
            "person_generation": person_gen,
        }
        if inputs.neg_prompt:
            optional["negative_prompt"] = inputs.neg_prompt

        first_frame = None
        if image_mode == "REFERENCE" and ref_pils:
            # Veo 3.1 "ingredients to video" — subject/style reference images.
            ref_objs = []
            for _pil in ref_pils:
                _imgobj = self._pil_to_image(types, _pil)
                try:
                    ref_objs.append(types.VideoGenerationReferenceImage(
                        image=_imgobj, reference_type="asset"))
                except Exception:
                    ref_objs.append(_imgobj)  # older SDK: pass raw image
            optional["reference_images"] = ref_objs
            print(f"[Google Veo] sending {len(ref_objs)} reference image(s) ({model})")
        elif image_mode == "INTERPOLATE":
            first_frame = self._pil_to_image(types, inputs.image)
            last_frame = self._pil_to_image(types, inputs.last_image)
            if last_frame is not None:
                optional["last_frame"] = last_frame
        elif image_mode == "FIRST":
            first_frame = self._pil_to_image(types, inputs.image)

        def _build_config(opts):
            cfg = types.GenerateVideosConfig(
                aspect_ratio=aspect, resolution=resolution,
                duration_seconds=duration, number_of_videos=1,
            )
            for _k, _v in opts.items():
                try:
                    setattr(cfg, _k, _v)
                except Exception:
                    print(f"[Google Veo] config field {_k!r} not supported here; skipped.")
            return cfg

        gen_kwargs = dict(model=model, prompt=inputs.prompt)
        if first_frame is not None:
            gen_kwargs["image"] = first_frame

        self.set_phase(inputs, "Submitting to cloud")
        # Submit, dropping any optional config field the API rejects for this mode.
        # Vertex-only fields (seed, generate_audio, enhance_prompt, …) are refused
        # by the Developer API either at request-build (ValueError) or by the
        # server (400 INVALID_ARGUMENT) — handle both and retry without the field.
        attempt_opts = dict(optional)
        operation = None
        while True:
            try:
                operation = client.models.generate_videos(
                    config=_build_config(attempt_opts), **gen_kwargs
                )
                break
            except genai_errors.APIError as e:
                _dropped = (
                    self._match_optional_field(attempt_opts, getattr(e, "message", "") or str(e))
                    if getattr(e, "code", None) == 400 else None
                )
                if _dropped not in self._DROPPABLE:
                    raise self._friendly_error(e)
                print(f"[Google Veo] dropping unsupported field {_dropped!r}: {e}")
                del attempt_opts[_dropped]
            except (ValueError, TypeError) as e:
                _dropped = self._match_optional_field(attempt_opts, str(e))
                if _dropped not in self._DROPPABLE:
                    raise self._friendly_error(e)
                print(f"[Google Veo] dropping unsupported field {_dropped!r}: {e}")
                del attempt_opts[_dropped]

        try:
            print(f"[Google Veo] Job submitted ({model}); generation can take a few minutes.")

            # ── Poll (safe in the background worker thread) ────────────────
            self.set_phase(inputs, "Waiting for cloud")
            while not operation.done:
                time.sleep(20)
                operation = client.operations.get(operation)
                print("[Google Veo] Still generating...")
        except genai_errors.APIError as e:
            raise self._friendly_error(e)

        if getattr(operation, "error", None):
            raise RuntimeError(f"Veo API error: {operation.error}")

        response = getattr(operation, "response", None)
        videos = getattr(response, "generated_videos", None) or []
        if not videos:
            raise RuntimeError("Veo API did not return any video data.")

        # ── Download + save ────────────────────────────────────────────────
        self.set_phase(inputs, "Saving")
        generated = videos[0]
        dst_path = solve_path(
            clean_filename(str(inputs.seed) + "_" + (inputs.prompt[:30] or "veo")) + ".mp4"
        )
        try:
            client.files.download(file=generated.video)
            generated.video.save(dst_path)
        except genai_errors.APIError as e:
            raise self._friendly_error(e)

        if not os.path.exists(dst_path):
            raise RuntimeError("Veo video download failed.")
        print(f"[Google Veo] Downloaded to {dst_path}")
        return dst_path
