"""Cloud text-to-image / image-editing via Google Gemini ("Nano Banana").

Three selectable variants (scene.nano_banana_model):
  - gemini-2.5-flash-image      (Nano Banana, fast)
  - gemini-3-pro-image-preview  (Nano Banana Pro, up to 4K)
  - imagen-4.0-generate-001     (Imagen 4)

The API key comes from the addon preference ``gemini_api_key`` (falls back to the
GEMINI_API_KEY environment variable).  Image strips (including META strips bundling
first/last/middle frames) are passed as references for editing / composition.
"""

import os
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import find_strip_by_name, load_strip_as_pil

# Nano Banana accepts up to 9 reference images; how many picker rows are shown
# is driven by scene.nano_banana_ref_count.  _REF_ATTRS lists every possible slot
# so the queue/metadata plumbing can enumerate them uniformly.
NANO_BANANA_MAX_REFS = 9
_REF_ATTRS = [f"nano_banana_ref_strip_{i}" for i in range(1, NANO_BANANA_MAX_REFS + 1)]


class GoogleNanoBananaPlugin(ModelPlugin):
    MODEL_ID     = "google/nano-banana"
    DISPLAY_NAME = "Google Nano Banana (cloud)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Cloud text-to-image / image editing via the Google Gemini API"

    # Gemini image models are non-deterministic (no seed) and have no separate
    # negative-prompt parameter, so neither is exposed — every shown control works.
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.API_KEY
    UI_SECTIONS  = [UISection.PROMPT]
    PARAMS       = ParamSpec(width=1024, height=1024)
    REQUIRED_PACKAGES = ["google.genai", "PIL"]

    supports_inpaint = False
    supports_img2img = True    # image editing / composition from reference strips
    show_enhance     = False   # local Quality/Speed/Upscale toggles don't apply to cloud
    uses_strip_power = False    # Gemini editing has no img2img "strength" parameter
    supports_batch   = True     # batch count → multiple separate API calls

    # ---- UI ---------------------------------------------------------------
    def draw_custom_ui(self, col, context) -> bool:
        # Klein-style reference-image pickers (composition / editing).  Imagen is
        # text-to-image only, so image inputs are hidden when it is selected.
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        model = getattr(scene, "nano_banana_model", "gemini-2.5-flash-image")
        if model.startswith("imagen"):
            return True
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        col.prop(scene, "nano_banana_ref_count")
        ref_count = max(1, min(getattr(scene, "nano_banana_ref_count", 3), len(_REF_ATTRS)))
        for i, attr in enumerate(_REF_ATTRS[:ref_count], 1):
            row = col.row(align=True)
            row.prop_search(vse_scene, attr, vse_scene.sequence_editor, "strips",
                            text="Ref.", icon="FILE_IMAGE")
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"nano_banana_select{i}"
        return True

    def draw_post_seed_ui(self, col, context):
        scene = context.scene
        model = getattr(scene, "nano_banana_model", "gemini-2.5-flash-image")
        col.prop(scene, "nano_banana_model")
        col.prop(scene, "nano_banana_aspect")
        # 2K/4K (image_size) is honoured only by Nano Banana Pro.
        if model == "gemini-3-pro-image-preview":
            col.prop(scene, "nano_banana_resolution")

    # ---- Lifecycle --------------------------------------------------------
    def load(self, prefs, scene, **kw):
        # No client here — load() is cached across runs and the key may change.
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
                "Google API quota exceeded (429). Image generation is not on the free "
                "tier (limit 0) or you hit a rate limit — enable billing on your Google "
                "AI Studio / Cloud project, or wait and retry. " + msg[:200]
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
    def _usage_note(response) -> str:
        """Short token-usage string from a response's usage_metadata, or ''."""
        um = getattr(response, "usage_metadata", None)
        total = getattr(um, "total_token_count", None) if um is not None else None
        try:
            return f"{int(total):,} tokens" if total else ""
        except Exception:
            return ""

    @staticmethod
    def _enum_name(v):
        """Best-effort readable name for an SDK enum / value (or None)."""
        if v is None:
            return None
        return getattr(v, "name", None) or str(v)

    @classmethod
    def _diagnose_response(cls, response) -> str:
        """Summarise a response that yielded no image: finish reasons, prompt
        block reasons, safety ratings and any text the model returned instead.

        Returns a one-line human-readable summary (also printed in full)."""
        bits = []
        # Prompt-level block (request rejected before any candidate).
        pf = getattr(response, "prompt_feedback", None)
        if pf is not None:
            br = cls._enum_name(getattr(pf, "block_reason", None))
            if br:
                bits.append(f"prompt blocked: {br}")
            bmsg = getattr(pf, "block_reason_message", None)
            if bmsg:
                bits.append(f"block message: {bmsg}")

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            bits.append("no candidates returned")
        for i, cand in enumerate(candidates):
            fr = cls._enum_name(getattr(cand, "finish_reason", None))
            if fr:
                bits.append(f"candidate[{i}] finish_reason={fr}")
            fm = getattr(cand, "finish_message", None)
            if fm:
                bits.append(f"candidate[{i}] finish_message={fm}")
            # Surface any safety ratings that actually tripped.
            for sr in (getattr(cand, "safety_ratings", None) or []):
                if getattr(sr, "blocked", False):
                    cat = cls._enum_name(getattr(sr, "category", None))
                    prob = cls._enum_name(getattr(sr, "probability", None))
                    bits.append(f"candidate[{i}] safety blocked: {cat} ({prob})")
            # Collect any text parts (the model often explains a refusal here).
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                txt = getattr(part, "text", None)
                if txt:
                    bits.append(f"candidate[{i}] text: {txt.strip()[:300]}")

        summary = "; ".join(bits) if bits else "response contained no image and no diagnostic info"
        print("[Nano Banana] empty-image diagnostics: " + summary)
        # Full repr is invaluable when the structured fields above come up empty.
        try:
            print("[Nano Banana] raw response repr (truncated):\n" + repr(response)[:2000])
        except Exception:
            pass
        return summary

    def _collect_ref_images(self, inputs: ModelInputs, scene):
        """Return PIL reference images from the picker strips (+ any primary image).

        Each picker resolves via its add-time path (queue) or, failing that, the
        strip name (interactive) — mirroring FLUX Kontext/Klein.
        """
        from PIL import Image
        refs = []
        if inputs.image is not None:
            refs.append(inputs.image)
        # Honour the active reference count so stale names in hidden slots
        # (the UI only shows the first nano_banana_ref_count rows) don't leak in.
        ref_count = max(1, min(getattr(scene, "nano_banana_ref_count", 3), len(_REF_ATTRS)))
        for attr in _REF_ATTRS[:ref_count]:
            img = None
            path = getattr(scene, attr + "_path", "") or ""
            if path and os.path.isfile(path):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    print(f"[Nano Banana] Could not open reference {path!r}: {e}")
            if img is None:
                name = getattr(scene, attr, "") or ""
                if name:
                    strip = find_strip_by_name(scene, name)
                    if strip:
                        try:
                            img = load_strip_as_pil(strip)
                        except Exception as e:
                            print(f"[Nano Banana] Could not load reference strip {name!r}: {e}")
            if img is not None:
                refs.append(img)
        return refs

    # ---- Generation -------------------------------------------------------
    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import io
        from google import genai
        from google.genai import types
        from google.genai import errors as genai_errors
        from PIL import Image

        api_key = self._get_api_key(prefs)
        if not api_key:
            raise RuntimeError(
                "Google Gemini API key is missing. Set it in the Pallaidium add-on "
                "preferences (Google Gemini API Key) or the GEMINI_API_KEY env var."
            )

        model       = getattr(scene, "nano_banana_model", "gemini-2.5-flash-image")
        aspect      = getattr(scene, "nano_banana_aspect", "1:1")
        resolution  = getattr(scene, "nano_banana_resolution", "1K")

        client = genai.Client(api_key=api_key)
        self.set_phase(inputs, "Submitting to cloud")

        # ── Imagen models use a different endpoint ─────────────────────────
        if model.startswith("imagen"):
            cfg = types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect,
            )
            try:
                resp = client.models.generate_images(
                    model=model, prompt=inputs.prompt, config=cfg,
                )
            except genai_errors.APIError as e:
                raise self._friendly_error(e)
            inputs.usage_note = self._usage_note(resp)
            self.set_phase(inputs, "Saving")
            gens = resp.generated_images or []
            print(f"[Nano Banana] Imagen request: model={model} aspect={aspect} "
                  f"prompt={inputs.prompt[:120]!r} -> {len(gens)} image(s)")
            for gen in gens:
                img = gen.image
                # SDK may expose a PIL image directly or raw bytes.
                pil = getattr(img, "_pil_image", None)
                if pil is not None:
                    return pil
                data = getattr(img, "image_bytes", None)
                if data:
                    return Image.open(io.BytesIO(data)).convert("RGB")
            # Imagen reports content filtering per-image and overall.
            reasons = []
            for gen in gens:
                rai = getattr(gen, "rai_filtered_reason", None)
                if rai:
                    reasons.append(str(rai))
            filt = getattr(resp, "positive_prompt_safety_attributes", None)
            print("[Nano Banana] Imagen empty-image diagnostics: "
                  + (("; ".join(reasons)) if reasons else "no images and no filter reason")
                  + (f"; safety_attributes={filt}" if filt is not None else ""))
            try:
                print("[Nano Banana] Imagen raw response repr (truncated):\n" + repr(resp)[:2000])
            except Exception:
                pass
            detail = "; ".join(reasons) if reasons else "no images returned (likely content-filtered)"
            raise RuntimeError("Imagen API returned no image data — " + detail)

        # ── Gemini image models (generate_content) ─────────────────────────
        ref_images = self._collect_ref_images(inputs, scene)
        contents = [inputs.prompt] + ref_images
        print(f"[Nano Banana] request: model={model} aspect={aspect} "
              f"resolution={resolution} ref_images={len(ref_images)} "
              f"prompt={inputs.prompt[:120]!r}")

        # Build image_config defensively — aspect_ratio is widely supported;
        # image_size (2K/4K) is honoured only by Nano Banana Pro.  Skip "1K"
        # so the Flash model (which rejects image_size) keeps working.
        image_config = None
        try:
            image_config = types.ImageConfig(aspect_ratio=aspect)
            # image_size (2K/4K) is honoured only by Nano Banana Pro; never send
            # it for the Flash model, which rejects it.
            if resolution in ("2K", "4K") and model == "gemini-3-pro-image-preview":
                try:
                    image_config.image_size = resolution
                except Exception:
                    print(f"[Nano Banana] image_size={resolution!r} unsupported; ignored.")
        except Exception:
            print("[Nano Banana] ImageConfig unsupported by this SDK; using defaults.")

        cfg_kwargs = {"response_modalities": ["IMAGE"]}
        if image_config is not None:
            cfg_kwargs["image_config"] = image_config
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=cfg,
            )
        except genai_errors.APIError as e:
            raise self._friendly_error(e)

        inputs.usage_note = self._usage_note(response)
        self.set_phase(inputs, "Saving")
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            for part in (getattr(content, "parts", None) or []):
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    return Image.open(io.BytesIO(inline.data)).convert("RGB")

        # No image part — surface *why* (safety block, text-only reply, etc.).
        detail = self._diagnose_response(response)
        raise RuntimeError("Google Gemini API returned no image data — " + detail)
