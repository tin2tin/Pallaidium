"""Image captioning and Ideogram-4 prompt extraction via Florence-2."""

import json
import re

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs

try:
    import bpy as _bpy
    if not hasattr(_bpy.types.Scene, "florence2_mode"):
        _bpy.types.Scene.florence2_mode = _bpy.props.EnumProperty(
            name="Mode",
            items=[
                ("CAPTION",   "Caption",    "Detailed image caption as plain text"),
                ("IDEOGRAM4", "Box Json", "Extract structured Ideogram 4 prompt JSON"),
            ],
            default="CAPTION",
        )
    if not hasattr(_bpy.types.Scene, "florence2_send_to_mask"):
        _bpy.types.Scene.florence2_send_to_mask = _bpy.props.BoolProperty(
            name="Send to Mask Editor",
            description="After generation, open result as mask layers in the Image Editor",
            default=False,
        )
except Exception:
    pass


_GENERIC_CAPTION_PREFIXES = (
    "the image is a still from a movie or tv show. ",
    "the image is a still from a movie or tv show, ",
    "the image is a still from a movie or tv show ",
    "the image is a still from ",
    "the image shows a ",
    "the image shows ",
    "the image depicts a ",
    "the image depicts ",
    "the image features a ",
    "the image features ",
    "the image captures a ",
    "the image captures ",
    "the image contains a ",
    "the image contains ",
    "the image presents a ",
    "the image presents ",
    "the image is of a ",
    "the image is of ",
    "the image is ",
    "this image shows a ",
    "this image shows ",
    "this image depicts a ",
    "this image depicts ",
    "this image features a ",
    "this image features ",
    "in the image, a ",
    "in the image, ",
    "in this image, a ",
    "in this image, ",
    "the image ",
    "it shows a ",
    "it shows ",
    "it depicts a ",
    "it depicts ",
    "it features a ",
    "it features ",
    "it captures a ",
    "it captures ",
)


def _strip_caption_prefix(text: str) -> str:
    """Remove generic Florence-2 preamble so the unique content comes first.

    Loops until no further prefix can be stripped, handling chained phrases
    like "The image is a still from a movie or TV show. It shows a close-up…"
    """
    t = text.strip()
    while True:
        lower = t.lower()
        matched = False
        for prefix in _GENERIC_CAPTION_PREFIXES:
            if lower.startswith(prefix):
                t = t[len(prefix):].strip()
                matched = True
                break
        if not matched:
            break
    return t[0].upper() + t[1:] if t else t


class Florence2Plugin(ModelPlugin):
    MODEL_ID     = "florence-community/Florence-2-large"
    DISPLAY_NAME = "Image Captioning: Florence-2"
    MODEL_TYPE   = "text"
    DESCRIPTION  = "Detailed image caption or Ideogram 4 structured prompt JSON"

    INPUTS       = InputSpec.IMAGE
    UI_SECTIONS  = []
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "PIL", "transformers"]

    requires_input_strip = True
    supports_batch       = False   # deterministic caption → batch makes identical copies

    def load(self, prefs, scene, **kw):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        cache_dir = prefs.hf_cache_dir or None
        model = Florence2ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=prefs.local_files_only,
        )
        processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            cache_dir=cache_dir,
            local_files_only=prefs.local_files_only,
        )
        return {"model": model, "processor": processor, "tokenizer": None}

    def draw_custom_ui(self, col, context) -> bool:
        col.prop(context.scene, "florence2_mode", expand=True)
        if context.scene.florence2_mode == "IDEOGRAM4":
            col.prop(context.scene, "florence2_send_to_mask", text="Send to Box Editor")
        return False

    def draw_post_seed_ui(self, col, context) -> None:
        if context.scene.florence2_mode == "IDEOGRAM4":
            col.separator()
            col.operator("florence2.open_box_editor", text="Open Box Editor", icon="MOD_MASK")

    # ------------------------------------------------------------------

    def _run_task(self, model, processor, image, task: str) -> dict:
        proc_inputs = processor(text=task, images=image, return_tensors="pt")
        proc_inputs = {k: v.to(model.device) for k, v in proc_inputs.items()}
        generated_ids = model.generate(
            **proc_inputs,
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=False,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # post_process_generation matches `task` by EXACT key and returns the
        # result keyed by it.  Tasks that carry inline arguments (a region's
        # <loc_*> tokens, a grounding phrase, …) must be reduced to their bare
        # <TOKEN> here, or parsing falls back to pure_text and the result is
        # stored under the wrong key — silently dropping bboxes/descriptions.
        m = re.match(r"<[^>]+>", task)
        task_token = m.group(0) if m else task
        return processor.post_process_generation(
            generated_text,
            task=task_token,
            image_size=(image.width, image.height),
        )

    def generate(self, pipe, inputs: ModelInputs, scene, prefs) -> str:
        model     = pipe["model"]
        processor = pipe["processor"]
        image     = inputs.image

        if image is None:
            raise ValueError("Florence-2 requires an image input — select an IMAGE or MOVIE strip.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        mode = getattr(scene, "florence2_mode", "CAPTION")
        if mode == "IDEOGRAM4":
            return self._ideogram4(model, processor, image, inputs)
        return self._caption(model, processor, image, inputs)

    # ------------------------------------------------------------------
    # Mode: CAPTION

    # Cinematographic angle terms, ordered from most-specific to least.
    _ANGLE_RULES = [
        (("over-the-shoulder", "over the shoulder"),            "over-the-shoulder shot"),
        (("looking at the camera", "facing the camera",
          "faces the camera", "facing forward",
          "facing front", "full frontal",
          "looking toward the camera", "looking forward"),       "frontal shot"),
        (("from behind", "seen from behind", "back view",
          "rear view", "from the back", "their back",
          "his back", "her back", "back to the camera",
          "back turned", "back is turned", "facing away",
          "from the rear", "turned away"),                       "rear shot"),
        (("three-quarter", "three quarter",
          "turned slightly", "angled"),                          "three-quarter shot"),
        (("left profile", "right profile", "side profile",
          "in profile", "from the side", "side view",
          "profile view", "profile", "side"),                    "side-profile shot"),
    ]

    # Human-readable facing phrase per cinematic-angle label.
    _FACING_PHRASE = {
        "over-the-shoulder shot": "over-the-shoulder",
        "frontal shot":           "facing the camera",
        "rear shot":              "seen from behind",
        "three-quarter shot":     "three-quarter view",
        "side-profile shot":      "in profile",
    }

    # Gender-cue words (whole-word matched).
    _MALE_WORDS   = ("man", "male", "gentleman", "boy", "he", "his", "him",
                     "father", "husband", "businessman", "guy")
    _FEMALE_WORDS = ("woman", "female", "lady", "girl", "she", "her",
                     "mother", "wife", "businesswoman")

    def _facing_phrase(self, desc: str) -> str:
        """Return a human-readable facing phrase ('seen from behind', …) or ''."""
        return self._FACING_PHRASE.get(self._cinematic_angle(desc), "")

    # Gaze: capture a "looking …" / "gazing …" clause (direction or target).
    _GAZE_RE = re.compile(r"\b(?:looking|gazing|staring|glancing)\b[^.,;]*", re.IGNORECASE)

    def _looking_note(self, text: str) -> str:
        """Extract a concise gaze phrase ('looking at the man', 'looking to the
        left', 'looking down', …) from a description, or '' if none."""
        if not text:
            return ""
        m = self._GAZE_RE.search(text)
        if not m:
            return ""
        note = re.sub(r"\s+(?:with|and|who|while|as|in|on)\b.*$", "",
                      m.group(0).strip().lower()).strip()
        return note[:60]

    def _infer_gender(self, text: str) -> str:
        """Infer 'man' / 'woman' from description text, or '' if ambiguous."""
        t = (text or "").lower()
        male   = any(re.search(rf"\b{w}\b", t) for w in self._MALE_WORDS)
        female = any(re.search(rf"\b{w}\b", t) for w in self._FEMALE_WORDS)
        if male and not female:
            return "man"
        if female and not male:
            return "woman"
        return ""

    def _cinematic_angle(self, desc: str) -> str:
        t = desc.lower()
        for keywords, label in self._ANGLE_RULES:
            if any(kw in t for kw in keywords):
                return label
        return ""

    @staticmethod
    def _clean_desc(text: str) -> str:
        """Strip leftover model tokens (<loc_*>, <s>, …) and tidy whitespace."""
        cleaned = re.sub(r"<[^>]*>", "", text or "")
        return re.sub(r"\s+", " ", cleaned).strip(" ,.;")

    @staticmethod
    def _detect_panels(image):
        """Detect a storyboard/collage grid from its uniform gutter lines.

        Returns a list of Ideogram bboxes [y1,x1,y2,x2] (0-1000), one per panel,
        or [] when no clear multi-panel grid is found.  Gutters are detected as
        full-width/full-height near-uniform lines; this works best for clean
        grids with solid separators (heavily letterboxed stills may split).
        """
        try:
            import numpy as np
            g = np.asarray(image.convert("L"), dtype=np.float32)
        except Exception:
            return []
        H, W = g.shape
        if H < 60 or W < 60:
            return []

        col_sep = g.std(axis=0) < 10.0   # uniform full-height column = gutter
        row_sep = g.std(axis=1) < 10.0   # uniform full-width  row    = gutter

        def content_spans(sep, n):
            min_len = max(8, int(n * 0.12))   # ignore bands thinner than 12 %
            spans, i = [], 0
            while i < n:
                if not sep[i]:
                    j = i
                    while j < n and not sep[j]:
                        j += 1
                    if j - i >= min_len:
                        spans.append((i, j))
                    i = j
                else:
                    i += 1
            return spans

        cols = content_spans(col_sep, W)
        rows = content_spans(row_sep, H)
        n_panels = len(cols) * len(rows)
        if n_panels < 2 or n_panels > 24:
            return []   # not a grid (or implausibly many cells)

        return [
            [round(y1 / H * 1000), round(x1 / W * 1000),
             round(y2 / H * 1000), round(x2 / W * 1000)]
            for (y1, y2) in rows for (x1, x2) in cols
        ]

    def _region_desc(self, model, processor, image, bbox_px) -> str:
        W, H = image.width, image.height
        x1, y1, x2, y2 = bbox_px
        lx1 = round(x1 / W * 999); ly1 = round(y1 / H * 999)
        lx2 = round(x2 / W * 999); ly2 = round(y2 / H * 999)
        task = f"<REGION_TO_DESCRIPTION><loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
        return self._clean_desc(
            self._run_task(model, processor, image, task).get("<REGION_TO_DESCRIPTION>", ""))

    def _caption(self, model, processor, image, inputs) -> str:
        _PERSON_LABELS = {"person", "man", "woman", "boy", "girl", "child", "human", "people"}

        self.set_phase(inputs, "Captioning")
        text = self._run_task(model, processor, image, "<MORE_DETAILED_CAPTION>").get(
            "<MORE_DETAILED_CAPTION>", ""
        )

        self.set_phase(inputs, "Detecting persons")
        od_data = self._run_task(model, processor, image, "<OD>").get("<OD>", {})
        person_bboxes = [
            bbox for bbox, label in zip(
                od_data.get("bboxes", []), od_data.get("labels", [])
            )
            if label.lower().strip() in _PERSON_LABELS
        ]

        if person_bboxes:
            person_notes = []  # (angle, gaze_desc)
            for i, bbox_px in enumerate(person_bboxes[:3]):
                self.set_phase(inputs, f"Person {i + 1}")
                angle = gaze = ""
                try:
                    region = self._region_desc(model, processor, image, bbox_px)
                    angle = self._cinematic_angle(region)
                except Exception:
                    pass
                try:
                    self.set_phase(inputs, f"Gaze {i + 1}")
                    query = (
                        "the object the person is looking at"
                        if len(person_bboxes) == 1
                        else f"the object person {i + 1} is looking at"
                    )
                    rec = self._run_task(
                        model, processor, image,
                        f"<REFERRING_EXPRESSION_COMPREHENSION>{query}",
                    ).get("<REFERRING_EXPRESSION_COMPREHENSION>", {})
                    gaze_bboxes = rec.get("bboxes", [])
                    if gaze_bboxes:
                        gaze = self._region_desc(model, processor, image, gaze_bboxes[0])
                except Exception:
                    pass
                if angle or gaze:
                    person_notes.append((angle, gaze))

            extras = []
            singular = len(person_bboxes) == 1
            for i, (angle, gaze) in enumerate(person_notes):
                prefix = "The person" if singular else f"Person {i + 1}"
                parts = []
                if angle:
                    parts.append(f"seen from a {angle}")
                if gaze:
                    parts.append(f"looking at {gaze}")
                if parts:
                    extras.append(f"{prefix} is {', '.join(parts)}.")
            if extras:
                text = text.rstrip(".") + ". " + " ".join(extras)

        print("Florence-2 caption:", text)
        return text

    # ------------------------------------------------------------------
    # Mode: IDEOGRAM4

    def _ideogram4(self, model, processor, image, inputs) -> str:
        W, H = image.width, image.height

        # ---- 1. Captions ----
        self.set_phase(inputs, "Caption")
        caption = self._run_task(model, processor, image, "<MORE_DETAILED_CAPTION>").get(
            "<MORE_DETAILED_CAPTION>", ""
        ).strip()
        background = self._run_task(model, processor, image, "<CAPTION>").get(
            "<CAPTION>", ""
        ).strip()

        # ---- 2. Dense region captions ----
        self.set_phase(inputs, "Dense regions")
        dense_data = self._run_task(model, processor, image, "<DENSE_REGION_CAPTION>").get(
            "<DENSE_REGION_CAPTION>", {}
        )

        # ---- 3. Object detection ----
        self.set_phase(inputs, "Object detection")
        od_data = self._run_task(model, processor, image, "<OD>").get("<OD>", {})

        # ---- 4. OCR ----
        self.set_phase(inputs, "OCR")
        ocr_data = self._run_task(model, processor, image, "<OCR_WITH_REGION>").get(
            "<OCR_WITH_REGION>", {}
        )

        # ---- helpers ----
        _PERSON_LABELS = {"person", "man", "woman", "boy", "girl", "child", "people", "human"}

        def is_person_label(label: str) -> bool:
            return label.lower().strip() in _PERSON_LABELS

        def is_face_label(label: str) -> bool:
            l = label.lower().strip()
            return l == "head" or "face" in l

        def bbox_contains_center(outer_ib, inner_ib) -> bool:
            """True if the center of inner_ib falls inside outer_ib ([y1,x1,y2,x2])."""
            cy = (inner_ib[0] + inner_ib[2]) / 2
            cx = (inner_ib[1] + inner_ib[3]) / 2
            return (outer_ib[0] <= cy <= outer_ib[2]
                    and outer_ib[1] <= cx <= outer_ib[3])

        def region_description(bbox_px) -> str:
            """Run <REGION_TO_DESCRIPTION> for a pixel bbox; return description string."""
            x1, y1, x2, y2 = bbox_px
            # Florence-2 location tokens use 0–999 scale
            lx1 = round(x1 / W * 999)
            ly1 = round(y1 / H * 999)
            lx2 = round(x2 / W * 999)
            ly2 = round(y2 / H * 999)
            task = f"<REGION_TO_DESCRIPTION><loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
            try:
                result = self._run_task(model, processor, image, task)
                return self._clean_desc(result.get("<REGION_TO_DESCRIPTION>", ""))
            except Exception:
                return ""

        def dominant_palette(count=5):
            import numpy as np
            small = image.resize((80, 80))
            arr   = np.asarray(small).reshape(-1, 3)
            bins  = np.clip((arr // 32) * 32 + 16, 0, 255).astype(np.uint8)
            colors, counts = np.unique(bins, axis=0, return_counts=True)
            order = np.argsort(counts)[::-1][:count]
            return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in colors[order]]

        def infer_style(text: str) -> dict:
            t = text.lower()
            is_photo = any(w in t for w in ("photograph", "photo", "camera", "shot", "lens", "dslr", "film"))
            is_art   = any(w in t for w in ("painting", "illustration", "drawing", "sketch", "render",
                                             "artwork", "watercolor", "oil paint", "digital art", "3d"))
            lighting_words = ("natural light", "sunlight", "golden hour", "backlit", "overcast",
                              "studio light", "soft light", "hard light", "neon", "candlelight",
                              "dramatic lighting", "ambient", "daylight", "moonlight")
            lighting = next((w for w in lighting_words if w in t), "natural light")
            aesthetic_words = ("cinematic", "minimalist", "vibrant", "moody", "dark", "bright",
                               "high contrast", "soft", "ethereal", "gritty", "vintage", "modern")
            aesthetics = next((w for w in aesthetic_words if w in t), "photorealistic")
            style: dict = {"aesthetics": aesthetics, "lighting": lighting}
            if is_art and not is_photo:
                art_words = ("watercolor", "oil painting", "pencil sketch", "digital art",
                             "3d render", "illustration", "comic", "anime")
                art_style = next((w for w in art_words if w in t), "digital illustration")
                style["medium"] = "digital" if "digital" in t else "traditional"
                style["art_style"] = art_style
            else:
                style["photo"] = "eye-level, natural perspective"
                style["medium"] = "photograph"
            style["color_palette"] = dominant_palette()
            return style

        def sample_color(bbox):
            from PIL import Image as _PIL
            y1, x1, y2, x2 = bbox
            left   = int(x1 / 1000 * W)
            top    = int(y1 / 1000 * H)
            right  = max(left + 1, int(x2 / 1000 * W))
            bottom = max(top  + 1, int(y2 / 1000 * H))
            crop = image.crop((left, top, right, bottom)).resize((1, 1), _PIL.Resampling.BILINEAR)
            r, g, b = crop.getpixel((0, 0))
            return f"#{r:02X}{g:02X}{b:02X}"

        def font_size_label(bbox):
            height_fraction = (bbox[2] - bbox[0]) / 1000.0
            if height_fraction > 0.15:
                return "huge"
            if height_fraction > 0.07:
                return "large"
            if height_fraction > 0.03:
                return "medium"
            return "small"

        def detect_lighting(text: str):
            """Return (direction, setting) inferred from caption keywords."""
            t = text.lower()
            # -- setting (most specific match wins) --
            _setting_rules = [
                ("golden hour",    "golden hour"),
                ("magic hour",     "golden hour"),
                ("sunset",         "golden hour"),
                ("sunrise",        "golden hour"),
                ("backlit",        "backlit"),
                ("back light",     "backlit"),
                ("silhouette",     "backlit"),
                ("rim light",      "rim light"),
                ("studio light",   "studio light"),
                ("softbox",        "studio light"),
                ("neon",           "neon light"),
                ("candle",         "candlelight"),
                ("firelight",      "candlelight"),
                ("moonlight",      "moonlight"),
                ("night",          "low-light"),
                ("overcast",       "overcast"),
                ("cloudy",         "overcast"),
                ("diffuse",        "diffuse light"),
                ("soft light",     "soft light"),
                ("hard light",     "hard light"),
                ("dramatic light", "dramatic light"),
                ("chiaroscuro",    "dramatic light"),
                ("sunlight",       "sunlight"),
                ("daylight",       "daylight"),
                ("window light",   "window light"),
                ("indoor",         "indoor light"),
                ("natural light",  "natural light"),
            ]
            setting = "natural light"
            for keyword, label in _setting_rules:
                if keyword in t:
                    setting = label
                    break

            # -- direction from caption keywords --
            _dir_rules = [
                (("backlit", "silhouette", "back light", "rim light"),    "from behind"),
                (("side lit", "side light", "rembrandt", "split light"),  "from side"),
                (("overhead", "top light", "lit from above",
                  "above", "skylight"),                                    "from above"),
                (("underlighting", "lit from below", "from below"),       "from below"),
                (("front lit", "frontal light", "flat light",
                  "facing the light", "facing camera"),                    "frontal"),
                (("three-quarter", "45 degree", "butterfly light"),       "three-quarter"),
                (("window", "from the left", "left side"),                "from left"),
                (("from the right", "right side"),                        "from right"),
            ]
            direction = "natural"
            for keywords, label in _dir_rules:
                if any(kw in t for kw in keywords):
                    direction = label
                    break

            return direction, setting

        def to_ideogram(bbox_px):
            x1, y1, x2, y2 = bbox_px
            return [
                round(y1 / H * 1000), round(x1 / W * 1000),
                round(y2 / H * 1000), round(x2 / W * 1000),
            ]

        def area(b):
            return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

        def iou(a, b):
            iy1, ix1 = max(a[0], b[0]), max(a[1], b[1])
            iy2, ix2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
            if inter == 0:
                return 0.0
            return inter / (area(a) + area(b) - inter)

        # ---- build dense items ----
        dense_items = []
        for bbox_px, label in zip(
            dense_data.get("bboxes", []),
            dense_data.get("labels", []),
        ):
            ib = to_ideogram(bbox_px)
            if area(ib) > 40:
                dense_items.append({"bbox": ib, "description": label})

        # ---- build elements from OD, enriched by dense descriptions ----
        elements = []
        person_infos = []   # {ib, gender, facing} for face/body-part association
        od_bboxes = od_data.get("bboxes", [])
        od_labels = od_data.get("labels", [])
        if od_bboxes:
            od_items = list(zip(od_bboxes, od_labels))
            persons = [(b, l) for b, l in od_items if is_person_label(l)]
            faces   = [(b, l) for b, l in od_items if is_face_label(l)]
            others  = [(b, l) for b, l in od_items
                       if not is_person_label(l) and not is_face_label(l)]

            # Budget for <REGION_TO_DESCRIPTION> calls (each is a model pass).
            region_budget = 8

            # Detected face boxes — a person box with NO face inside it is the
            # most reliable signal that the figure is turned away (back to camera);
            # Florence's region captions rarely state orientation explicitly.
            face_ibs = [to_ideogram(b) for b, _ in faces]

            def _articled(text: str) -> str:
                return text if re.match(r"^(a |an |the )", text.lower()) else f"a {text}"

            # Pre-pass: which person boxes own a face (face center inside them)?
            person_ibs      = [to_ideogram(b) for b, _ in persons]
            person_has_face = [
                any(bbox_contains_center(pib, fib) for fib in face_ibs)
                for pib in person_ibs
            ]

            def _is_back(idx: int) -> bool:
                """Faceless figure = back to camera, unless it's just an overlapping
                duplicate of another facing person (e.g. a desk/lower-body crop)."""
                if person_has_face[idx]:
                    return False
                pib = person_ibs[idx]
                for j, pjb in enumerate(person_ibs):
                    if j != idx and person_has_face[j] and iou(pib, pjb) > 0.3:
                        return False
                return True

            # ---- persons: enrich with gender + facing + region detail ----
            for p_idx, (bbox_px, label) in enumerate(persons):
                ib   = person_ibs[p_idx]
                rich = ""
                if region_budget > 0:
                    region_budget -= 1
                    self.set_phase(inputs, f"Person {len(person_infos) + 1}")
                    rich = region_description(bbox_px)
                gender = (label.lower().strip()
                          if label.lower().strip() in ("man", "woman", "boy", "girl")
                          else (self._infer_gender(rich) or "person"))

                back = _is_back(p_idx)
                facing = "seen from behind" if back else self._facing_phrase(rich)

                base = rich or gender
                if gender != "person" and not re.search(rf"\b{gender}\b", base.lower()):
                    base = f"{gender}, {base}"
                if back:
                    detail = f"the back of {_articled(base)}"
                else:
                    detail = base
                    if facing and not self._cinematic_angle(detail):
                        detail = f"{detail}, {facing}"
                    # Surface gaze ('looking at the man', 'looking to the left').
                    gaze = self._looking_note(rich)
                    if gaze and "look" not in detail.lower():
                        detail = f"{detail}, {gaze}"
                person_infos.append({"ib": ib, "gender": gender, "facing": facing, "back": back})
                elements.append({"type": "obj", "bbox": ib, "desc": detail, "label": label})

            # ---- faces: reuse the containing person's gender/facing when possible ----
            for bbox_px, label in faces:
                ib = to_ideogram(bbox_px)
                match, best_ov = None, 0.0
                for pi in person_infos:
                    if bbox_contains_center(pi["ib"], ib):
                        ov = iou(pi["ib"], ib)
                        if match is None or ov > best_ov:
                            match, best_ov = pi, ov
                gender = match["gender"] if match else ""
                facing = match["facing"] if match else ""
                rich   = ""
                if not gender and region_budget > 0:
                    region_budget -= 1
                    self.set_phase(inputs, "Face")
                    rich   = region_description(bbox_px)
                    gender = self._infer_gender(rich)
                    facing = self._facing_phrase(rich)
                head = f"{gender}'s face" if gender in ("man", "woman", "boy", "girl") else "face"
                parts = [head]
                if facing:
                    parts.append(facing)
                if rich and "face" not in rich.lower()[:16]:
                    parts.append(rich)
                elements.append({"type": "obj", "bbox": ib, "desc": ", ".join(parts), "label": label})

            # ---- other objects: enrich from dense region captions ----
            for bbox_px, label in others:
                ib   = to_ideogram(bbox_px)
                desc = label
                if dense_items:
                    best = max(dense_items, key=lambda d: iou(ib, d["bbox"]))
                    if iou(ib, best["bbox"]) > 0.2:
                        desc = best["description"]
                # Clothing/items sitting inside a back-turned person inherit "back of".
                if any(pi["back"] and bbox_contains_center(pi["ib"], ib)
                       for pi in person_infos):
                    desc = f"the back of {_articled(desc)}"
                elements.append({"type": "obj", "bbox": ib, "desc": desc, "label": label})
        elif dense_items:
            elements = [
                {"type": "obj", "bbox": d["bbox"], "desc": d["description"], "label": d["description"]}
                for d in dense_items[:20]
            ]

        # ---- body parts: always locate eyes / hands / feet ----
        # Open-vocabulary detection per part is more reliable than grounding a
        # multi-phrase caption.  Run one query each and tag with the owner.
        self.set_phase(inputs, "Body parts")
        _PART_QUERIES = (("eye", "human eye"), ("hand", "human hand"), ("foot", "human foot"))
        for part, query in _PART_QUERIES:
            try:
                pr = self._run_task(
                    model, processor, image,
                    f"<OPEN_VOCABULARY_DETECTION>{query}",
                )
                pdata = pr.get("<OPEN_VOCABULARY_DETECTION>") or {}
            except Exception:
                pdata = {}
            p_bboxes = pdata.get("bboxes", [])
            p_labels = pdata.get("bboxes_labels", pdata.get("labels", []))
            for bbox_px, label in zip(p_bboxes, p_labels or [part] * len(p_bboxes)):
                lab = (label or part).lower()
                if part not in lab and lab not in part:
                    continue              # keep only the queried part
                ib = to_ideogram(bbox_px)
                if area(ib) < 4:          # drop degenerate boxes
                    continue
                owner = next(
                    (pi for pi in person_infos if bbox_contains_center(pi["ib"], ib)),
                    None,
                )
                desc = (f"{owner['gender']}'s {part}"
                        if owner and owner["gender"] in ("man", "woman", "boy", "girl")
                        else part)
                elements.append({"type": "obj", "bbox": ib, "desc": desc, "label": part})

        # ---- deduplicate ----
        deduped = []
        for elem in elements:
            if not any(
                iou(elem["bbox"], kept["bbox"]) > 0.85 and elem["label"] == kept["label"]
                for kept in deduped
            ):
                deduped.append(elem)
        elements = deduped

        # ---- OCR text elements ----
        for quad, label in zip(
            ocr_data.get("quad_boxes", []),
            ocr_data.get("labels", []),
        ):
            xs = [quad[i] for i in range(0, 8, 2)]
            ys = [quad[i] for i in range(1, 8, 2)]
            ib = [
                round(min(ys) / H * 1000), round(min(xs) / W * 1000),
                round(max(ys) / H * 1000), round(max(xs) / W * 1000),
            ]
            clean = label.strip()
            has_alnum = any(c.isalnum() for c in clean)
            if area(ib) > 20 and len(clean) >= 2 and has_alnum:
                elements.append({
                    "type":  "text",
                    "bbox":  ib,
                    "desc":  f"Text reading '{clean}'",
                    "label": clean,
                    "text":  clean,
                    "color": sample_color(ib),
                    "font":  font_size_label(ib),
                })

        # ---- sort top-to-bottom, left-to-right; cap at 40 ----
        elements.sort(key=lambda e: (e["bbox"][0], e["bbox"][1]))
        elements = elements[:40]

        # ---- assemble final JSON ----
        ordered = []
        for elem in elements:
            if elem["type"] == "text":
                # Schema key order: type, bbox, text, desc, color_palette (optional)
                e = {
                    "type": "text",
                    "bbox": elem["bbox"],
                    "text": elem.get("text", ""),
                    "desc": elem["desc"],
                }
                color_hex = elem.get("color", "")
                if color_hex and color_hex.upper() not in ("#FFFFFF", "#FEFEFE"):
                    e["color_palette"] = [color_hex.upper()]
                ordered.append(e)
            else:
                # Schema key order: type, bbox, desc, color_palette (optional)
                ordered.append({
                    "type": "obj",
                    "bbox": elem["bbox"],
                    "desc": elem["desc"],
                })

        def detect_shot_type(elems):
            """Return shot type from the largest detected subject bbox."""
            obj_elems = [e for e in elems if e.get("type") == "obj"]
            if not obj_elems:
                return "wide shot"
            largest = max(
                obj_elems,
                key=lambda e: (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]),
            )
            b = largest["bbox"]  # [y1, x1, y2, x2] on 0-1000 scale
            size = max((b[2] - b[0]) / 1000.0, (b[3] - b[1]) / 1000.0)
            if size > 0.65:
                return "close-up"
            elif size > 0.35:
                return "medium shot"
            else:
                return "wide shot"

        # ---- Florence-2 light source detection ----
        self.set_phase(inputs, "Light")
        light_element = None
        try:
            rec_result = self._run_task(
                model, processor, image,
                "<REFERRING_EXPRESSION_COMPREHENSION>the main light source",
            )
            light_bbox_px = (rec_result.get("<REFERRING_EXPRESSION_COMPREHENSION>") or {}).get("bboxes", [[]])[0]
            if light_bbox_px:
                light_ib   = to_ideogram(light_bbox_px)
                light_desc = region_description(light_bbox_px) or "light source"
                light_dir, light_setting = detect_lighting(caption)
                light_element = {
                    "type": "obj",
                    "bbox": light_ib,
                    "desc": f"{light_desc} ({light_setting}, from {light_dir})",
                }
        except Exception:
            pass
        if light_element is None:
            light_dir, light_setting = detect_lighting(caption)

        shot_type  = detect_shot_type(ordered)
        _unique = _strip_caption_prefix(caption) if caption else ""
        _unique_lc = (_unique[0].lower() + _unique[1:]) if _unique else ""
        scene_desc = f"{shot_type.capitalize()} of {_unique_lc}" if _unique_lc else shot_type.capitalize()
        # Frame outline(s): one per storyboard/collage panel when a grid is
        # detected, otherwise a single full-image frame.
        panels = self._detect_panels(image)
        if len(panels) >= 2:
            frame_elements = [
                {"type": "obj", "bbox": pb, "desc": f"Storyboard panel {i + 1}"}
                for i, pb in enumerate(panels)
            ]
        else:
            frame_elements = [
                {"type": "obj", "bbox": [0, 0, 1000, 1000], "desc": f"Frame – {shot_type}"}
            ]
        extra = [light_element] if light_element else []
        result = {
            "high_level_description": scene_desc,
            "style_description": infer_style(caption),
            "light_direction": light_dir,
            "light_setting":   light_setting,
            "compositional_deconstruction": {
                "background": background or caption or "Background inferred from image.",
                "elements":   frame_elements + extra + ordered,
            },
        }

        output = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        print("Florence-2 Ideogram4 JSON:", output[:300])
        return output
