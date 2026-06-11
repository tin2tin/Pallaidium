"""Image captioning and Ideogram-4 prompt extraction via Florence-2."""

import json

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs

try:
    import bpy as _bpy
    if not hasattr(_bpy.types.Scene, "florence2_mode"):
        _bpy.types.Scene.florence2_mode = _bpy.props.EnumProperty(
            name="Mode",
            items=[
                ("CAPTION",   "Caption",    "Detailed image caption as plain text"),
                ("IDEOGRAM4", "Ideogram 4", "Extract structured Ideogram 4 prompt JSON"),
            ],
            default="CAPTION",
        )
except Exception:
    pass


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
        return False

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
        return processor.post_process_generation(
            generated_text,
            task=task,
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

    def _caption(self, model, processor, image, inputs) -> str:
        self.set_phase(inputs, "Captioning")
        parsed = self._run_task(model, processor, image, "<MORE_DETAILED_CAPTION>")
        text = parsed["<MORE_DETAILED_CAPTION>"]
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
        _FACE_LABELS = {"person", "man", "woman", "boy", "girl", "child", "face", "human", "people"}

        def is_face_label(label: str) -> bool:
            return label.lower().strip() in _FACE_LABELS

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
                return (result.get("<REGION_TO_DESCRIPTION>", "") or "").strip()
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
                style["photo"] = "photograph"
                style["medium"] = "digital camera"
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
        od_bboxes = od_data.get("bboxes", [])
        od_labels = od_data.get("labels", [])
        if od_bboxes:
            face_pass = 0
            for bbox_px, label in zip(od_bboxes, od_labels):
                ib   = to_ideogram(bbox_px)
                desc = label
                if is_face_label(label) and face_pass < 6:
                    face_pass += 1
                    self.set_phase(inputs, f"Face {face_pass}")
                    rich = region_description(bbox_px)
                    if rich:
                        desc = rich
                elif dense_items:
                    best = max(dense_items, key=lambda d: iou(ib, d["bbox"]))
                    if iou(ib, best["bbox"]) > 0.2:
                        desc = best["description"]
                elements.append({"type": "obj", "bbox": ib, "desc": desc, "label": label})
        elif dense_items:
            elements = [
                {"type": "obj", "bbox": d["bbox"], "desc": d["description"], "label": d["description"]}
                for d in dense_items[:20]
            ]

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
                ordered.append({
                    "type":  "text",
                    "bbox":  elem["bbox"],
                    "text":  elem.get("text", ""),
                    "color": elem.get("color", ""),
                    "font":  elem.get("font", ""),
                    "desc":  elem["desc"],
                })
            else:
                ordered.append({
                    "type": "obj",
                    "bbox": elem["bbox"],
                    "desc": elem["desc"],
                })

        result = {
            "high_level_description": caption or "Image scene.",
            "style_description": infer_style(caption),
            "compositional_deconstruction": {
                "background": background or caption or "Background inferred from image.",
                "elements":   ordered,
            },
        }

        output = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        print("Florence-2 Ideogram4 JSON:", output[:300])
        return output
