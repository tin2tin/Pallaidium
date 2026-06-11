"""Florence-2 Ideogram4 → Blender Mask Editor integration.

After a Florence-2 Ideogram4 generation job completes with
`florence2_send_to_mask` enabled, `apply_florence_json_to_mask` is called on
the main thread to:
  - create / update a named Mask in bpy.data.masks
  - add one rectangle mask layer per JSON element
  - open (or reuse) an Image Editor area in MASK mode with the source image
    as background
The sidebar panel (Florence-2 tab) shows the properties of the active layer
and provides an "Export JSON to Strip" button.

All per-layer metadata is stored as JSON in Florence2MaskProps.f2_layers_json
(keyed by layer name) because bpy.types.MaskLayer supports neither
PointerProperty nor IDProperties (custom props).
"""

import json
import os

import bpy
from bpy.props import PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup


# ---------------------------------------------------------------------------
# Property group — mask-level (bpy.types.Mask IS an ID subtype)
# ---------------------------------------------------------------------------

class Florence2MaskProps(PropertyGroup):
    f2_high_level_description: StringProperty(name="Description")
    f2_background:             StringProperty(name="Background")
    f2_style_json:             StringProperty(name="Style JSON")
    # Per-layer metadata keyed by layer.name: {name: {type,desc,text,color,font,bbox}}
    f2_layers_json:            StringProperty(name="Layers JSON", default="{}")


# ---------------------------------------------------------------------------
# Per-layer metadata helpers (stored on the Mask, not on MaskLayer)
# ---------------------------------------------------------------------------

def _get_layers_data(mask) -> dict:
    try:
        return json.loads(mask.florence2_props.f2_layers_json or "{}")
    except Exception:
        return {}


def _set_layers_data(mask, data: dict) -> None:
    mask.florence2_props.f2_layers_json = json.dumps(data, ensure_ascii=False)


def _layer_is_florence(layer, mask) -> bool:
    return layer.name in _get_layers_data(mask)


def _get_layer_data(layer, mask) -> dict:
    return _get_layers_data(mask).get(layer.name, {})


# ---------------------------------------------------------------------------
# Layer / color conventions
# ---------------------------------------------------------------------------

_FACE_WORDS = frozenset(
    ("person", "man", "woman", "boy", "girl", "child", "face", "human", "people")
)

def _layer_fill_color(elem_type: str, desc: str):
    if elem_type == "text":
        return (1.0, 0.5, 0.0, 0.5)
    words = desc.lower().split()
    if any(w in _FACE_WORDS for w in words):
        return (0.0, 0.7, 1.0, 0.5)
    return (0.2, 0.8, 0.2, 0.5)


def _layer_name(elem_type: str, desc: str, text: str) -> str:
    if elem_type == "text" and text:
        return f'[T] "{text[:24]}"'[:63]
    if elem_type == "text":
        return f"[T] {desc[:40]}"[:63]
    words = desc.lower().split()
    if any(w in _FACE_WORDS for w in words):
        return f"[F] {desc[:40]}"[:63]
    return desc[:63]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _bbox_to_mask(bbox, W: int, H: int):
    """Ideogram [y1,x1,y2,x2] 0-1000 → mask center + half-sizes."""
    y1, x1, y2, x2 = bbox
    aspect = W / H if H else 1.0
    cx = ((x1 + x2) / 2 / 1000 - 0.5) * aspect
    cy = -((y1 + y2) / 2 / 1000 - 0.5)
    hx = (x2 - x1) / 1000 / 2 * aspect
    hy = (y2 - y1) / 1000 / 2
    return cx, cy, hx, hy


def _spline_to_bbox(spline, W: int, H: int):
    """Read current spline corners → [y1,x1,y2,x2] 0-1000."""
    aspect = W / H if H else 1.0
    pts = [p.co for p in spline.points]
    if not pts:
        return [0, 0, 1000, 1000]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = round((min(xs) / aspect + 0.5) * 1000)
    x2 = round((max(xs) / aspect + 0.5) * 1000)
    y1 = round((-max(ys) + 0.5) * 1000)
    y2 = round((-min(ys) + 0.5) * 1000)
    clamp = lambda v: max(0, min(1000, v))
    return [clamp(y1), clamp(x1), clamp(y2), clamp(x2)]


# ---------------------------------------------------------------------------
# Image loading (no operators)
# ---------------------------------------------------------------------------

def _load_image(source_image_path: str):
    """Return bpy.data.images entry for path, loading it if needed."""
    if not source_image_path or not os.path.isfile(source_image_path):
        print(f"[Florence2Mask] Image not found: {source_image_path!r}")
        return None
    abs_src = os.path.normcase(os.path.abspath(source_image_path))
    for existing in bpy.data.images:
        try:
            if os.path.normcase(os.path.abspath(bpy.path.abspath(existing.filepath))) == abs_src:
                print(f"[Florence2Mask] Reusing existing image: {existing.name!r}")
                return existing
        except Exception:
            pass
    try:
        img = bpy.data.images.load(source_image_path)
        print(f"[Florence2Mask] Loaded image: {img.name!r}  size={img.size[:]}")
        return img
    except Exception as exc:
        print(f"[Florence2Mask] Could not load image: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core utility — called from main_ops / queue_ops after generation
# ---------------------------------------------------------------------------

def apply_florence_json_to_mask(json_str: str, source_image_path: str) -> None:
    """Parse JSON, build/update a Mask, open the mask editor. Main-thread only."""
    print(f"[Florence2Mask] apply_florence_json_to_mask called, source={source_image_path!r}")

    try:
        data = json.loads(json_str)
    except Exception as exc:
        print(f"[Florence2Mask] JSON parse error: {exc}")
        return

    elements = data.get("compositional_deconstruction", {}).get("elements", [])
    print(f"[Florence2Mask] Parsed JSON — {len(elements)} element(s)")

    # --- Load source image (no ops) ---
    img = _load_image(source_image_path)
    W = img.size[0] if img else 1920
    H = img.size[1] if img else 1080
    print(f"[Florence2Mask] Canvas size: {W}×{H}")

    # --- Get / create Mask (no ops) ---
    if source_image_path:
        mask_name = os.path.splitext(os.path.basename(source_image_path))[0][:63]
    else:
        mask_name = "Florence2"
    mask = bpy.data.masks.get(mask_name)
    if mask is None:
        mask = bpy.data.masks.new(name=mask_name)
        print(f"[Florence2Mask] Created new mask: {mask_name!r}")
    else:
        print(f"[Florence2Mask] Reusing existing mask: {mask_name!r}")

    # --- Store mask-level properties ---
    mp = mask.florence2_props
    mp.f2_high_level_description = data.get("high_level_description", "")
    mp.f2_background = data.get("compositional_deconstruction", {}).get("background", "")
    mp.f2_style_json = json.dumps(data.get("style_description", {}), ensure_ascii=False)

    # --- Remove stale Florence layers (identified via f2_layers_json) ---
    old_data = _get_layers_data(mask)
    removed = 0
    for layer in list(mask.layers):
        if layer.name in old_data:
            mask.layers.remove(layer)
            removed += 1
    print(f"[Florence2Mask] Removed {removed} stale layer(s)")

    # --- Add one layer per element; build new layers dict ---
    new_layers_data = {}
    for elem in elements[:40]:
        elem_type = elem.get("type", "obj")
        desc      = (elem.get("desc") or "element").strip()
        text_val  = (elem.get("text") or "").strip()
        bbox      = elem.get("bbox") or [0, 0, 1000, 1000]

        name = _layer_name(elem_type, desc, text_val)
        # Ensure unique name (Blender may suffix duplicates)
        layer = mask.layers.new(name=name)
        actual_name = layer.name  # Blender may have suffixed it

        # Fill color by type
        try:
            layer.fill_color = _layer_fill_color(elem_type, desc)
        except AttributeError:
            pass

        # Store metadata on the Mask (MaskLayer supports neither PointerProperty
        # nor IDProperties, so we serialise to the mask-level JSON blob)
        new_layers_data[actual_name] = {
            "type":  elem_type,
            "desc":  desc,
            "text":  text_val,
            "color": elem.get("color", ""),
            "font":  elem.get("font",  ""),
            "bbox":  bbox,
        }

        # Rectangle spline
        cx, cy, hx, hy = _bbox_to_mask(bbox, W, H)
        hx = max(hx, 0.001)
        hy = max(hy, 0.001)
        spline = layer.splines.new()
        spline.use_cyclic = True
        spline.points.add(3)
        corners = [
            (cx - hx, cy - hy),
            (cx + hx, cy - hy),
            (cx + hx, cy + hy),
            (cx - hx, cy + hy),
        ]
        for i, (px, py) in enumerate(corners):
            pt = spline.points[i]
            pt.co           = (px, py)
            pt.handle_left  = (px, py)
            pt.handle_right = (px, py)
            try:
                pt.handle_type = "VECTOR"
            except AttributeError:
                pass

        print(f"[Florence2Mask]   layer {actual_name!r}  type={elem_type}  bbox={bbox}")

    _set_layers_data(mask, new_layers_data)

    if mask.layers:
        mask.layers.active_index = 0

    print(f"[Florence2Mask] Created {len(new_layers_data)} layer(s) in mask {mask_name!r}")

    # --- Open / configure an Image Editor area (no image.open op needed) ---
    _open_mask_in_editor(mask, img)


# ---------------------------------------------------------------------------
# Area management (no operators except area_split as soft fallback)
# ---------------------------------------------------------------------------

def _open_mask_in_editor(mask, img) -> None:
    """Configure an Image Editor area in MASK mode with mask and image set."""
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        areas  = screen.areas

        image_ed = next((a for a in areas if a.type == "IMAGE_EDITOR"), None)

        if image_ed is None:
            target = next(
                (a for a in areas
                 if a.type == "SEQUENCE_EDITOR"
                 and hasattr(a.spaces[0], "view_type")
                 and a.spaces[0].view_type == "PREVIEW"),
                None,
            ) or next((a for a in areas if a.type == "SEQUENCE_EDITOR"), None)

            if target is None:
                print("[Florence2Mask] No Image Editor or Sequencer area found — open one manually.")
                return

            ids_before = {id(a) for a in areas}
            try:
                region = next(r for r in target.regions if r.type == "WINDOW")
                with bpy.context.temp_override(window=window, area=target, region=region):
                    bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
            except Exception as exc:
                print(f"[Florence2Mask] Area split failed: {exc}")
                return

            image_ed = next((a for a in screen.areas if id(a) not in ids_before), None)
            if image_ed is None:
                print("[Florence2Mask] Could not locate new area after split.")
                return
            image_ed.type = "IMAGE_EDITOR"

        space = image_ed.spaces[0]
        try:
            space.mode = "MASK"
        except Exception as exc:
            print(f"[Florence2Mask] Could not set MASK mode: {exc}")
        if img is not None:
            try:
                space.image = img
                print(f"[Florence2Mask] Set image editor background: {img.name!r}")
            except Exception as exc:
                print(f"[Florence2Mask] Could not set image: {exc}")
        try:
            space.mask = mask
            print(f"[Florence2Mask] Set mask in editor: {mask.name!r}")
        except Exception as exc:
            print(f"[Florence2Mask] Could not set mask: {exc}")
        return


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class FLORENCE2_PT_mask_panel(Panel):
    bl_label       = "Florence-2"
    bl_space_type  = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Florence-2"

    def draw(self, context):
        layout = self.layout
        col    = layout.column(align=True)
        space  = context.space_data

        if getattr(space, "mode", "") != "MASK":
            col.label(text="Switch Image Editor to Mask mode,", icon="INFO")
            col.label(text="or run Florence-2 Ideogram 4")
            col.label(text="from the Generate panel.")
            return

        mask = getattr(space, "mask", None)
        if mask is None:
            col.label(text="Run Florence-2 Ideogram 4", icon="INFO")
            col.label(text="from the Generate panel.")
            return

        mp = mask.florence2_props
        if not mp.f2_high_level_description:
            col.label(text="No Florence-2 data on this mask.", icon="INFO")
            col.operator("florence2.export_strip", icon="SEQUENCE")
            return

        # Mask-level description
        box = col.box()
        box.label(text="Scene", icon="SCENE_DATA")
        box.prop(mp, "f2_high_level_description", text="")
        box.label(text="Background:")
        box.prop(mp, "f2_background", text="")

        # Active layer
        active = mask.layers.active
        if active:
            ld = _get_layer_data(active, mask)
            if ld:
                f2_type = ld.get("type", "obj")
                f2_desc = ld.get("desc", "")
                box = col.box()
                icon = "FONT_DATA" if f2_type == "text" else (
                    "COMMUNITY" if any(w in f2_desc.lower().split() for w in _FACE_WORDS)
                    else "OBJECT_DATA"
                )
                box.label(text=active.name, icon=icon)
                box.label(text=f"Type: {f2_type}")
                box.label(text="Description:")
                box.label(text=f2_desc, icon="NONE")
                if f2_type == "text":
                    box.label(text=f"Text:  {ld.get('text', '')}")
                    box.label(text=f"Color: {ld.get('color', '')}")
                    box.label(text=f"Font:  {ld.get('font', '')}")
            else:
                col.label(text="(Non-Florence layer selected)", icon="INFO")

        col.separator()
        col.operator("florence2.export_strip", icon="SEQUENCE")


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------

class FLORENCE2_OT_export_strip(Operator):
    bl_idname      = "florence2.export_strip"
    bl_label       = "Export JSON to Strip"
    bl_description = "Serialize mask elements to an Ideogram 4 JSON text strip in the sequencer"

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        if not space or space.type != "IMAGE_EDITOR":
            return False
        return getattr(space, "mask", None) is not None

    def execute(self, context):
        space = context.space_data
        mask  = space.mask
        img   = getattr(space, "image", None)
        W     = img.size[0] if img else 1920
        H     = img.size[1] if img else 1080

        mp         = mask.florence2_props
        layers_data = _get_layers_data(mask)

        elements = []
        for layer in mask.layers:
            ld = layers_data.get(layer.name)
            if not ld:
                continue
            bbox = _spline_to_bbox(layer.splines[0], W, H) if layer.splines else ld.get("bbox", [0, 0, 1000, 1000])

            elem = {"type": ld["type"], "bbox": bbox, "desc": ld.get("desc", "")}
            if ld["type"] == "text":
                elem["text"]  = ld.get("text",  "")
                elem["color"] = ld.get("color", "")
                elem["font"]  = ld.get("font",  "")
            elements.append(elem)

        try:
            style = json.loads(mp.f2_style_json) if mp.f2_style_json else {}
        except Exception:
            style = {}

        result = {
            "high_level_description": mp.f2_high_level_description,
            "style_description": style,
            "compositional_deconstruction": {
                "background": mp.f2_background,
                "elements":   elements,
            },
        }
        json_str = json.dumps(result, separators=(",", ":"), ensure_ascii=False)

        seq_scene = getattr(context, "sequencer_scene", None) or context.scene
        if not seq_scene.sequence_editor:
            seq_scene.sequence_editor_create()

        frame_start = seq_scene.frame_current
        duration    = 100
        from ..utils.helpers import find_first_empty_channel
        channel = find_first_empty_channel(frame_start, frame_start + duration)

        strip = seq_scene.sequence_editor.strips.new_effect(
            name=json_str[:63],
            type="TEXT",
            frame_start=frame_start,
            length=duration,
            channel=channel,
        )
        strip.text       = json_str
        strip.wrap_width = 0.0
        strip.font_size  = 12

        self.report({"INFO"}, f"JSON strip inserted at frame {frame_start}, channel {channel}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = [
    Florence2MaskProps,
    FLORENCE2_OT_export_strip,
    FLORENCE2_PT_mask_panel,
]


def register():
    print("[Florence2Mask] Registering classes...")
    for cls in classes:
        bpy.utils.register_class(cls)
        print(f"[Florence2Mask]   registered {cls.__name__}")
    bpy.types.Mask.florence2_props = PointerProperty(type=Florence2MaskProps)
    print("[Florence2Mask] florence2_props assigned to bpy.types.Mask — registration complete")


def unregister():
    if hasattr(bpy.types.Mask, "florence2_props"):
        del bpy.types.Mask.florence2_props
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    print("[Florence2Mask] Unregistered")
