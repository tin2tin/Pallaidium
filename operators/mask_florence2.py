"""Florence-2 Ideogram4 → Blender Mask Editor integration.

After a Florence-2 Ideogram4 generation job completes with
`florence2_send_to_mask` enabled, `apply_florence_json_to_mask` is called on
the main thread to:
  - create / update a named Mask in bpy.data.masks
  - add one rectangle mask layer per JSON element
  - open (or reuse) an Image Editor area in MASK mode with the source image
    as background

Per-layer metadata is stored as a CollectionProperty on Florence2MaskProps
(which lives on bpy.types.Mask, an ID subtype).  Each item's .name matches
the corresponding MaskLayer.name so lookups are O(1) via collection.get().
MaskLayer itself supports neither PointerProperty nor IDProperties.
"""

import json
import os

import bpy
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty,
    IntProperty, PointerProperty, StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class Florence2LayerData(PropertyGroup):
    """Metadata for one Florence-2 mask layer.  .name == MaskLayer.name."""
    f2_type:  EnumProperty(
        name="Type",
        items=[("obj", "Object", ""), ("text", "Text", "")],
        default="obj",
    )
    f2_desc:  StringProperty(name="Description")
    f2_text:  StringProperty(name="Text Content")
    f2_color: StringProperty(name="Color")
    f2_font:  StringProperty(name="Font")
    f2_bbox:  StringProperty(name="BBox JSON", default="[0,0,1000,1000]")


class Florence2MaskProps(PropertyGroup):
    f2_high_level_description: StringProperty(name="Scene Description")
    f2_background:             StringProperty(name="Background")
    f2_style_json:             StringProperty(name="Style JSON")
    f2_layers:                 CollectionProperty(type=Florence2LayerData)


# ---------------------------------------------------------------------------
# Per-layer helpers
# ---------------------------------------------------------------------------

def _get_layer_data(layer, mask) -> "Florence2LayerData | None":
    """Return the Florence2LayerData item for this layer, or None."""
    return mask.florence2_props.f2_layers.get(layer.name)


def _layer_is_florence(layer, mask) -> bool:
    return mask.florence2_props.f2_layers.get(layer.name) is not None


def _add_layer_data(mask, layer_name: str, elem: dict) -> "Florence2LayerData":
    col = mask.florence2_props.f2_layers
    item = col.get(layer_name)
    if item is None:
        item = col.add()
        item.name = layer_name
    item.f2_type  = elem.get("type",  "obj")
    item.f2_desc  = elem.get("desc",  "")
    item.f2_text  = elem.get("text",  "")
    item.f2_color = elem.get("color", "")
    item.f2_font  = elem.get("font",  "")
    item.f2_bbox  = json.dumps(elem.get("bbox", [0, 0, 1000, 1000]))
    return item


def _clear_layer_data(mask) -> None:
    mask.florence2_props.f2_layers.clear()


# ---------------------------------------------------------------------------
# Color / name conventions
# ---------------------------------------------------------------------------

_FACE_WORDS = frozenset(
    ("person", "man", "woman", "boy", "girl", "child", "face", "human", "people")
)

def _layer_fill_color(elem_type: str, desc: str):
    if elem_type == "text":
        return (1.0, 0.5, 0.0, 0.5)
    if any(w in desc.lower().split() for w in _FACE_WORDS):
        return (0.0, 0.7, 1.0, 0.5)
    return (0.2, 0.8, 0.2, 0.5)


def _layer_name(elem_type: str, desc: str, text: str) -> str:
    if elem_type == "text" and text:
        return f'[T] "{text[:24]}"'[:63]
    if elem_type == "text":
        return f"[T] {desc[:40]}"[:63]
    if any(w in desc.lower().split() for w in _FACE_WORDS):
        return f"[F] {desc[:40]}"[:63]
    return desc[:63]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _bbox_to_mask(bbox, W: int, H: int):
    """Ideogram [y1,x1,y2,x2] 0-1000 → mask center + half-sizes.

    Blender mask coordinate space (observed):
      X: (x/1000 - 0.5)*aspect + 0.5  →  range [0.5-aspect/2, 0.5+aspect/2]
      Y: 1 - y/1000                   →  lower-left origin, range [0, 1]
    Ideogram y=0 is the image TOP, so Y must be flipped.
    """
    y1, x1, y2, x2 = bbox
    aspect = W / H if H else 1.0
    cx = ((x1 + x2) / 2 / 1000 - 0.5) * aspect + 0.5   # X: centred+aspect, then +0.5 offset
    cy = 1.0 - (y1 + y2) / 2 / 1000                    # Y: lower-left origin [0, 1]
    hx = (x2 - x1) / 1000 / 2 * aspect
    hy = (y2 - y1) / 1000 / 2
    return cx, cy, hx, hy


def _spline_to_bbox(spline, W: int, H: int):
    """Live spline corners → Ideogram [y1,x1,y2,x2] 0-1000."""
    aspect = W / H if H else 1.0
    pts = [p.co for p in spline.points]
    if not pts:
        return [0, 0, 1000, 1000]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = round(((min(xs) - 0.5) / aspect + 0.5) * 1000)
    x2 = round(((max(xs) - 0.5) / aspect + 0.5) * 1000)
    y1 = round((1.0 - max(ys)) * 1000)
    y2 = round((1.0 - min(ys)) * 1000)
    clamp = lambda v: max(0, min(1000, v))
    return [clamp(y1), clamp(x1), clamp(y2), clamp(x2)]


# ---------------------------------------------------------------------------
# Image loading (no operators)
# ---------------------------------------------------------------------------

def _load_image(source_image_path: str):
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
# Core: build mask from JSON
# ---------------------------------------------------------------------------

def apply_florence_json_to_mask(json_str: str, source_image_path: str) -> None:
    """Parse Ideogram4 JSON, populate a Mask, open in Image Editor. Main thread."""
    print(f"[Florence2Mask] apply called, source={source_image_path!r}")

    try:
        data = json.loads(json_str)
    except Exception as exc:
        print(f"[Florence2Mask] JSON parse error: {exc}")
        return

    elements = data.get("compositional_deconstruction", {}).get("elements", [])
    print(f"[Florence2Mask] {len(elements)} element(s)")

    img = _load_image(source_image_path)
    W = img.size[0] if img else 1920
    H = img.size[1] if img else 1080

    # Get or create mask
    if source_image_path:
        mask_name = os.path.splitext(os.path.basename(source_image_path))[0][:63]
    else:
        mask_name = "Florence2"
    mask = bpy.data.masks.get(mask_name) or bpy.data.masks.new(name=mask_name)
    print(f"[Florence2Mask] Mask: {mask_name!r}")

    # Store mask-level props
    mp = mask.florence2_props
    mp.f2_high_level_description = data.get("high_level_description", "")
    mp.f2_background = data.get("compositional_deconstruction", {}).get("background", "")
    mp.f2_style_json = json.dumps(data.get("style_description", {}), ensure_ascii=False)

    # Clear everything
    for layer in list(mask.layers):
        mask.layers.remove(layer)
    _clear_layer_data(mask)

    # Add one layer per element
    for elem in elements[:40]:
        elem_type = elem.get("type", "obj")
        desc      = (elem.get("desc") or "element").strip()
        text_val  = (elem.get("text") or "").strip()
        bbox      = elem.get("bbox") or [0, 0, 1000, 1000]

        layer = mask.layers.new(name=_layer_name(elem_type, desc, text_val))
        actual_name = layer.name

        try:
            layer.fill_color = _layer_fill_color(elem_type, desc)
        except AttributeError:
            pass

        _add_layer_data(mask, actual_name, {
            "type":  elem_type,
            "desc":  desc,
            "text":  text_val,
            "color": elem.get("color", ""),
            "font":  elem.get("font",  ""),
            "bbox":  bbox,
        })

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
        print(f"[Florence2Mask]   {actual_name!r}  {elem_type}  {bbox}")

    if mask.layers:
        mask.active_layer_index = 0

    _open_mask_in_editor(mask, img)
    print(f"[Florence2Mask] Done — {len(elements)} layer(s) in {mask_name!r}")


# ---------------------------------------------------------------------------
# Area management
# ---------------------------------------------------------------------------

def _open_mask_in_editor(mask, img) -> None:
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
                print("[Florence2Mask] No suitable area to split — open Image Editor manually.")
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
                print(f"[Florence2Mask] Background image: {img.name!r}")
            except Exception as exc:
                print(f"[Florence2Mask] Could not set image: {exc}")
        try:
            space.mask = mask
            print(f"[Florence2Mask] Mask set: {mask.name!r}")
        except Exception as exc:
            print(f"[Florence2Mask] Could not set mask: {exc}")

        try:
            region = next((r for r in image_ed.regions if r.type == "WINDOW"), None)
            if region:
                with bpy.context.temp_override(window=window, area=image_ed, region=region):
                    bpy.ops.image.view_all(fit_view=True)
                print("[Florence2Mask] View fitted")
        except Exception as exc:
            print(f"[Florence2Mask] view_all failed: {exc}")
        return


# ---------------------------------------------------------------------------
# Sidebar panel
# ---------------------------------------------------------------------------

class FLORENCE2_PT_mask_panel(Panel):
    bl_label       = "Florence-2"
    bl_space_type  = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Florence-2"

    def draw(self, context):
        layout = self.layout
        space  = context.space_data

        if getattr(space, "mode", "") != "MASK":
            col = layout.column(align=True)
            col.label(text="Switch Image Editor to Mask mode,", icon="INFO")
            col.label(text="or run Florence-2 Ideogram 4")
            col.label(text="from the Generate panel.")
            return

        mask = getattr(space, "mask", None)
        if mask is None:
            layout.label(text="Run Florence-2 Ideogram 4", icon="INFO")
            layout.label(text="from the Generate panel.")
            return

        mp = mask.florence2_props

        # ── Scene-level description ──────────────────────────────────────────
        box = layout.box()
        box.label(text="Scene", icon="SCENE_DATA")
        box.prop(mp, "f2_high_level_description", text="")
        box.label(text="Background:")
        box.prop(mp, "f2_background", text="")

        # ── Active layer — indexed via active_layer_index for reliable sync ──
        idx    = getattr(mask, "active_layer_index", 0)
        n      = len(mask.layers)
        active = mask.layers[idx] if n and 0 <= idx < n else None

        if active is None:
            layout.separator()
            layout.operator("florence2.export_strip", icon="SEQUENCE")
            return

        layout.separator()

        # ── Layer stack controls ─────────────────────────────────────────────
        row = layout.row(align=True)
        row.operator("mask.layer_new",    text="", icon="ADD")
        row.operator("mask.layer_remove", text="", icon="REMOVE")
        row.separator()
        row.operator("mask.layer_move", text="", icon="TRIA_UP").type   = "UP"
        row.operator("mask.layer_move", text="", icon="TRIA_DOWN").type = "DOWN"
        row.separator()
        row.operator("mask.select_all", text="", icon="RESTRICT_SELECT_OFF").action = "SELECT"
        row.operator("mask.handle_type_set", text="", icon="IPO_CONSTANT").type = "VECTOR"

        box = layout.box()

        # Header: layer counter + visibility toggles
        hdr = box.row(align=True)
        hdr.label(text=f"Layer {idx + 1} / {n}", icon="LAYER_ACTIVE")
        for attr in ("hide", "hide_select", "hide_render"):
            try:
                hdr.prop(active, attr, text="", emboss=False)
            except TypeError:
                pass

        # Editable layer name
        box.prop(active, "name", text="Name")

        # Built-in MaskLayer properties
        row = box.row(align=True)
        try:
            row.prop(active, "alpha", text="Opacity")
        except TypeError:
            pass
        try:
            row.prop(active, "invert", toggle=True, text="Invert")
        except TypeError:
            pass
        try:
            box.prop(active, "blend", text="Blend")
        except TypeError:
            pass
        try:
            box.prop(active, "falloff", text="Falloff")
        except TypeError:
            pass
        try:
            box.prop(active, "fill_color", text="Fill Color")
        except TypeError:
            pass

        # ── Current spline position ──────────────────────────────────────────
        if active.splines:
            img = getattr(space, "image", None)
            W   = img.size[0] if img else 1920
            H   = img.size[1] if img else 1080
            try:
                y1, x1, y2, x2 = _spline_to_bbox(active.splines[0], W, H)
                pbox = box.box()
                pbox.label(text="Position (0 – 1000):", icon="TRANSFORM_ORIGINS")
                row = pbox.row(align=True)
                row.label(text=f"X  {x1} – {x2}")
                row.label(text=f"Y  {y1} – {y2}")
            except Exception:
                pass

        # ── Florence-2 metadata ──────────────────────────────────────────────
        ld = _get_layer_data(active, mask)
        if ld:
            layout.separator()
            fbox = layout.box()
            fbox.label(text="Florence-2 Metadata", icon="NODE_COMPOSITING")
            fbox.prop(ld, "f2_type", text="Type")
            fbox.label(text="Description:")
            fbox.prop(ld, "f2_desc", text="")
            if ld.f2_type == "text":
                fbox.label(text="Text Content:")
                fbox.prop(ld, "f2_text", text="")
                row = fbox.row(align=True)
                row.prop(ld, "f2_color", text="Color")
                row.prop(ld, "f2_font",  text="Font")
        else:
            layout.label(text="(Non-Florence layer)", icon="INFO")

        layout.separator()
        layout.operator("florence2.export_strip", icon="SEQUENCE")


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------

class FLORENCE2_OT_export_strip(Operator):
    bl_idname      = "florence2.export_strip"
    bl_label       = "Export JSON to Strip"
    bl_description = "Re-serialize mask bounding boxes to an Ideogram 4 JSON text strip"

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
        mp    = mask.florence2_props

        elements = []
        for layer in mask.layers:
            ld = _get_layer_data(layer, mask)
            if not ld:
                continue
            bbox = (_spline_to_bbox(layer.splines[0], W, H) if layer.splines
                    else json.loads(ld.f2_bbox or "[0,0,1000,1000]"))

            elem = {"type": ld.f2_type, "bbox": bbox, "desc": ld.f2_desc}
            if ld.f2_type == "text":
                elem["text"]  = ld.f2_text
                elem["color"] = ld.f2_color
                elem["font"]  = ld.f2_font
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

        self.report({"INFO"}, f"JSON strip at frame {frame_start}, channel {channel}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Panel sync helpers
# ---------------------------------------------------------------------------

_msgbus_owner = object()


def _tag_image_editors_redraw():
    """Tag every Image Editor in MASK mode for redraw."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    space = area.spaces[0]
                    if getattr(space, "mode", "") == "MASK":
                        area.tag_redraw()
    except Exception:
        pass


def _depsgraph_handler(scene, depsgraph):
    """Redraw panel whenever any Mask data-block is modified.

    slide_point and other mask operators mutate spline point selection on the
    Mask ID, which travels through the depsgraph but does NOT fire the
    active_layer_index msgbus subscription.  Watching depsgraph updates covers
    point selection, handle drags, and layer changes in one place.
    """
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Mask):
            _tag_image_editors_redraw()
            return


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = [
    Florence2LayerData,    # must be before Florence2MaskProps (CollectionProperty ref)
    Florence2MaskProps,
    FLORENCE2_OT_export_strip,
    FLORENCE2_PT_mask_panel,
]


def register():
    print("[Florence2Mask] Registering...")
    for cls in classes:
        bpy.utils.register_class(cls)
        print(f"[Florence2Mask]   {cls.__name__}")
    bpy.types.Mask.florence2_props = PointerProperty(type=Florence2MaskProps)

    # msgbus: fires when active_layer_index changes (layer list clicks)
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.Mask, "active_layer_index"),
        owner=_msgbus_owner,
        args=(),
        notify=_tag_image_editors_redraw,
    )

    # depsgraph handler: fires when mask data changes (slide_point, handle drags)
    if _depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_handler)

    print("[Florence2Mask] Registration complete")


def unregister():
    if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if hasattr(bpy.types.Mask, "florence2_props"):
        del bpy.types.Mask.florence2_props
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    print("[Florence2Mask] Unregistered")
