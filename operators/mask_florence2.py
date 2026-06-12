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
    FloatVectorProperty, IntProperty, PointerProperty, StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

class Florence2ColorEntry(PropertyGroup):
    """One hex color in a palette CollectionProperty."""
    def _rgb_update(self, _context):
        r, g, b = (max(0, min(255, round(c * 255))) for c in self.color_rgb)
        self.hex_color = f"#{r:02X}{g:02X}{b:02X}"

    hex_color: StringProperty(
        name="",
        description="#RRGGBB hex color",
        default="#FFFFFF",
        maxlen=7,
    )
    color_rgb: FloatVectorProperty(
        name="",
        subtype="COLOR",
        size=3,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0),
        update=_rgb_update,
    )


class Florence2LayerData(PropertyGroup):
    """Metadata for one Florence-2 mask layer.  .name == MaskLayer.name."""
    f2_type:  EnumProperty(
        name="Type",
        items=[("obj", "Object", ""), ("text", "Text", "")],
        default="obj",
    )
    f2_desc:  StringProperty(name="Description")
    f2_text:  StringProperty(name="Text Content")
    f2_color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    f2_font:  EnumProperty(
        name="Font Size",
        items=[
            ("small",  "Small",  "Small / caption text  (< 3 % of image height)"),
            ("medium", "Medium", "Medium body text (3 – 7 % of image height)"),
            ("large",  "Large",  "Large heading text (7 – 15 % of image height)"),
            ("huge",   "Huge",   "Headline / display text (> 15 % of image height)"),
        ],
        default="small",
    )
    f2_bbox:    StringProperty(name="BBox JSON", default="[0,0,1000,1000]")
    f2_palette: CollectionProperty(
        type=Florence2ColorEntry,
        name="Color Palette",
        description="Per-element palette, up to 5 #RRGGBB colors",
    )


class Florence2MaskProps(PropertyGroup):
    f2_high_level_description: StringProperty(name="Scene Description")
    f2_background:             StringProperty(name="Background")
    # style_description individual fields
    f2_is_photo:               BoolProperty(
        name="Photo",
        description="Photo/camera style uses photo+medium; art style uses medium+art_style",
        default=True,
    )
    f2_aesthetics:             StringProperty(name="Aesthetics")
    f2_lighting:               StringProperty(name="Lighting")
    f2_photo:                  StringProperty(name="Camera / Lens")
    f2_medium:                 StringProperty(name="Medium")
    f2_art_style:              StringProperty(name="Art Style")
    f2_style_palette:          CollectionProperty(
        type=Florence2ColorEntry,
        name="Style Palette",
        description="Image-wide palette, up to 16 #RRGGBB colors",
    )
    f2_style_json:             StringProperty(name="Style JSON")   # legacy fallback
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
    palette = elem.get("color_palette") or []
    _list_to_palette(item.f2_palette, palette)
    # f2_color kept for layer-fill display; derive from first palette entry or legacy key
    first = palette[0] if palette else elem.get("color", "")
    item.f2_color = _parse_color_string(first)
    item.f2_font  = _FONT_ALIAS_MAP.get((elem.get("font") or "").strip().lower(), "small")
    item.f2_bbox  = json.dumps(elem.get("bbox", [0, 0, 1000, 1000]))
    return item


def _clear_layer_data(mask) -> None:
    mask.florence2_props.f2_layers.clear()


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def _list_to_palette(col, hex_list: list) -> None:
    """Populate a CollectionProperty of Florence2ColorEntry from a list of hex strings."""
    col.clear()
    for h in hex_list:
        entry = col.add()
        entry.hex_color = str(h).upper()
        rgb = _parse_color_string(h)
        entry.color_rgb = rgb[:3]


def _palette_to_list(col) -> list:
    """Serialize a palette CollectionProperty to a list of uppercase hex strings."""
    return [e.hex_color.upper() for e in col if e.hex_color.strip()]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_CSS_NAMED_COLORS = {
    "red":     (1.000, 0.000, 0.000), "green":   (0.000, 0.502, 0.000),
    "blue":    (0.000, 0.000, 1.000), "white":   (1.000, 1.000, 1.000),
    "black":   (0.000, 0.000, 0.000), "yellow":  (1.000, 1.000, 0.000),
    "orange":  (1.000, 0.647, 0.000), "purple":  (0.502, 0.000, 0.502),
    "pink":    (1.000, 0.753, 0.796), "gray":    (0.502, 0.502, 0.502),
    "grey":    (0.502, 0.502, 0.502), "cyan":    (0.000, 1.000, 1.000),
    "magenta": (1.000, 0.000, 1.000), "brown":   (0.647, 0.165, 0.165),
    "silver":  (0.753, 0.753, 0.753), "gold":    (1.000, 0.843, 0.000),
}

_FONT_ALIAS_MAP = {
    "small":  "small",
    "medium": "medium",
    "large":  "large",
    "huge":   "huge",
}


def _parse_color_string(color_str: str) -> tuple:
    """Parse a CSS hex or named color to an (r, g, b, 1.0) float tuple."""
    s = (color_str or "").strip()
    if s.startswith("#"):
        h = s[1:]
        try:
            if len(h) == 3:
                r, g, b = (int(c + c, 16) / 255 for c in h)
            elif len(h) == 6:
                r = int(h[0:2], 16) / 255
                g = int(h[2:4], 16) / 255
                b = int(h[4:6], 16) / 255
            else:
                return (1.0, 1.0, 1.0, 1.0)
            return (r, g, b, 1.0)
        except ValueError:
            pass
    rgb = _CSS_NAMED_COLORS.get(s.lower())
    if rgb:
        return (rgb[0], rgb[1], rgb[2], 1.0)
    return (1.0, 1.0, 1.0, 1.0)


def _color_to_hex(rgba) -> str:
    """Convert an (r, g, b[, a]) float tuple to a #RRGGBB hex string."""
    r, g, b = (max(0, min(255, round(c * 255))) for c in rgba[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


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
        msg = f"JSON parse error: {exc}"
        print(f"[Florence2Mask] {msg}")
        try:
            bpy.context.window_manager.popup_menu(
                lambda self, _: self.layout.label(text=msg, icon="ERROR"),
                title="Box Editor — Invalid JSON",
                icon="ERROR",
            )
        except Exception:
            pass
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
    style = data.get("style_description", {})
    mp.f2_style_json = json.dumps(style, ensure_ascii=False)  # legacy fallback
    mp.f2_is_photo           = "photo" in style
    mp.f2_aesthetics         = style.get("aesthetics", "")
    mp.f2_lighting           = style.get("lighting", "")
    mp.f2_photo              = style.get("photo", "")
    mp.f2_medium             = style.get("medium", "")
    mp.f2_art_style          = style.get("art_style", "")
    _list_to_palette(mp.f2_style_palette, style.get("color_palette") or [])

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
            pt.select_control_point = False
            pt.select_left_handle   = False
            pt.select_right_handle  = False
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
            target = (
                next(
                    (a for a in areas
                     if a.type == "SEQUENCE_EDITOR"
                     and hasattr(a.spaces[0], "view_type")
                     and a.spaces[0].view_type == "PREVIEW"),
                    None,
                )
                or next((a for a in areas if a.type == "SEQUENCE_EDITOR"), None)
                or next((a for a in areas if a.type == "TEXT_EDITOR"), None)
            )

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
            except Exception:
                pass
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

        # Switch N-panel to the Box Editor tab
        try:
            ui_region = next((r for r in image_ed.regions if r.type == "UI"), None)
            if ui_region:
                ui_region.active_panel_category = "Box Editor"
                space.show_region_ui = True
                print("[Florence2Mask] Box Editor tab selected")
        except Exception as exc:
            print(f"[Florence2Mask] Tab switch failed: {exc}")
        return


# ---------------------------------------------------------------------------
# Palette drawing helper
# ---------------------------------------------------------------------------

def _draw_palette(layout, col, target: str, layer_name: str = "", limit: int = 16) -> None:
    """Draw each palette entry as a color widget + X button, plus an Add row."""
    for i, entry in enumerate(col):
        row = layout.row(align=True)
        row.prop(entry, "color_rgb", text="")
        op = row.operator("florence2.palette_remove", text="", icon="X")
        op.target     = target
        op.layer_name = layer_name
        op.index      = i
    if len(col) < limit:
        op = layout.operator("florence2.palette_add", text="+ Add Color", icon="ADD")
        op.target     = target
        op.layer_name = layer_name


# ---------------------------------------------------------------------------
# Sidebar panel
# ---------------------------------------------------------------------------

class FLORENCE2_PT_mask_panel(Panel):
    bl_label       = "Box Editor"
    bl_space_type  = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category    = "Box Editor"

    def draw(self, context):
        layout = self.layout
        space  = context.space_data

        if getattr(space, "mode", "") != "MASK":
            col = layout.column(align=True)
            col.label(text="Switch Image Editor to Mask mode,", icon="INFO")
            col.label(text="or use the button below:")
            col.operator("florence2.open_box_editor", text="Open Box Editor", icon="MOD_MASK")
            return

        mask = getattr(space, "mask", None)
        if mask is None:
            layout.label(text="No Box Editor mask active.", icon="INFO")
            layout.operator("florence2.open_box_editor", text="New Box Editor Mask", icon="ADD")
            return

        mp = mask.florence2_props

        # ── Scene-level description ──────────────────────────────────────────
        box = layout.box()
        box.label(text="Scene", icon="SCENE_DATA")
        box.textbox(mp, "f2_high_level_description", placeholder="Scene description...")
        box.label(text="Background:")
        box.textbox(mp, "f2_background", placeholder="Background description...")

        # ── Style description ────────────────────────────────────────────────
        box = layout.box()
        box.label(text="Style", icon="MATERIAL")
        row = box.row(align=True)
        row.prop(mp, "f2_is_photo", text="Photo", toggle=True)
        row.prop(mp, "f2_is_photo", text="Art", toggle=True, invert_checkbox=True)
        box.label(text="Aesthetics:")
        box.textbox(mp, "f2_aesthetics", placeholder="e.g. cinematic, vibrant, moody")
        box.label(text="Lighting:")
        box.textbox(mp, "f2_lighting", placeholder="e.g. golden hour, soft shadows")
        if mp.f2_is_photo:
            box.label(text="Camera / Lens:")
            box.textbox(mp, "f2_photo", placeholder="e.g. 35mm, f/1.4, eye-level")
            box.label(text="Medium:")
            box.textbox(mp, "f2_medium", placeholder="e.g. photograph")
        else:
            box.label(text="Medium:")
            box.textbox(mp, "f2_medium", placeholder="e.g. illustration, 3d_render")
            box.label(text="Art Style:")
            box.textbox(mp, "f2_art_style", placeholder="e.g. flat vector, bold outlines")
        box.label(text="Color Palette (up to 16):")
        _draw_palette(box, mp.f2_style_palette, "style", limit=16)

        layout.separator()

        # ── One box: layer list + toolbar + active layer details ─────────────
        box = layout.box()

        box.template_list(
            "MASK_UL_layers", "",
            mask, "layers",
            mask, "active_layer_index",
            rows=5,
        )

        # Layer stack operator toolbar
        row = box.row(align=True)
        row.operator("florence2.layer_new_with_square", text="", icon="ADD")
        row.operator("mask.layer_remove", text="", icon="REMOVE")
        row.separator()
        row.operator("mask.layer_move", text="", icon="TRIA_UP").direction   = "UP"
        row.operator("mask.layer_move", text="", icon="TRIA_DOWN").direction = "DOWN"
        row.separator()
        row.operator("mask.select_all", text="", icon="RESTRICT_SELECT_OFF").action = "SELECT"

        active = mask.layers.active
        if active is None:
            layout.separator()
            layout.operator("florence2.export_strip", icon="SEQUENCE")
            return

        box.separator()

        # Header: layer name + visibility toggles
        hdr = box.row(align=True)
        hdr.prop(active, "name", text="", icon="LAYER_ACTIVE")
        for attr in ("hide", "hide_select", "hide_render"):
            try:
                hdr.prop(active, attr, text="", emboss=False)
            except TypeError:
                pass

        # ── Florence-2 metadata ───────────────────────────────────────────────
        ld = _get_layer_data(active, mask)
        if ld:
            box.prop(ld, "f2_type", text="Type")
            box.label(text="Description:")
            box.textbox(ld, "f2_desc", placeholder="Object description...")
            if ld.f2_type == "text":
                box.label(text="Text Content:")
                box.textbox(ld, "f2_text", placeholder="Text to render in image...")
            box.label(text="Color Palette (up to 5):")
            _draw_palette(box, ld.f2_palette, "layer", layer_name=active.name, limit=5)
        else:
            box.label(text="(Non-Florence layer)", icon="INFO")

        layout.separator()
        layout.operator("florence2.export_strip", icon="SEQUENCE")


# ---------------------------------------------------------------------------
# Palette add / remove operators
# ---------------------------------------------------------------------------

def _resolve_palette(context, target: str, layer_name: str):
    """Return the correct CollectionProperty for palette ops, or None."""
    mask = getattr(getattr(context, "space_data", None), "mask", None)
    if not mask:
        return None
    if target == "style":
        return mask.florence2_props.f2_style_palette
    ld = mask.florence2_props.f2_layers.get(layer_name)
    return ld.f2_palette if ld else None


class FLORENCE2_OT_palette_add(Operator):
    bl_idname      = "florence2.palette_add"
    bl_label       = "Add Color"
    bl_description = "Add a color entry to this palette"
    bl_options     = {"REGISTER", "UNDO"}

    target:     EnumProperty(items=[("layer", "Layer", ""), ("style", "Style", "")], default="style")
    layer_name: StringProperty()

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space and space.type == "IMAGE_EDITOR" and getattr(space, "mask", None) is not None

    def execute(self, context):
        col = _resolve_palette(context, self.target, self.layer_name)
        if col is None:
            return {"CANCELLED"}
        entry = col.add()
        entry.hex_color = "#FFFFFF"
        return {"FINISHED"}


class FLORENCE2_OT_palette_remove(Operator):
    bl_idname      = "florence2.palette_remove"
    bl_label       = "Remove Color"
    bl_description = "Remove this color from the palette"
    bl_options     = {"REGISTER", "UNDO"}

    target:     EnumProperty(items=[("layer", "Layer", ""), ("style", "Style", "")], default="style")
    layer_name: StringProperty()
    index:      IntProperty(default=0)

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space and space.type == "IMAGE_EDITOR" and getattr(space, "mask", None) is not None

    def execute(self, context):
        col = _resolve_palette(context, self.target, self.layer_name)
        if col is None:
            return {"CANCELLED"}
        if 0 <= self.index < len(col):
            col.remove(self.index)
        return {"FINISHED"}


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

            # Per-element color_palette; for text fall back to f2_color if palette empty
            cp = _palette_to_list(ld.f2_palette)
            if not cp and ld.f2_type == "text":
                hex_c = _color_to_hex(ld.f2_color)
                if hex_c not in ("#FFFFFF", "#FEFEFE"):
                    cp = [hex_c]

            if ld.f2_type == "text":
                # Schema key order: type, bbox, text, desc, color_palette
                elem = {"type": "text", "bbox": bbox, "text": ld.f2_text, "desc": ld.f2_desc}
                if cp:
                    elem["color_palette"] = cp
            else:
                # Schema key order: type, bbox, desc, color_palette
                elem = {"type": "obj", "bbox": bbox, "desc": ld.f2_desc}
                if cp:
                    elem["color_palette"] = cp
            elements.append(elem)

        # Build style_description from individual props; fall back to legacy JSON
        if mp.f2_aesthetics or mp.f2_lighting or mp.f2_medium:
            style = {}
            if mp.f2_aesthetics:
                style["aesthetics"] = mp.f2_aesthetics
            if mp.f2_lighting:
                style["lighting"] = mp.f2_lighting
            if mp.f2_is_photo:
                # key order: aesthetics, lighting, photo, medium, color_palette
                if mp.f2_photo:
                    style["photo"] = mp.f2_photo
                if mp.f2_medium:
                    style["medium"] = mp.f2_medium
            else:
                # key order: aesthetics, lighting, medium, art_style, color_palette
                if mp.f2_medium:
                    style["medium"] = mp.f2_medium
                if mp.f2_art_style:
                    style["art_style"] = mp.f2_art_style
            sp = _palette_to_list(mp.f2_style_palette)
            if sp:
                style["color_palette"] = sp
        else:
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


class FLORENCE2_OT_layer_new_with_square(Operator):
    bl_idname      = "florence2.layer_new_with_square"
    bl_label       = "New Box Layer"
    bl_description = "Add a new Box Editor layer with a centered square spline"
    bl_options     = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        if not space or space.type != "IMAGE_EDITOR":
            return False
        return getattr(space, "mask", None) is not None

    def execute(self, context):
        global _last_active_layer_name

        space = context.space_data
        mask  = space.mask
        img   = getattr(space, "image", None)
        W     = img.size[0] if img else 1920
        H     = img.size[1] if img else 1080

        layer = mask.layers.new(name="New Layer")

        # Register Box Editor metadata BEFORE activating the layer, so the
        # depsgraph handler never sees it without f2_layers data.
        _add_layer_data(mask, layer.name, {
            "type": "obj", "desc": "", "text": "",
            "color": "", "font": "", "bbox": [400, 400, 600, 600],
        })

        try:
            layer.fill_color = (0.2, 0.8, 0.2, 0.5)
        except AttributeError:
            pass

        # Suppress the depsgraph handler's "active layer changed" branch for
        # this activation — the metadata already exists so no extra sync needed.
        _last_active_layer_name = layer.name

        for i, lyr in enumerate(mask.layers):
            if lyr == layer:
                mask.active_layer_index = i
                break

        # Centered square: bbox [y1,x1,y2,x2] = [400,400,600,600] on 0-1000 scale
        cx, cy, hx, hy = _bbox_to_mask([400, 400, 600, 600], W, H)
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
            pt.select_control_point = False
            pt.select_left_handle   = False
            pt.select_right_handle  = False
            try:
                pt.handle_type = "VECTOR"
            except AttributeError:
                pass

        return {"FINISHED"}


class FLORENCE2_OT_open_box_editor(Operator):
    bl_idname      = "florence2.open_box_editor"
    bl_label       = "Open Box Editor"
    bl_description = "Open (or switch) an Image Editor to Box Editor / Mask mode"

    def execute(self, context):
        # Reuse or create the mask ──────────────────────────────────────────
        mask = None

        # 1. Active Image Editor in MASK mode already has one?
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    sp = area.spaces[0]
                    if getattr(sp, "mode", "") == "MASK" and getattr(sp, "mask", None):
                        mask = sp.mask
                        break
            if mask:
                break

        # 2. Any Image Editor at all has a mask?
        if not mask:
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "IMAGE_EDITOR":
                        sp = area.spaces[0]
                        if getattr(sp, "mask", None):
                            mask = sp.mask
                            break
                if mask:
                    break

        # 3. Any mask already in bpy.data.masks?
        if not mask and bpy.data.masks:
            mask = bpy.data.masks[0]

        # 4. Create a fresh "Box Editor" mask and initialise its props
        if not mask:
            mask = bpy.data.masks.new(name="Box Editor")
            mp = mask.florence2_props
            mp.f2_high_level_description = ""
            mp.f2_background             = ""
            mp.f2_is_photo    = True
            mp.f2_aesthetics  = ""
            mp.f2_lighting    = ""
            mp.f2_photo       = ""
            mp.f2_medium      = ""
            mp.f2_art_style   = ""
            mp.f2_style_json  = "{}"
            print("[BoxEditor] Created new mask 'Box Editor' with florence2_props")

        # Open it in an Image Editor area ──────────────────────────────────
        _open_mask_in_editor(mask, None)
        return {"FINISHED"}


class FLORENCE2_OT_text_to_box_editor(Operator):
    bl_idname      = "florence2.text_to_box_editor"
    bl_label       = "Text to Box Editor"
    bl_description = "Parse the current text block as Ideogram 4 JSON and send it to the Box Editor"

    @classmethod
    def poll(cls, context):
        return (
            getattr(context, "area", None) is not None
            and context.area.type == "TEXT_EDITOR"
            and getattr(context.space_data, "text", None) is not None
        )

    def execute(self, context):
        text_block = context.space_data.text
        json_str = text_block.as_string()
        if not json_str.strip():
            self.report({"WARNING"}, "Text block is empty.")
            return {"CANCELLED"}
        apply_florence_json_to_mask(json_str, "")
        return {"FINISHED"}


def _text_menu_draw(self, context):
    self.layout.operator("florence2.text_to_box_editor")



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


_active_mask_name = None   # cached during rich depsgraph ticks; used by msgbus callbacks


def _do_select_all_on_active(mask) -> int:
    """Deselect every point except those on mask.layers.active. Returns count selected."""
    active = mask.layers.active
    if not active:
        print("[F2SelectAll] _do_select_all_on_active: no active layer")
        return 0
    total_sel = 0
    for layer in mask.layers:
        for spline in layer.splines:
            for point in spline.points:
                sel = (layer == active)
                point.select_control_point = sel
                point.select_left_handle   = sel
                point.select_right_handle  = sel
                if sel:
                    total_sel += 1
    print(f"[F2SelectAll] selected {total_sel} point(s) on {active.name!r}")
    return total_sel


def _get_or_cache_active_mask():
    """Resolve the active mask using several fallback strategies.

    The msgbus callback runs in a restricted context where edit_mask is None.
    This helper tries four escalating methods so the callback still works:
      1. bpy.context.edit_mask  (works in depsgraph / operator contexts)
      2. _active_mask_name cache (populated every depsgraph tick)
      3. Screen area scan       (walks every IMAGE_EDITOR in MASK mode)
      4. Single-mask fallback   (if only one mask exists in the file)
    """
    global _active_mask_name

    # 1. Standard context path
    try:
        if hasattr(bpy.context, "edit_mask") and bpy.context.edit_mask is not None:
            _active_mask_name = bpy.context.edit_mask.name
            print(f"[F2Mask] resolved via edit_mask: {_active_mask_name!r}")
            return bpy.context.edit_mask
    except Exception:
        pass

    # 2. Name cache updated on every depsgraph tick
    if _active_mask_name and _active_mask_name in bpy.data.masks:
        print(f"[F2Mask] resolved via name cache: {_active_mask_name!r}")
        return bpy.data.masks[_active_mask_name]

    # 3. Walk all IMAGE_EDITOR areas in MASK mode
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    space = area.spaces[0]
                    if getattr(space, "mode", "") == "MASK":
                        m = getattr(space, "mask", None)
                        if m:
                            _active_mask_name = m.name
                            print(f"[F2Mask] resolved via area scan: {_active_mask_name!r}")
                            return m
    except Exception:
        pass

    # 4. Single-mask file shortcut
    if len(bpy.data.masks) == 1:
        _active_mask_name = bpy.data.masks[0].name
        print(f"[F2Mask] resolved via single-mask fallback: {_active_mask_name!r}")
        return bpy.data.masks[0]

    print("[F2Mask] could not resolve active mask")
    return None


def _select_all_points_on_active_layer():
    """msgbus callback: active_layer_index changed in UIList."""
    try:
        ctx_type = type(bpy.context).__name__
        print(f"[F2SelectAll] called — context type={ctx_type!r}")

        mask = _get_or_cache_active_mask()
        print(f"[F2SelectAll] resolved mask={getattr(mask,'name',None)!r}")
        if not mask:
            print("[F2SelectAll] EARLY EXIT — no mask resolved by any method")
            return

        active = mask.layers.active
        idx    = mask.active_layer_index
        print(f"[F2SelectAll] layers.active={getattr(active,'name',None)!r}  active_layer_index={idx}")
        if not active:
            print("[F2SelectAll] EARLY EXIT — no active layer")
            return

        _do_select_all_on_active(mask)
        _tag_image_editors_redraw()
    except Exception as e:
        import traceback
        print(f"[F2SelectAll] ERROR: {e}")
        traceback.print_exc()


_last_selection_state   = None
_last_active_layer_name = None   # tracks UIList / active-layer changes


@bpy.app.handlers.persistent
def _depsgraph_handler(scene, depsgraph=None):
    """Detect which layer the user clicked by scanning point selection state."""
    global _last_selection_state, _last_active_layer_name, _active_mask_name

    # Resolve mask — also keeps _active_mask_name warm for msgbus callbacks
    mask = _get_or_cache_active_mask()
    if not mask:
        return

    print(f"[F2Handler] TICK  mask={mask.name!r}  layers={len(mask.layers)}")

    # ── UIList-click detection: active_layer pointer changed ──────────────────
    active_layer    = mask.layers.active
    cur_active_name = getattr(active_layer, "name", None)
    if cur_active_name != _last_active_layer_name:
        print(f"[F2Handler] active_layer changed: {_last_active_layer_name!r} -> {cur_active_name!r}")
        _last_active_layer_name = cur_active_name
        if active_layer:
            _do_select_all_on_active(mask)
        # Sync index in case Blender's active pointer is ahead of active_layer_index
        for i, layer in enumerate(mask.layers):
            if layer.name == cur_active_name and mask.active_layer_index != i:
                print(f"[F2Handler] syncing active_layer_index -> {i}")
                mask.active_layer_index = i
                break
        _tag_image_editors_redraw()
        return   # let selection settle on the next tick

    current_selection = []

    # Method A: active_point on active layer
    active_splines  = active_layer.splines if active_layer else None
    active_point_ma = getattr(active_splines, "active_point", None)
    print(f"[F2Handler] MethodA: active_layer={cur_active_name!r}  active_point={active_point_ma!r}")
    if active_layer:
        active_spline = active_layer.splines.active
        active_point  = active_layer.splines.active_point
        if active_point and active_spline:
            try:
                sp_idx = list(active_layer.splines).index(active_spline)
                pt_idx = list(active_spline.points).index(active_point)
                current_selection.append({
                    "layer_name": active_layer.name,
                    "spline_idx": sp_idx,
                    "point_idx":  pt_idx,
                    "is_active":  True,
                })
            except ValueError:
                pass

    # Method B: scan select flags across ALL layers
    # (detects clicks on non-active layers — Blender marks the point selected
    #  but does NOT change mask.layers.active when clicking a foreign layer)
    print(f"[F2Handler] MethodB: scanning {len(mask.layers)} layer(s)")
    for layer in mask.layers:
        if layer.hide:
            print(f"[F2Handler]   layer {layer.name!r} hidden, skipping")
            continue
        for sp_idx, spline in enumerate(layer.splines):
            for pt_idx, point in enumerate(spline.points):
                if (point.select_control_point or
                        point.select_left_handle or
                        point.select_right_handle):
                    if not any(
                        d["layer_name"] == layer.name and
                        d["spline_idx"] == sp_idx and
                        d["point_idx"]  == pt_idx
                        for d in current_selection
                    ):
                        print(f"[F2Handler]   MethodB hit: layer={layer.name!r} sp={sp_idx} pt={pt_idx}")
                        current_selection.append({
                            "layer_name": layer.name,
                            "spline_idx": sp_idx,
                            "point_idx":  pt_idx,
                            "is_active":  False,
                        })

    cur_summary  = [(d['layer_name'], d['is_active']) for d in current_selection]
    last_summary = [(d['layer_name'], d['is_active']) for d in (_last_selection_state or [])]
    print(f"[F2Handler] current={cur_summary}  last={last_summary}")

    if current_selection == _last_selection_state:
        print("[F2Handler] no change — skipping")
        return

    # Snapshot last_keys BEFORE updating state (winner logic reads it)
    last_keys = {
        (d["layer_name"], d["spline_idx"], d["point_idx"])
        for d in (_last_selection_state or [])
    }
    _last_selection_state = current_selection

    if not current_selection:
        return

    # Winner: prefer NEWLY appeared MethodB entries over the stale is_active
    # entry.  "New" means this (layer, spline, point) tuple wasn't in last state.
    # This correctly handles clicking a non-active layer whose active_point
    # pointer lags behind on the old layer.
    new_entries = [
        d for d in current_selection
        if (d["layer_name"], d["spline_idx"], d["point_idx"]) not in last_keys
           and not d["is_active"]   # MethodB entries only — ignore MethodA churn
    ]
    print(f"[F2Handler] new_entries={[(d['layer_name'], d['point_idx']) for d in new_entries]}")

    if new_entries:
        winner = new_entries[0]["layer_name"]
        print(f"[F2Handler] winner from new MethodB entry: {winner!r}")
    else:
        winner = next(
            (d["layer_name"] for d in current_selection if d["is_active"]),
            current_selection[0]["layer_name"],
        )
        print(f"[F2Handler] winner from existing selection: {winner!r}")

    print(f"[F2Handler] winner={winner!r} current idx={mask.active_layer_index}")
    for i, layer in enumerate(mask.layers):
        if layer.name == winner and mask.active_layer_index != i:
            print(f"[F2Handler] writing active_layer_index {mask.active_layer_index} -> {i}")
            mask.active_layer_index = i
            _last_active_layer_name = winner   # suppress the echo on next tick
            break

    _tag_image_editors_redraw()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = [
    Florence2ColorEntry,   # must be before Florence2LayerData / Florence2MaskProps
    Florence2LayerData,    # must be before Florence2MaskProps
    Florence2MaskProps,
    FLORENCE2_OT_palette_add,
    FLORENCE2_OT_palette_remove,
    FLORENCE2_OT_export_strip,
    FLORENCE2_OT_layer_new_with_square,
    FLORENCE2_OT_open_box_editor,
    FLORENCE2_OT_text_to_box_editor,
    FLORENCE2_PT_mask_panel,
]


def register():
    print("[Florence2Mask] Registering...")
    for cls in classes:
        bpy.utils.register_class(cls)
        print(f"[Florence2Mask]   {cls.__name__}")
    bpy.types.Mask.florence2_props = PointerProperty(type=Florence2MaskProps)

    # Ensure the "Send to Box Editor" scene toggle is always registered,
    # even when the Florence-2 text captioning plugin has not been loaded.
    if not hasattr(bpy.types.Scene, "florence2_send_to_mask"):
        bpy.types.Scene.florence2_send_to_mask = bpy.props.BoolProperty(
            name="Send to Box Editor",
            description="After generation, open the result as mask layers in the Box Editor",
            default=False,
        )

    # When the user clicks a layer in the UIList, select all its points.
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.Mask, "active_layer_index"),
        owner=_msgbus_owner,
        args=(),
        notify=_select_all_points_on_active_layer,
    )
    # Redraw on layer select-flag or active-pointer changes.
    for rna_key in (
        (bpy.types.MaskLayer, "select"),
        (bpy.types.MaskLayers, "active"),
    ):
        bpy.msgbus.subscribe_rna(
            key=rna_key,
            owner=_msgbus_owner,
            args=(),
            notify=_tag_image_editors_redraw,
        )

    # depsgraph handler: fires when mask data changes (slide_point, handle drags)
    if _depsgraph_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_depsgraph_handler)

    def _append_text_menu():
        try:
            bpy.types.TEXT_MT_edit.append(_text_menu_draw)
            print("[Florence2Mask] TEXT_MT_edit menu item appended")
        except Exception as exc:
            print(f"[Florence2Mask] TEXT_MT_edit append failed: {exc}")
        return None  # run once

    bpy.app.timers.register(_append_text_menu, first_interval=0.1)

    print("[Florence2Mask] Registration complete")


def unregister():
    if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
    try:
        bpy.types.TEXT_MT_edit.remove(_text_menu_draw)
    except Exception:
        pass
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if hasattr(bpy.types.Mask, "florence2_props"):
        del bpy.types.Mask.florence2_props
    if hasattr(bpy.types.Scene, "florence2_send_to_mask"):
        del bpy.types.Scene.florence2_send_to_mask
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    print("[Florence2Mask] Unregistered")
