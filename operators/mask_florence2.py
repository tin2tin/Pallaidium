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


# Set True to re-enable [Florence2Mask] debug logging.
_DEBUG = False


def _dbg(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)


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

def _mask_fit(W: int, H: int):
    """Return (sx, sy, ox, oy) mapping the backdrop image into Blender's mask space.

    Blender's image-editor mask coordinate space is isotropic (a square): one
    unit on X equals one unit on Y on screen.  The backdrop image is fit into
    this square by its LONGER edge (→ that axis spans [0,1]) while the shorter
    edge is scaled by the aspect ratio and centered.  So for a landscape image
    (W>=H): X spans [0,1], Y spans [(1-H/W)/2, (1+H/W)/2].
    """
    if not W or not H:
        return 1.0, 1.0, 0.0, 0.0
    if W >= H:
        sx, sy = 1.0, H / W
    else:
        sx, sy = W / H, 1.0
    return sx, sy, (1.0 - sx) / 2.0, (1.0 - sy) / 2.0


def _bbox_to_mask(bbox, W: int, H: int):
    """Ideogram [y1,x1,y2,x2] 0-1000 → mask center + half-sizes.

    Ideogram x runs left→right, y runs top→bottom (y=0 is the image TOP).
    Blender mask Y has a lower-left origin, so Y is flipped.  Coordinates are
    fit to the backdrop via _mask_fit() so boxes overlay the image exactly.
    """
    y1, x1, y2, x2 = bbox
    sx, sy, ox, oy = _mask_fit(W, H)
    cx = ox + (x1 + x2) / 2 / 1000 * sx
    cy = oy + (1.0 - (y1 + y2) / 2 / 1000) * sy
    hx = (x2 - x1) / 1000 / 2 * sx
    hy = (y2 - y1) / 1000 / 2 * sy
    return cx, cy, hx, hy


def _spline_to_bbox(spline, W: int, H: int):
    """Live spline corners → Ideogram [y1,x1,y2,x2] 0-1000 (inverse of _bbox_to_mask)."""
    sx, sy, ox, oy = _mask_fit(W, H)
    pts = [p.co for p in spline.points]
    if not pts:
        return [0, 0, 1000, 1000]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1 = round((min(xs) - ox) / sx * 1000)
    x2 = round((max(xs) - ox) / sx * 1000)
    y1 = round((1.0 - (max(ys) - oy) / sy) * 1000)   # top    (larger mask Y)
    y2 = round((1.0 - (min(ys) - oy) / sy) * 1000)   # bottom (smaller mask Y)
    clamp = lambda v: max(0, min(1000, v))
    return [clamp(y1), clamp(x1), clamp(y2), clamp(x2)]


# ---------------------------------------------------------------------------
# Commented-JSON helpers  (// lines are treated as comments)
# ---------------------------------------------------------------------------

def _parse_commented_json(text: str) -> "tuple[str, dict]":
    """Strip // comment lines and return (clean_json_str, metadata_dict).

    Lines whose first non-whitespace characters are ``//`` are treated as
    comments and excluded from the JSON payload.  Key-value comments in the
    form ``// key: value`` are collected into the returned metadata dict
    (keys are lower-cased and stripped).
    """
    meta: dict = {}
    json_lines: list = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            body = stripped[2:].strip()
            if ":" in body:
                k, _, v = body.partition(":")
                meta[k.strip().lower()] = v.strip()
        else:
            json_lines.append(line)
    return "\n".join(json_lines), meta


# ---------------------------------------------------------------------------
# Core: populate a mask from a parsed JSON dict
# ---------------------------------------------------------------------------

def _populate_mask_from_data(mask, data: dict, W: int, H: int) -> None:
    """Write all props and layers from an Ideogram-4 data dict into *mask*."""
    elements = data.get("compositional_deconstruction", {}).get("elements", [])

    mp = mask.florence2_props
    mp.f2_high_level_description = data.get("high_level_description", "")
    mp.f2_background = data.get("compositional_deconstruction", {}).get("background", "")
    style = data.get("style_description", {})
    mp.f2_style_json = json.dumps(style, ensure_ascii=False)
    mp.f2_is_photo   = "photo" in style
    mp.f2_aesthetics = style.get("aesthetics", "")
    mp.f2_lighting   = style.get("lighting", "")
    mp.f2_photo      = style.get("photo", "")
    mp.f2_medium     = style.get("medium", "")
    mp.f2_art_style  = style.get("art_style", "")
    _list_to_palette(mp.f2_style_palette, style.get("color_palette") or [])

    for layer in list(mask.layers):
        mask.layers.remove(layer)
    _clear_layer_data(mask)

    for elem in elements[:40]:
        try:
            elem_type = elem.get("type", "obj")
            desc      = (elem.get("desc") or "element").strip()
            text_val  = (elem.get("text") or "").strip()
            bbox      = elem.get("bbox") or [0, 0, 1000, 1000]

            layer = mask.layers.new(name=_layer_name(elem_type, desc, text_val))
            actual_name = layer.name
            try:
                layer.fill_color = _layer_fill_color(elem_type, desc)
            except Exception:
                pass
            _add_layer_data(mask, actual_name, {
                "type":  elem_type, "desc": desc, "text": text_val,
                "color": elem.get("color", ""), "font": elem.get("font", ""), "bbox": bbox,
            })
            cx, cy, hx, hy = _bbox_to_mask(bbox, W, H)
            hx = max(hx, 0.001)
            hy = max(hy, 0.001)
            spline = layer.splines.new()
            spline.use_cyclic = True
            spline.points.add(3)
            corners = [
                (cx - hx, cy - hy), (cx + hx, cy - hy),
                (cx + hx, cy + hy), (cx - hx, cy + hy),
            ]
            for i, (px, py) in enumerate(corners):
                pt = spline.points[i]
                pt.co = pt.handle_left = pt.handle_right = (px, py)
                pt.select_control_point = pt.select_left_handle = pt.select_right_handle = False
                try:
                    pt.handle_type = "VECTOR"
                except AttributeError:
                    pass
            _dbg(f"[Florence2Mask]   {actual_name!r}  {elem_type}  {bbox}")
        except Exception as _elem_exc:
            _dbg(f"[Florence2Mask]   skipped element {elem.get('desc', '?')!r}: {_elem_exc}")

    if mask.layers:
        mask.active_layer_index = 0


# ---------------------------------------------------------------------------
# Core: build mask from JSON  (generation callback — always creates new)
# ---------------------------------------------------------------------------

def apply_florence_json_to_mask(json_str: str, source_image_path: str) -> None:
    """Parse Ideogram4 JSON, populate a *new* Mask, open in Image Editor. Main thread."""
    _dbg(f"[Florence2Mask] apply called, source={source_image_path!r}")

    clean_json, meta = _parse_commented_json(json_str)
    # Explicit arg wins; fall back to // image: comment embedded in the JSON.
    if not source_image_path:
        source_image_path = meta.get("image", "")

    try:
        data = json.loads(clean_json)
    except Exception as exc:
        msg = f"JSON parse error: {exc}"
        _dbg(f"[Florence2Mask] {msg}")
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
    _dbg(f"[Florence2Mask] {len(elements)} element(s)")

    # Always load a fresh image datablock (never reuse an existing one)
    img = None
    if source_image_path and os.path.isfile(source_image_path):
        try:
            img = bpy.data.images.load(source_image_path)
            _dbg(f"[Florence2Mask] Loaded image: {img.name!r}  size={img.size[:]}")
        except Exception as exc:
            _dbg(f"[Florence2Mask] Could not load image: {exc}")
    if img is None:
        img = bpy.data.images.new("Box Editor", width=1920, height=1080)
        _dbg(f"[Florence2Mask] Created blank image: {img.name!r}")

    W = img.size[0] if img else 1920
    H = img.size[1] if img else 1080

    # Always create a fresh mask; name it from the scene description.
    mask_name = (data.get("high_level_description") or "Florence2")[:63]
    mask = bpy.data.masks.new(name=mask_name)
    # Give the image the same name so the panel can find it by mask.name lookup.
    img.name = mask.name
    _dbg(f"[Florence2Mask] Mask/image: {mask.name!r}")

    _populate_mask_from_data(mask, data, W, H)
    _open_mask_in_editor(mask, img)
    _dbg(f"[Florence2Mask] Done — {len(elements)} layer(s) in {mask.name!r}")


# ---------------------------------------------------------------------------
# Area management
# ---------------------------------------------------------------------------

def _open_mask_in_editor(mask, img) -> None:
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        areas  = screen.areas

        image_ed = next((a for a in areas if a.type == "IMAGE_EDITOR"), None)

        if image_ed is None:
            # Prefer a dedicated PREVIEW area; fall back to any SEQUENCE_EDITOR.
            target = (
                next(
                    (a for a in areas
                     if a.type == "SEQUENCE_EDITOR"
                     and getattr(a.spaces[0], "view_type", "") == "PREVIEW"),
                    None,
                )
                or next(
                    (a for a in areas
                     if a.type == "SEQUENCE_EDITOR"
                     and getattr(a.spaces[0], "view_type", "") == "SEQUENCER_PREVIEW"),
                    None,
                )
                or next((a for a in areas if a.type == "SEQUENCE_EDITOR"), None)
                or next((a for a in areas if a.type == "TEXT_EDITOR"), None)
            )

            if target is None:
                _dbg("[Florence2Mask] No suitable area to split — open Image Editor manually.")
                return

            # If target is the combined Sequencer+Preview, switch it to pure PREVIEW
            # first so area_split doesn't unpack it into two separate panels.
            if getattr(target.spaces[0], "view_type", "") == "SEQUENCER_PREVIEW":
                try:
                    target.spaces[0].view_type = "PREVIEW"
                except Exception:
                    pass

            # Snapshot area coords before split; coord-based detection is more
            # reliable than Python id() which can change after Blender reallocates.
            coords_before = {(a.x, a.y, a.width, a.height) for a in screen.areas}
            try:
                region = next(r for r in target.regions if r.type == "WINDOW")
                with bpy.context.temp_override(window=window, area=target, region=region):
                    bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
            except Exception as exc:
                _dbg(f"[Florence2Mask] Area split failed: {exc}")
                return

            new_areas = [
                a for a in screen.areas
                if (a.x, a.y, a.width, a.height) not in coords_before
            ]
            if not new_areas:
                _dbg("[Florence2Mask] Could not locate new area after split.")
                return
            # Take the rightmost new area (the split-off right half).
            image_ed = max(new_areas, key=lambda a: a.x)
            image_ed.type = "IMAGE_EDITOR"

        space = image_ed.spaces[0]
        try:
            space.mode = "MASK"
        except Exception as exc:
            _dbg(f"[Florence2Mask] Could not set MASK mode: {exc}")
        if img is not None:
            try:
                space.image = img
                _dbg(f"[Florence2Mask] Background image: {img.name!r}")
            except Exception:
                pass
        try:
            space.mask = mask
            _dbg(f"[Florence2Mask] Mask set: {mask.name!r}")
        except Exception as exc:
            _dbg(f"[Florence2Mask] Could not set mask: {exc}")

        try:
            region = next((r for r in image_ed.regions if r.type == "WINDOW"), None)
            if region:
                with bpy.context.temp_override(window=window, area=image_ed, region=region):
                    bpy.ops.image.view_all(fit_view=True)
                _dbg("[Florence2Mask] View fitted")
        except Exception as exc:
            _dbg(f"[Florence2Mask] view_all failed: {exc}")

        # Defer sidebar/tab switch — area must be fully initialised first
        def _open_sidebar():
            try:
                for win in bpy.context.window_manager.windows:
                    for area in win.screen.areas:
                        if area.type != "IMAGE_EDITOR":
                            continue
                        sp = area.spaces[0]
                        if getattr(sp, "mode", "") != "MASK":
                            continue
                        ui_region = next(
                            (r for r in area.regions if r.type == "UI"), None
                        )
                        sp.show_region_ui = True
                        # Switch the N-panel to the Box Editor tab.  Only attempt
                        # it when the space actually exposes active_panel_category
                        # — SpaceImageEditor does not, and calling the operator
                        # with an invalid data_path spams a non-catchable
                        # "context_path_validate error" to the console.
                        if ui_region and hasattr(sp, "active_panel_category"):
                            with bpy.context.temp_override(
                                window=win, area=area, region=ui_region
                            ):
                                bpy.ops.wm.context_set_string(
                                    data_path="space_data.active_panel_category",
                                    value="Box Editor",
                                )
                        _dbg("[Florence2Mask] Box Editor sidebar opened")
            except Exception as exc:
                _dbg(f"[Florence2Mask] Tab switch failed: {exc}")

        bpy.app.timers.register(_open_sidebar, first_interval=0.1)
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

        row = layout.row(align=True)
        row.operator("florence2.new_box_editor", text="New",  icon="FILE_NEW")
        row.operator("florence2.load_box_json",  text="Load", icon="FILEBROWSER")
        row.operator("florence2.save_box_json",  text="Save", icon="FILE_TICK")

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
        box.label(text="Boxes", icon="MOD_MASK")

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
            box.prop(ld, "f2_type", text="")
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
# Shared serialization helper
# ---------------------------------------------------------------------------

def _mask_to_json_dict(mask, W: int = 1920, H: int = 1080) -> dict:
    """Serialize a Box Editor mask to an Ideogram-4 JSON-compatible dict."""
    mp = mask.florence2_props
    elements = []
    for layer in mask.layers:
        ld = _get_layer_data(layer, mask)
        if not ld:
            continue
        bbox = (_spline_to_bbox(layer.splines[0], W, H) if layer.splines
                else json.loads(ld.f2_bbox or "[0,0,1000,1000]"))
        cp = _palette_to_list(ld.f2_palette)
        if not cp and ld.f2_type == "text":
            hex_c = _color_to_hex(ld.f2_color)
            if hex_c not in ("#FFFFFF", "#FEFEFE"):
                cp = [hex_c]
        if ld.f2_type == "text":
            elem = {"type": "text", "bbox": bbox, "text": ld.f2_text, "desc": ld.f2_desc}
            if cp:
                elem["color_palette"] = cp
        else:
            elem = {"type": "obj", "bbox": bbox, "desc": ld.f2_desc}
            if cp:
                elem["color_palette"] = cp
        elements.append(elem)

    if mp.f2_aesthetics or mp.f2_lighting or mp.f2_medium:
        style: dict = {}
        if mp.f2_aesthetics: style["aesthetics"] = mp.f2_aesthetics
        if mp.f2_lighting:   style["lighting"]   = mp.f2_lighting
        if mp.f2_is_photo:
            if mp.f2_photo:  style["photo"]  = mp.f2_photo
            if mp.f2_medium: style["medium"] = mp.f2_medium
        else:
            if mp.f2_medium:    style["medium"]    = mp.f2_medium
            if mp.f2_art_style: style["art_style"] = mp.f2_art_style
        sp = _palette_to_list(mp.f2_style_palette)
        if sp: style["color_palette"] = sp
    else:
        try:
            style = json.loads(mp.f2_style_json) if mp.f2_style_json else {}
        except Exception:
            style = {}

    return {
        "high_level_description": mp.f2_high_level_description,
        "style_description": style,
        "compositional_deconstruction": {
            "background": mp.f2_background,
            "elements":   elements,
        },
    }


# ---------------------------------------------------------------------------
# New / Load / Save operators
# ---------------------------------------------------------------------------

class FLORENCE2_OT_new_box_editor(Operator):
    bl_idname      = "florence2.new_box_editor"
    bl_label       = "New Box Editor"
    bl_description = "Create a new empty Box Editor mask"
    bl_options     = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space and space.type == "IMAGE_EDITOR"

    def execute(self, _context):
        bpy.ops.mask.new()
        return {"FINISHED"}


class FLORENCE2_OT_load_box_json(Operator):
    bl_idname      = "florence2.load_box_json"
    bl_label       = "Load JSON"
    bl_description = "Load an Ideogram 4 JSON file into the Box Editor"
    bl_options     = {"REGISTER", "UNDO"}

    filepath:    StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space and space.type == "IMAGE_EDITOR"

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from pathlib import Path
        try:
            file_text = Path(self.filepath).read_text(encoding="utf-8")
            clean, meta = _parse_commented_json(file_text)
            data = json.loads(clean)
        except Exception as exc:
            self.report({"ERROR"}, f"Could not read/parse file: {exc}")
            return {"CANCELLED"}

        # Resolve background image from embedded // image: comment
        img = None
        img_path = meta.get("image", "")
        if img_path:
            img_abs = bpy.path.abspath(img_path)
            if os.path.isfile(img_abs):
                try:
                    img = bpy.data.images.load(img_abs)
                except Exception as exc:
                    self.report({"WARNING"}, f"Could not load image: {exc}")
            else:
                self.report({"WARNING"}, f"Image not found: {img_path}")

        bpy.ops.mask.new()
        if img is None:
            bpy.ops.image.new(name="Box Editor", width=1920, height=1080)

        space = context.space_data
        mask  = space.mask
        if img is not None:
            space.image = img
        else:
            img = space.image
        W = img.size[0] if img else 1920
        H = img.size[1] if img else 1080
        _populate_mask_from_data(mask, data, W, H)
        scene_name = (data.get("high_level_description") or "")[:63]
        if scene_name:
            mask.name = scene_name
            if img:
                img.name = mask.name
        _open_mask_in_editor(mask, img)
        return {"FINISHED"}


class FLORENCE2_OT_save_box_json(Operator):
    bl_idname      = "florence2.save_box_json"
    bl_label       = "Save JSON"
    bl_description = "Save the Box Editor layout to an Ideogram 4 JSON file"
    bl_options     = {"REGISTER"}

    filepath:     StringProperty(subtype="FILE_PATH")
    filename_ext  = ".json"
    filter_glob:  StringProperty(default="*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        space = getattr(context, "space_data", None)
        return space and space.type == "IMAGE_EDITOR" and getattr(space, "mask", None) is not None

    def invoke(self, context, event):
        mask = context.space_data.mask
        self.filepath = (mask.name or "box_editor") + ".json"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from pathlib import Path
        space = context.space_data
        mask  = space.mask
        img   = getattr(space, "image", None)
        W     = img.size[0] if img else 1920
        H     = img.size[1] if img else 1080
        data      = _mask_to_json_dict(mask, W, H)
        json_body = json.dumps(data, indent=2, ensure_ascii=False)
        if img and getattr(img, "filepath", ""):
            img_abs = bpy.path.abspath(img.filepath)
            if os.path.isfile(img_abs):
                json_body = f"// image: {img_abs}\n" + json_body
        try:
            Path(self.filepath).write_text(json_body, encoding="utf-8")
        except Exception as exc:
            self.report({"ERROR"}, f"Could not write file: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved: {self.filepath}")
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

        result   = _mask_to_json_dict(mask, W, H)
        json_str = json.dumps(result, indent=2, ensure_ascii=False)

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
    bl_description = "Add a new Box Editor layer with a square spline at the 2D cursor"
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

        # Anchor the square at the 2D cursor (image UV space, 0-1, lower-left
        # origin); fall back to the image centre when no cursor is available.
        cursor = getattr(space, "cursor_location", None)
        cu = float(cursor[0]) if cursor is not None and len(cursor) >= 2 else 0.5
        cv = float(cursor[1]) if cursor is not None and len(cursor) >= 2 else 0.5
        cu = min(max(cu, 0.0), 1.0)
        cv = min(max(cv, 0.0), 1.0)
        # Ideogram bbox [y1,x1,y2,x2] (0-1000, y from top) for a 200-unit square.
        idx = cu * 1000.0
        idy = (1.0 - cv) * 1000.0
        _clamp = lambda v: max(0, min(1000, int(round(v))))
        anchor_bbox = [_clamp(idy - 100), _clamp(idx - 100),
                       _clamp(idy + 100), _clamp(idx + 100)]

        layer = mask.layers.new(name="New Layer")

        # Register Box Editor metadata BEFORE activating the layer, so the
        # depsgraph handler never sees it without f2_layers data.
        _add_layer_data(mask, layer.name, {
            "type": "obj", "desc": "", "text": "",
            "color": "", "font": "", "bbox": anchor_bbox,
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

        # Square at the anchor bbox, mapped to mask space.
        cx, cy, hx, hy = _bbox_to_mask(anchor_bbox, W, H)
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
_last_mask_per_editor: dict = {}   # id(area) → last mask name; used by poll timer


def _find_image_for_mask(mask):
    """Find the background image for *mask* by trying several name forms."""
    name = mask.name
    # 1. exact datablock name (spaces, as set by img.name = mask.name)
    img = bpy.data.images.get(name)
    if img:
        return img
    # 2. underscores variant (clean_filename output)
    name_under = name.replace(" ", "_")
    img = bpy.data.images.get(name_under)
    if img:
        return img
    # 3. with .png extension (Blender uses filename as initial datablock name)
    img = bpy.data.images.get(name + ".png")
    if img:
        return img
    img = bpy.data.images.get(name_under + ".png")
    if img:
        return img
    # 4. match by filepath stem (most permissive — handles truncated names)
    name_stem_lc = name_under.lower()
    for img in bpy.data.images:
        fp = img.filepath
        if fp:
            stem = os.path.splitext(os.path.basename(bpy.path.abspath(fp)))[0].lower()
            if stem == name_stem_lc:
                return img
    return None


def _auto_switch_image_for_mask():
    """When the active mask changes, switch to the matching image (if it exists)."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "IMAGE_EDITOR":
                    continue
                space = area.spaces[0]
                mask = getattr(space, "mask", None)
                if not mask:
                    continue
                img = _find_image_for_mask(mask)
                if img and space.image is not img:
                    space.image = img
    except Exception:
        pass


def _poll_mask_change():
    """Timer: poll every 0.25 s for mask changes and auto-switch the background image."""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type != "IMAGE_EDITOR":
                    continue
                space = area.spaces[0]
                key = id(area)
                mask = getattr(space, "mask", None)
                mask_name = mask.name if mask else None
                if _last_mask_per_editor.get(key) != mask_name:
                    _last_mask_per_editor[key] = mask_name
                    if mask:
                        img = _find_image_for_mask(mask)
                        if img and space.image is not img:
                            space.image = img
    except Exception:
        pass
    return 0.25


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
            _f2dbg(f"[F2Mask] resolved via edit_mask: {_active_mask_name!r}")
            return bpy.context.edit_mask
    except Exception:
        pass

    # 2. Name cache updated on every depsgraph tick
    if _active_mask_name and _active_mask_name in bpy.data.masks:
        _f2dbg(f"[F2Mask] resolved via name cache: {_active_mask_name!r}")
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
                            _f2dbg(f"[F2Mask] resolved via area scan: {_active_mask_name!r}")
                            return m
    except Exception:
        pass

    # 4. Single-mask file shortcut
    if len(bpy.data.masks) == 1:
        _active_mask_name = bpy.data.masks[0].name
        _f2dbg(f"[F2Mask] resolved via single-mask fallback: {_active_mask_name!r}")
        return bpy.data.masks[0]

    return None


_last_selection_state   = None
_last_active_layer_name = None   # tracks UIList / active-layer changes
_suppress_select_all    = False  # True when active_layer_index was written by MethodB

# Set to True to print the verbose [F2Handler]/[F2Mask] layer-tracking trace.
_F2_DEBUG = False


def _f2dbg(*args, **kwargs):
    if _F2_DEBUG:
        print(*args, **kwargs)


@bpy.app.handlers.persistent
def _depsgraph_handler(scene, depsgraph=None):
    """Detect which layer the user clicked by scanning point selection state."""
    global _last_selection_state, _last_active_layer_name, _active_mask_name, _suppress_select_all

    # Resolve mask — also keeps _active_mask_name warm for msgbus callbacks
    mask = _get_or_cache_active_mask()
    if not mask:
        return

    _f2dbg(f"[F2Handler] TICK  mask={mask.name!r}  layers={len(mask.layers)}")

    # ── UIList-click detection: active_layer pointer changed ──────────────────
    active_layer    = mask.layers.active
    cur_active_name = getattr(active_layer, "name", None)
    if cur_active_name != _last_active_layer_name:
        _f2dbg(f"[F2Handler] active_layer changed: {_last_active_layer_name!r} -> {cur_active_name!r}")
        _last_active_layer_name = cur_active_name
        if active_layer and not _suppress_select_all:
            # Only auto-select-all when the change came from a UIList click,
            # not when _depsgraph_handler itself wrote active_layer_index (MethodB).
            _do_select_all_on_active(mask)
        _suppress_select_all = False  # consume the flag regardless
        # Sync index in case Blender's active pointer is ahead of active_layer_index
        for i, layer in enumerate(mask.layers):
            if layer.name == cur_active_name and mask.active_layer_index != i:
                _f2dbg(f"[F2Handler] syncing active_layer_index -> {i}")
                mask.active_layer_index = i
                break
        _tag_image_editors_redraw()
        return   # let selection settle on the next tick

    current_selection = []

    # Method A: active_point on active layer
    active_splines  = active_layer.splines if active_layer else None
    active_point_ma = getattr(active_splines, "active_point", None)
    _f2dbg(f"[F2Handler] MethodA: active_layer={cur_active_name!r}  active_point={active_point_ma!r}")
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
    _f2dbg(f"[F2Handler] MethodB: scanning {len(mask.layers)} layer(s)")
    for layer in mask.layers:
        if layer.hide:
            _f2dbg(f"[F2Handler]   layer {layer.name!r} hidden, skipping")
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
                        _f2dbg(f"[F2Handler]   MethodB hit: layer={layer.name!r} sp={sp_idx} pt={pt_idx}")
                        current_selection.append({
                            "layer_name": layer.name,
                            "spline_idx": sp_idx,
                            "point_idx":  pt_idx,
                            "is_active":  False,
                        })

    cur_summary  = [(d['layer_name'], d['is_active']) for d in current_selection]
    last_summary = [(d['layer_name'], d['is_active']) for d in (_last_selection_state or [])]
    _f2dbg(f"[F2Handler] current={cur_summary}  last={last_summary}")

    if current_selection == _last_selection_state:
        _f2dbg("[F2Handler] no change — skipping")
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
    _f2dbg(f"[F2Handler] new_entries={[(d['layer_name'], d['point_idx']) for d in new_entries]}")

    if new_entries:
        winner = new_entries[0]["layer_name"]
        _f2dbg(f"[F2Handler] winner from new MethodB entry: {winner!r}")
    else:
        winner = next(
            (d["layer_name"] for d in current_selection if d["is_active"]),
            current_selection[0]["layer_name"],
        )
        _f2dbg(f"[F2Handler] winner from existing selection: {winner!r}")

    _f2dbg(f"[F2Handler] winner={winner!r} current idx={mask.active_layer_index}")
    for i, layer in enumerate(mask.layers):
        if layer.name == winner and mask.active_layer_index != i:
            _f2dbg(f"[F2Handler] writing active_layer_index {mask.active_layer_index} -> {i}")
            _suppress_select_all = True    # this write is programmatic — don't auto-select-all
            mask.active_layer_index = i
            _last_active_layer_name = winner
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
    FLORENCE2_OT_new_box_editor,
    FLORENCE2_OT_load_box_json,
    FLORENCE2_OT_save_box_json,
    FLORENCE2_OT_export_strip,
    FLORENCE2_OT_layer_new_with_square,
    FLORENCE2_OT_open_box_editor,
    FLORENCE2_OT_text_to_box_editor,
    FLORENCE2_PT_mask_panel,
]


def register():
    _dbg("[Florence2Mask] Registering...")
    for cls in classes:
        bpy.utils.register_class(cls)
        _dbg(f"[Florence2Mask]   {cls.__name__}")
    bpy.types.Mask.florence2_props = PointerProperty(type=Florence2MaskProps)

    # Ensure Florence-2 scene toggles are always registered so the panel
    # can draw them even before the florence2.py plugin module is imported.
    if not hasattr(bpy.types.Scene, "florence2_mode"):
        bpy.types.Scene.florence2_mode = bpy.props.EnumProperty(
            name="Mode",
            items=[
                ("CAPTION",   "Caption",  "Detailed image caption as plain text"),
                ("IDEOGRAM4", "Box Json", "Extract structured Ideogram 4 prompt JSON"),
            ],
            default="CAPTION",
        )
    if not hasattr(bpy.types.Scene, "florence2_send_to_mask"):
        bpy.types.Scene.florence2_send_to_mask = bpy.props.BoolProperty(
            name="Send to Box Editor",
            description="After generation, open the result as mask layers in the Box Editor",
            default=False,
        )
    # Auto-switch background image when the user picks a different mask.
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.SpaceImageEditor, "mask"),
        owner=_msgbus_owner,
        args=(),
        notify=_auto_switch_image_for_mask,
    )

    # Redraw on layer select-flag or active-pointer changes.
    # NOTE: active_layer_index is NOT subscribed here — writing it from the
    # depsgraph handler's MethodB (canvas click) path would re-fire and call
    # _do_select_all_on_active, clobbering the user's single-point selection.
    # UIList-click detection is handled entirely inside _depsgraph_handler.
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
            _dbg("[Florence2Mask] TEXT_MT_edit menu item appended")
        except Exception as exc:
            _dbg(f"[Florence2Mask] TEXT_MT_edit append failed: {exc}")
        return None  # run once

    bpy.app.timers.register(_append_text_menu, first_interval=0.1)

    # Poll timer: reliable fallback for mask→image auto-switch when msgbus misses events.
    if not bpy.app.timers.is_registered(_poll_mask_change):
        bpy.app.timers.register(_poll_mask_change, first_interval=1.0, persistent=True)

    _dbg("[Florence2Mask] Registration complete")


def unregister():
    if bpy.app.timers.is_registered(_poll_mask_change):
        bpy.app.timers.unregister(_poll_mask_change)
    if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
    try:
        bpy.types.TEXT_MT_edit.remove(_text_menu_draw)
    except Exception:
        pass
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if hasattr(bpy.types.Mask, "florence2_props"):
        del bpy.types.Mask.florence2_props
    if hasattr(bpy.types.Scene, "florence2_mode"):
        del bpy.types.Scene.florence2_mode
    if hasattr(bpy.types.Scene, "florence2_send_to_mask"):
        del bpy.types.Scene.florence2_send_to_mask
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _dbg("[Florence2Mask] Unregistered")
