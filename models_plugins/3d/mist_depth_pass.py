"""
Mist / Depth Pass — duplicate a Scene strip's scene and rig an EEVEE Mist
compositor pass on the copy, then insert the result as a new Scene strip
one channel above the original.

Not an AI model: generate() manipulates the VSE/scene data directly
(requires_main_thread=True), the same pattern used by
models_plugins/text/faster_whisper_transcribe.py.

Workflow:
  1. Select one or more SCENE strips in the VSE.
  2. Switch to the 3D model panel, pick "Mist / Depth Pass".
  3. Choose a Range mode (Auto / Custom / Scene World) and Invert toggle.
  4. Click Add to Queue — one job is created per selected Scene strip; each
     job duplicates that strip's scene (reusing an existing "<name>_Mist"
     duplicate if one already exists), wires up the Mist compositor graph,
     and drops a new Scene strip one channel above the source.
"""

import bpy
from mathutils import Vector

from ...models.base import ModelPlugin, InputSpec, ParamSpec, ModelInputs

MIST_SCENE_SUFFIX = "_Mist"


# ---------------------------------------------------------------------------
# Helpers (ported from the reference Blender Text-Editor script)
# ---------------------------------------------------------------------------

def _find_free_channel(seq_editor, start_frame: int, end_frame: int,
                        start_ch: int = 1) -> int:
    """Return the lowest channel number entirely free in [start_frame, end_frame)."""
    all_strips = list(seq_editor.strips_all)
    ch = max(1, start_ch)
    while True:
        for seq in all_strips:
            if (
                seq.channel == ch
                and seq.frame_final_start < end_frame
                and (seq.frame_final_start + seq.frame_final_duration) > start_frame
            ):
                break
        else:
            return ch
        ch += 1


def _duplicate_scene(source):
    new_scene = source.copy()
    new_scene.name = source.name + MIST_SCENE_SUFFIX
    return new_scene


def _find_existing_mist_scene(source):
    return bpy.data.scenes.get(source.name + MIST_SCENE_SUFFIX)


def _set_eevee(scene):
    # 5.x: 'BLENDER_EEVEE', 4.2-4.x: 'BLENDER_EEVEE_NEXT'
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            return
        except TypeError:
            continue


def _new_map_range_node(tree):
    try:
        node = tree.nodes.new("CompositorNodeMapRange")
        node.use_clamp = True
    except RuntimeError:
        node = tree.nodes.new("ShaderNodeMapRange")
        node.clamp = True
    return node


def _compute_auto_mist_range(scene):
    """Find the nearest/farthest geometry seen by the camera and return (start, depth)."""
    cam = scene.camera
    if cam is None:
        return None

    cam_pos  = cam.matrix_world.translation
    view_dir = -(cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0)))

    dists = []
    for obj in scene.objects:
        if obj.type not in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            d = (world_corner - cam_pos).dot(view_dir)
            if d > 0.0:
                dists.append(d)

    if not dists:
        return None

    near = min(dists)
    far  = max(dists)
    if far - near < 0.001:
        far = near + 1.0

    span = far - near
    return max(near - span * 0.05, 0.0), span * 1.1


def _setup_mist(scene, range_mode: str, custom_start: float, custom_depth: float, invert: bool):
    """Render Layers (Mist) -> Map Range (invert, clamp) -> Output."""
    _set_eevee(scene)

    for vl in scene.view_layers:
        vl.use_pass_mist = True

    if scene.world is None:
        scene.world = bpy.data.worlds.new(scene.name + "_World")

    if range_mode in {"AUTO", "CUSTOM"}:
        # Copy the world so the original scene's mist settings stay untouched.
        scene.world = scene.world.copy()
        mist = scene.world.mist_settings
        mist.falloff = "LINEAR"  # linear = true depth; the default (quadratic) clumps values

        if range_mode == "AUTO":
            auto = _compute_auto_mist_range(scene)
            if auto is not None:
                mist.start, mist.depth = auto
            # else: no camera/geometry — keep the scene's existing mist range.
        else:
            mist.start = custom_start
            mist.depth = custom_depth

    scene.render.use_compositing = True

    # Mist is data, not colour — AgX/Filmic tone-mapping would distort it.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look           = "None"
    scene.view_settings.exposure       = 0.0
    scene.view_settings.gamma          = 1.0

    # scene.copy() shares the compositor node group with the source scene —
    # always create a fresh one so the two scenes' compositors stay independent.
    if hasattr(scene, "compositing_node_group"):
        tree = bpy.data.node_groups.new(scene.name + "_MistComp", "CompositorNodeTree")
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        scene.compositing_node_group = tree
        out_node  = tree.nodes.new("NodeGroupOutput")
        out_input = out_node.inputs[0]
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        out_node  = tree.nodes.new("CompositorNodeComposite")
        out_input = out_node.inputs["Image"]

    out_node.location = (200, 0)

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (-400, 0)
    rl.scene = scene

    map_node = _new_map_range_node(tree)
    map_node.location = (-150, 0)
    map_node.inputs["From Min"].default_value = 0.0
    map_node.inputs["From Max"].default_value = 1.0
    map_node.inputs["To Min"].default_value = 1.0 if invert else 0.0
    map_node.inputs["To Max"].default_value = 0.0 if invert else 1.0

    tree.links.new(rl.outputs["Mist"], map_node.inputs["Value"])
    tree.links.new(map_node.outputs["Result"], out_input)


def _copy_strip_settings(src, dst):
    for prop in ("blend_type", "blend_alpha", "mute",
                 "frame_offset_start", "frame_offset_end"):
        if hasattr(src, prop) and hasattr(dst, prop):
            try:
                setattr(dst, prop, getattr(src, prop))
            except Exception:
                pass

    if hasattr(src, "transform"):
        try:
            dst.transform.offset_x = src.transform.offset_x
            dst.transform.offset_y = src.transform.offset_y
            dst.transform.scale_x  = src.transform.scale_x
            dst.transform.scale_y  = src.transform.scale_y
            dst.transform.rotation = src.transform.rotation
        except Exception:
            pass


def _find_source_strip(seq_editor, inputs: ModelInputs):
    """Locate the SCENE strip this job was queued for.

    Mirrors faster_whisper_transcribe.py's lookup: prefer the active/selected
    strip, fall back to matching insert_frame_start (the source strip's own
    start frame, set per-job by SEQUENCER_OT_add_to_queue).
    """
    def _is_scene_strip(s):
        return s.type == "SCENE" and s.scene is not None

    candidate = seq_editor.active_strip
    if candidate and not _is_scene_strip(candidate):
        candidate = None
    if candidate is None:
        for seq in seq_editor.strips_all:
            if seq.select and _is_scene_strip(seq):
                candidate = seq
                break
    if candidate is None and inputs.insert_frame_start:
        target = inputs.insert_frame_start
        for seq in seq_editor.strips_all:
            if _is_scene_strip(seq) and seq.frame_final_start == target:
                candidate = seq
                break
    if candidate is None:
        scene_strips = [s for s in seq_editor.strips_all if _is_scene_strip(s)]
        if scene_strips:
            candidate = scene_strips[0]
    return candidate


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class MistDepthPassPlugin(ModelPlugin):
    MODEL_ID     = "vse-mist-depth-pass"
    DISPLAY_NAME = "Mist / Depth Pass"
    MODEL_TYPE   = "3d"
    DESCRIPTION  = (
        "Duplicate the selected Scene strip's scene, rig an EEVEE Mist "
        "compositor pass on the copy, and insert the result as a new Scene "
        "strip one channel above."
    )

    INPUTS      = InputSpec(0)   # plugin locates the SCENE strip itself
    UI_SECTIONS = []
    PARAMS      = ParamSpec()

    requires_input_strip  = True   # user must select a Scene strip
    requires_main_thread  = True   # bpy scene duplication + compositor + strip insert
    supports_batch         = False  # deterministic — one output per input strip
    show_enhance            = False

    def load(self, prefs, scene, **kw) -> dict:
        return {}

    def generate(self, pipe, inputs: ModelInputs, scene, prefs):
        # In Blender 5.x the VSE shows context.workspace.sequencer_scene, which
        # can differ from the "active" scene (the `scene` param here is the
        # scene the queue timer was started from). Strips — and therefore the
        # selected Scene strip this job is for — live in the sequencer scene,
        # not necessarily `scene`. Mirrors the same fallback chain used by
        # SEQUENCER_OT_add_to_queue.execute() (operators/queue_ops.py).
        seq_scene = (
            getattr(bpy.context, "sequencer_scene", None)
            or getattr(bpy.context.workspace, "sequencer_scene", None)
            or scene
        )
        seq_editor = seq_scene.sequence_editor
        if not seq_editor:
            print("Mist / Depth Pass: No sequence editor found.")
            return None

        strip = _find_source_strip(seq_editor, inputs)
        if strip is None:
            print("Mist / Depth Pass: No Scene strip found.")
            return None

        source_scene = strip.scene
        if source_scene is None:
            print(f"Mist / Depth Pass: Strip '{strip.name}' has no scene assigned.")
            return None

        range_mode   = getattr(scene, "mist_range_mode",   "AUTO")
        custom_start = getattr(scene, "mist_custom_start", 0.0)
        custom_depth = getattr(scene, "mist_custom_depth", 100.0)
        invert       = getattr(scene, "mist_invert",       True)

        self.set_phase(inputs, "Preparing Mist scene")
        mist_scene = _find_existing_mist_scene(source_scene)
        if mist_scene is None:
            mist_scene = _duplicate_scene(source_scene)
        else:
            print(f"Mist / Depth Pass: Reusing existing scene '{mist_scene.name}'.")
        # Always (re)apply mist settings — even on a reused scene — so
        # changing mist_range_mode / mist_invert / custom values and
        # re-running actually takes effect instead of being a no-op.
        _setup_mist(mist_scene, range_mode, custom_start, custom_depth, invert)

        self.set_phase(inputs, "Inserting Scene strip")
        _ch_hint = inputs.insert_channel if inputs.insert_channel > 0 else strip.channel + 1
        channel  = _find_free_channel(
            seq_editor, strip.frame_final_start, strip.frame_final_end, _ch_hint,
        )

        new_strip = seq_editor.strips.new_scene(
            name=strip.name + MIST_SCENE_SUFFIX,
            scene=mist_scene,
            channel=channel,
            frame_start=int(strip.frame_start),
        )
        new_strip.frame_final_start = strip.frame_final_start
        new_strip.frame_final_end   = strip.frame_final_end

        if hasattr(strip, "view_layer"):
            try:
                new_strip.view_layer = strip.view_layer
            except Exception:
                pass

        _copy_strip_settings(strip, new_strip)

        strip.select     = False
        new_strip.select = True
        seq_editor.active_strip = new_strip

        print(f"Mist / Depth Pass: Created '{new_strip.name}' on channel {channel}.")
        return None  # plugin handled its own VSE strip creation

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        col.prop(scene, "mist_range_mode")
        if scene.mist_range_mode == "CUSTOM":
            col.prop(scene, "mist_custom_start")
            col.prop(scene, "mist_custom_depth")
        col.prop(scene, "mist_invert")
        return False
