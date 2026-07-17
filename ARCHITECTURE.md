# Architecture


Stack: Vite, TypeScript, Vue 3, three.js, camera-controls.

## Layout and dependency rule

```
src/
  core/    # document model: strokes, parts, undo, serialization. Pure TS,
           # no three.js, no DOM — unit-testable in isolation.
  engine/  # three.js layer: viewport, render loop, stroke meshes, gizmos.
  tools/   # draw / select / erase state machines consuming pointer events.
  ui/      # Vue components. Talks to core/engine, never the other way.
```

Imports may only point left-to-right in `core ← engine ← tools ← ui`.
The document (in `core/`) is the source of truth; three.js meshes are a
derived view of it. UI framework code stays out of the engine so the shell
is replaceable.

## Rendering

One requestAnimationFrame loop in `engine/Viewport.ts`. Scene changes call
`viewport.invalidate()`; the loop renders only when invalidated or when the
camera moved. Nothing else calls `renderer.render`.

## Camera bindings

The left mouse button is reserved for tools (drawing, selecting). Camera:
wheel = zoom, right drag = zoom, middle drag = pan, space+drag = pan,
alt+drag = orbit.

## Phases

1. **Viewport shell** — renderer, camera controls, infinite grid. (done)
2. **Document model** — `core/`: strokes, versioned JSON serialization,
   legacy Penzil import, command-pattern undo. (done)
3. **Stroke rendering** — `engine/ribbon.ts` (screen-space ribbon shader,
   per-vertex width) + `engine/StrokeRenderer.ts` (document → scene sync),
   fill via Earcut, Load button for legacy/v2 files. (done)
4. **Draw tool** — `engine/CanvasSurface.ts` (placeable plane/cube/cylinder/
   sphere with TransformControls gizmo) + `tools/DrawTool.ts` (raycast
   projection, moving-average smoothing, live preview, undoable commit).
   Toolbar: canvas shape/gizmo/visibility, undo/redo (Ctrl+Z/Ctrl+Shift+Z),
   save/load. (done)
5. **Tool parity** — `tools/EraseTool.ts` + `tools/SelectTool.ts` with
   centerline picking (`engine/picking.ts`), mirror drawing, line settings
   (color/width/fill), GLB export as variable-radius tubes
   (`engine/exportGlb.ts`). Keybinds: D/E/V tools, T/R/S gizmo mode,
   Delete/Esc selection, Ctrl+Z/Y undo. (done)
6. Articulation (in progress):
   - **Parts** — `stroke.partId` + part registry in the document;
     `tools/SegmentTool.ts` paints strokes into/out of the active part with
     a through-everything pen (`pickAllStrokes`). (done)
   - **Posing** — multi-select in `tools/SelectTool.ts` (double-click =
     part) with one pivot gizmo; named pose snapshots with viewport
     thumbnails, saved in the document. (done)
   - **Exploded view** — a persistent mode toggled from the Parts panel:
     in the idle no-tool state, dragging away from the model center
     (`tools/ExplodeTool.ts`) scales per-part outward offsets
     (`core/explode.ts`, always computed at the rest pose) by a factor
     clamped to [0, 4] — it never goes past the original pose. Offsets are
     stored on the parts; the mode survives tool switches so drawing,
     selecting, and segmenting work on the exploded model. Turning the mode
     off collapses exactly, including strokes added while exploded. (done)
   - **SketchLab import** — `engine/importSketchLab.ts` loads glTF/GLB
     exports of SketchLab documents (e.g. `SampleModels/`): stroke
     centerlines are recovered from the triangular tube meshes
     (`core/tubeCurve.ts`), baked to absolute world space, and tagged with
     their `Part_*` ancestor. `Joint_*` nodes give pivots directly; type,
     axis, and range are mined from the embedded animation (delta rotations
     → revolute axis, translation deltas → prismatic axis, observed
     excursion → range). (done)
   - **Joints** — screw-joint edges between parts (`core/types.ts`): one
     axis (pivot point + unit direction, Plücker-style) with four
     independent, individually ranged DoFs — `translation` (slide along the
     axis), `twist` (about it), and `swingU`/`swingV` (about a deterministic
     perpendicular basis, `jointBasis`). Fixed/revolute/prismatic/
     cylindrical/ball are just labels derived from which DoFs are unlocked
     (`jointKindLabel`); a locked DoF has range [0, 0]. Serialized since
     format v5 (v4 typed joints migrate on load). Design decision (settled):
     strokes always store absolute transforms and hierarchy lives on the
     joint edges — never relative coordinate frames on strokes. Pivot/axis
     are world-space at rest (all values 0); FK composes per-part delta
     transforms (`core/articulation.ts` + `core/rigid.ts`, composition
     order swing U → swing V → twist → slide), and articulating patches
     member strokes by Δ(new)∘Δ(old)⁻¹ through the undo stack, exactly the
     explode pattern. (done)
   - **Articulate tool** — `tools/ArticulateTool.ts` (keybind A): click a
     part to get a gizmo constrained to its driving joint's axis (ring for
     revolute, arrow for prismatic), clamped to the range. Clicking a joint
     in the Articulations panel selects it the same way (switches to the
     tool and shows its gizmo); the panel lists joints, resets the pose,
     and toggles IK:
     with IK on, dragging a part translates it freely and CCD
     (`solveIK`, per-DoF) solves the joint chain backwards. The rotate
     gizmo shows one ring per unlocked rotational DoF; T/R switch to the
     slide arrow when a joint has both. Disabled while exploded (rest
     pivots don't apply). (done)
   - **Joint tool** — `tools/JointTool.ts` (keybind J): authoring and
     editing are one process. Drag from a parent part to a child part to
     create a joint (default axis parent→child at the child's centroid,
     all DoFs locked; the child's old driver is replaced in the same undo
     step, and cycles are refused). Click a part to edit its driving
     joint: parent highlights purple, child yellow, and a gizmo places the
     axis (T moves the pivot, R aims the direction — roll is hidden since
     U/V derive from the axis). Ranges are *demonstrated*: arm a DoF in
     the Articulations panel and drag the child through the motion — the
     extremes reached become the range (mirror toggle symmetrizes); on
     release the part snaps back to rest. Ranges are visualized as
     translucent fills (`engine/jointRangeVisual.ts`): pie sectors in each
     rotational DoF's sweep plane and a bar along the axis for slide, live
     while demonstrating, committed ranges otherwise. The tool always works
     at the rest pose (posed joints are zeroed first, undoably).
     `engine/JointLines.ts` draws parent↔child lines while exploded. (done)
   - Articulation discovery, skinning: not yet.
