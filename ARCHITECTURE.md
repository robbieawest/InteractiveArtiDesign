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
   - **Exploded view** — per-part outward offsets stored on the parts
     (`core/explode.ts`); collapse reverses exactly, including strokes
     segmented while exploded. (done)
   - Joints (sliding/revolute), articulation discovery, skinning: not yet.
     The Articulations panel is a placeholder.
