# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A ground-up rewrite of [Penzil](https://github.com/jacopocolo/penzil) (3D sketching on
canvas surfaces), evolving toward SketchLab-style "Rapid Design of Articulated Objects":
part segmentation, posing, exploded view, and eventually joint/articulation discovery.
See `ARCHITECTURE.md` for module layout and phase status.

## Environment

- Node is managed by **nvm**, and the shell running tool commands does not source it.
  Prefix every command that needs node/npm with:
  ```bash
  export PATH="$HOME/.nvm/versions/node/v22.23.1/bin:$PATH"
  ```
- Commands: `npm test` (vitest, run mode), `npm run build` (vue-tsc + vite),
  `npm run typecheck`, `npm run dev`.

## Hard rules from the user

- **Never use Playwright or any browser automation** to verify this app. Verify with
  `npm test` and `npm run build`, then ask the user to check visuals in their browser —
  they will report back.
- **Mouse/keyboard only.** No pen/tablet/pressure support (pressure arrays exist for
  Penzil format parity but are stored as zeros).
- UI polish is deferred; functionality first.
- Do not commit or push unless the user asks.
- Joints are screws: one axis, four independently ranged DoFs (slide, twist, swing
  U/V) — no `type` enum, kinds are derived labels. They enter a document via SketchLab
  glTF import or the Joint tool (J), which authors/edits by demonstration. Don't build
  articulation discovery or skinning unprompted.

## Architecture invariants

- Strict layering: `core ← engine ← tools ← ui`. `src/core` is pure TypeScript —
  no three.js, no DOM imports there.
- `SketchDocument` (core) is the single source of truth; it emits events and
  `StrokeRenderer` (engine) mirrors it into the scene. Never mutate the scene as the
  primary record of anything.
- All user edits go through the undo stack (`src/core/undo.ts`) as Commands; batch
  multi-stroke edits into one compound command.
- Rendering is on-demand: after anything visual changes, call `viewport.invalidate()`.
- Strokes are screen-space ribbons built in `src/engine/ribbon.ts` (2 verts/point,
  clip-space offset in the vertex shader). CPU picking uses `ray.distanceSqToSegment`
  on the centerline plus raycasts on fill meshes (`src/engine/picking.ts`).
- Serialization: `FORMAT_VERSION` in `src/core/serialization.ts` — bump it when the
  document JSON shape changes, and keep loading all older versions plus legacy Penzil
  files (`src/core/legacyPenzil.ts`).
- Highlight semantics: orange = plain selection, purple = part selection / drawing
  into a part (`HIGHLIGHT_COLORS` in `StrokeRenderer.ts`).

## Interaction model (established, don't regress)

- Camera: left = draw/select (never camera), wheel + right-drag = zoom, middle = pan,
  space = pan, alt = orbit. Gizmos hide while drawing or moving the camera
  (suppression-reasons Set) and must reappear immediately on release.
- Keybinds: D draw, E erase, V select, G segment, A articulate, J joint, T/R/S gizmo
  translate/rotate/scale, Esc deselect, Ctrl+Z/Y undo/redo. Toolbar buttons carry
  tooltips explaining these.
- Tools toggle: clicking the active tool's button (or pressing its key again) turns
  it off, back to the default idle "none" state. There is no always-on tool.
- Selection: click = stroke, double-click = whole part, ctrl+click = multi-select
  toggle (demotes a part selection to a plain one — it never changes part membership).
  Part membership is only written by the segment pen and by the draw tool while a
  part is selected (auto-join with purple preview).
- Clicking a part in the Parts panel selects it (purple), including empty parts.
- Clicking a joint in the Articulations panel switches to the Articulate tool and
  shows that joint's gizmo (the row highlights); ✎ edits it with the Joint tool
  (DoF rows appear under it for range demonstration), × deletes it.
- Explode is a persistent mode toggled from the Parts panel: in the idle no-tool
  state, drag away from the model center to spread parts (factor clamped to [0, 4],
  never imploding). Switching tools keeps the exploded layout — drawing/selecting/
  segmenting on the exploded model is the point. Only the panel toggle collapses,
  restoring the original pose exactly.

## Deployment

GitHub Pages at https://robbieawest.github.io/InteractiveArtiDesign/ via
`.github/workflows/deploy.yml` (the custom workflow — NOT GitHub's suggested starter
workflow, which publishes the unbuilt repo root and yields a white screen).
Repo setting: Pages → Source = "GitHub Actions". `vite.config.ts` uses `base: "./"`
for the subpath; keep it.
