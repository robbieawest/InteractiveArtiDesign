// Multi-view stroke renders for methods that condition on images of the
// sketch rather than on the strokes as geometry.
//
// Shares its scene plumbing with the benchmark thumbnails via sketchRender.ts;
// what differs is that the *method* dictates the style. A ViewSpec arrives
// with the method declaration from the server (SurfacingAdapter.view_spec) and
// says how many views, how big, how thick and what colour. That indirection is
// the point: which numbers a model needs is a fact about the model — TRELLIS
// wants light thick strokes at 518px because of what its preprocessing and
// DINOv2 do to them — and encoding it here would put one experiment's
// requirements inside the editor. The defaults below exist only so a partial
// spec still renders something sane.

import * as THREE from "three";
import {
  renderSketchViews,
  disposeSketchRenderer,
  type SketchRenderStyle,
} from "./sketchRender";
import type { SurfacingSketch, ViewSpec } from "../surfacing/client";

const DEFAULTS: Required<Omit<ViewSpec, "overrides">> = {
  size: 518,
  count: 4,
  pitch: 0.35,
  layout: "ring",
  pitchMax: 1.2,
  yaw: 0,
  strokeColor: "#dcdcdc",
  strokeThickness: 0.012,
  margin: 1.15,
};

/** How close to straight down a view may get, in radians. Past this the view
 *  direction is parallel to the camera's up vector and the lookAt basis is
 *  degenerate — three.js resolves it to an arbitrary roll, so a "top" view
 *  would come out spun by some amount that depends on nothing. Just under 80
 *  degrees keeps a recognizable horizon in the image. */
const MAX_PITCH = 1.4;

/** Shading for a surfaced conditioning render (`renderSurfacedViews`). The
 *  strokes' own colour is a compromise struck for thin geometry — a tube a few
 *  pixels wide has to be light enough to survive the model's preprocessing —
 *  and a solid filling the frame wants the opposite of that at its edges. */
const SURFACED_ALBEDO = 0xe8e8e8;
const SURFACED_CONTOUR = 0x3c3c3c;
const SURFACED_CONTOUR_STRENGTH = 0.75;

function styleFor(spec: ViewSpec): SketchRenderStyle {
  const merged = { ...DEFAULTS, ...spec };
  return {
    size: merged.size,
    strokeColor: new THREE.Color(merged.strokeColor).getHex(),
    strokeThickness: merged.strokeThickness,
    // conditioning renders are of strokes alone; a surface would be the
    // model's own answer fed back to it
    surfaceColor: new THREE.Color(merged.strokeColor).getHex(),
    // anything under 1 drops alpha below the crop thresholds image models
    // tend to use, and a translucent stroke is not a stroke
    surfaceOpacity: 1,
    margin: merged.margin,
    // otherwise the rear views come out unlit
    headlight: true,
    ambientIntensity: 0.55,
    keyIntensity: 1.0,
  };
}

/** `count` directions evenly spaced in yaw. Deterministic on purpose — a
 *  sketch always produces the same views, so a difference between two runs is
 *  the method's, not the framing's.
 *
 *  `ring` holds one elevation, which makes every view a side view: cheap to
 *  reason about, and the views differ only in what the yaw reveals. `helix`
 *  climbs from `pitch` to `pitchMax` as it goes round, so the set ends looking
 *  down at the sketch. That matters for image-conditioned methods that take no
 *  camera poses: they reconcile views from image content alone, so two views
 *  that look alike are ambiguity rather than evidence, and elevation separates
 *  them far better than more yaws at the same height do.
 *
 *  `yaw0` offsets where the sequence starts. It is invisible to a full orbit —
 *  a ring is the same ring wherever it begins — and exists for the single-view
 *  case, where the one camera would otherwise be square to the sketch and show
 *  no depth at all. */
export function orbitDirections(
  count: number,
  pitch: number,
  layout: "ring" | "helix" = "ring",
  pitchMax: number = pitch,
  yaw0: number = 0,
): THREE.Vector3[] {
  const top = Math.min(pitchMax, MAX_PITCH);
  const base = Math.min(pitch, MAX_PITCH);
  const directions: THREE.Vector3[] = [];
  for (let i = 0; i < count; i++) {
    const yaw = yaw0 + (2 * Math.PI * i) / count;
    // a one-view helix has nowhere to climb to, and dividing by count - 1
    // would be a division by zero
    const elevation =
      layout === "helix" && count > 1
        ? base + ((top - base) * i) / (count - 1)
        : base;
    directions.push(
      new THREE.Vector3(
        Math.cos(elevation) * Math.sin(yaw),
        Math.sin(elevation),
        Math.cos(elevation) * Math.cos(yaw),
      ),
    );
  }
  return directions;
}

/** Render a sketch from several angles as PNG data URLs. */
export async function renderStrokeViews(
  sketch: SurfacingSketch,
  spec: ViewSpec = {},
): Promise<string[]> {
  const { count, pitch, layout, pitchMax, yaw } = { ...DEFAULTS, ...spec };
  return renderSketchViews(
    sketch,
    [],
    styleFor(spec),
    orbitDirections(count, pitch, layout, pitchMax, yaw),
  );
}

/** Views of a surfaced sketch: the same cameras and the same framing, with a
 *  shaded solid in place of the line art.
 *
 *  The strokes are left out rather than drawn over the surface. A model that
 *  reconstructs what it is shown is exactly the thing this is trying to stop
 *  showing a wireframe to, and strokes standing proud of the surface would put
 *  some of it back — the sketch is already in the run as geometry, through the
 *  inpainting constraint, which is where it belongs.
 *
 *  Opaque, and lit the same way: `styleFor` already pins `surfaceOpacity` to 1
 *  because image models crop on alpha, and the surface picks up the same
 *  colour the strokes would have had, so the only difference from a stroke
 *  render is what is in the frame. */
export async function renderSurfacedViews(
  sketch: SurfacingSketch,
  surface: ArrayBuffer,
  spec: ViewSpec = {},
): Promise<string[]> {
  const { count, pitch, layout, pitchMax, yaw } = { ...DEFAULTS, ...spec };
  return renderSketchViews(
    { ...sketch, strokes: [] },
    [surface],
    {
      ...styleFor(spec),
      // Lighter than the strokes would have been, with the contour dropped to
      // a dark grey: a solid needs the two ends of the range that line art
      // gets for free, and the matcap alone leaves a pale object low-contrast
      // against a pale background.
      surfaceColor: SURFACED_ALBEDO,
      surfaceContourColor: SURFACED_CONTOUR,
      surfaceContourStrength: SURFACED_CONTOUR_STRENGTH,
    },
    orbitDirections(count, pitch, layout, pitchMax, yaw),
  );
}

/** Views for a whole sketch, or one set per part when the run is part-based.
 *
 *  Keyed by part *id*, not name: names are user-typed and not unique, so two
 *  parts called "wheel" would collide into one set and both would be surfaced
 *  from the same views.
 *
 *  Parts are rendered in isolation rather than cropped out of a whole-sketch
 *  render: the point of a part-based run is that the model never sees the
 *  neighbouring geometry, and a shared frame would also shrink each part to
 *  the whole object's scale. */
export async function renderConditioningViews(
  sketch: SurfacingSketch,
  spec: ViewSpec = {},
  partBased = false,
): Promise<string[] | Record<string, string[]>> {
  if (!partBased) return renderStrokeViews(sketch, spec);

  const groups = new Map<string, SurfacingSketch["strokes"]>();
  for (const stroke of sketch.strokes) {
    if (stroke.partId === null) continue; // part-based runs ignore these
    const group = groups.get(stroke.partId);
    if (group) group.push(stroke);
    else groups.set(stroke.partId, [stroke]);
  }

  const views: Record<string, string[]> = {};
  for (const [partId, strokes] of groups) {
    views[partId] = await renderStrokeViews({ ...sketch, strokes }, spec);
  }
  return views;
}

export { disposeSketchRenderer };
