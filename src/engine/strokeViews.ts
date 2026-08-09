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

const DEFAULTS: Required<ViewSpec> = {
  size: 518,
  count: 4,
  pitch: 0.35,
  strokeColor: "#dcdcdc",
  strokeThickness: 0.012,
  margin: 1.15,
};

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

/** `count` directions evenly spaced in yaw at a constant pitch. Deterministic
 *  on purpose — a sketch always produces the same views, so a difference
 *  between two runs is the method's, not the framing's. */
export function orbitDirections(count: number, pitch: number): THREE.Vector3[] {
  const directions: THREE.Vector3[] = [];
  for (let i = 0; i < count; i++) {
    const yaw = (2 * Math.PI * i) / count;
    directions.push(
      new THREE.Vector3(
        Math.cos(pitch) * Math.sin(yaw),
        Math.sin(pitch),
        Math.cos(pitch) * Math.cos(yaw),
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
  const { count, pitch } = { ...DEFAULTS, ...spec };
  return renderSketchViews(
    sketch,
    [],
    styleFor(spec),
    orbitDirections(count, pitch),
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
