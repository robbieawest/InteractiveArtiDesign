// Offscreen thumbnail renderer for the benchmark window.
//
// Thumbnails are cheap on purpose: strokes draw as plain line segments rather
// than the screen-space ribbons the editor uses (or the tubes strokeViews.ts
// needs), because at 200px the difference is invisible and a ribbon pass per
// sketch is not. The scene plumbing lives in sketchRender.ts; this module is
// just the thumbnail's style and its one camera angle.

import * as THREE from "three";
import {
  renderSketchViews,
  disposeSketchRenderer,
  type SketchRenderStyle,
} from "./sketchRender";
import type { SurfacingSketch } from "../surfacing/client";

const THUMBNAIL_STYLE: SketchRenderStyle = {
  size: 256,
  strokeColor: 0x333333,
  strokeThickness: 0, // hairlines: invisible difference at this size
  surfaceColor: 0xff9c3c,
  surfaceOpacity: 0.85,
  // 1.02 rather than a comfortable margin: at 256px the sketch needs to fill
  // the frame to be recognisable in a grid
  margin: 1.02,
  headlight: false, // one fixed angle, so fixed lighting reads better
  ambientIntensity: 0.75,
  keyIntensity: 0.9,
};

/** Every thumbnail uses the same three-quarter angle so the grid reads as a
 *  set. */
const THUMBNAIL_DIRECTION = new THREE.Vector3(0.6, 0.45, 1);

/** Frees the shared context. Call when the benchmark window closes for good;
 *  the next thumbnail transparently makes a new one. */
export function disposeThumbnailRenderer(): void {
  disposeSketchRenderer();
}

/** Render one thumbnail: the sketch's stored pose, plus whatever surface
 *  geometry has arrived so far (per-part partials, or the finished object).
 *  Returns a data URL; the caller caches it and only re-renders when the
 *  geometry set changes. */
export async function renderThumbnail(
  sketch: SurfacingSketch,
  surfaces: ArrayBuffer[] = [],
): Promise<string> {
  const [url] = await renderSketchViews(sketch, surfaces, THUMBNAIL_STYLE, [
    THUMBNAIL_DIRECTION,
  ]);
  return url;
}
