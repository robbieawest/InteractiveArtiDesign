import type { SketchDocument } from "./SketchDocument";
import type { Vec3 } from "./types";

export interface ExplodeLayout {
  /** Outward offset per part at explode factor 1; scale for other factors. */
  offsets: Map<string, Vec3>;
  /** The sketch's overall center (mean of part centers), world space. */
  center: Vec3;
}

/**
 * Outward offset per part for the exploded view: each part moves along the
 * direction from the sketch's overall center to the part's own center
 * (stroke pivots averaged), by a distance proportional to the sketch's
 * spread so small and large sketches both separate visibly.
 */
export function computeExplodeLayout(doc: SketchDocument): ExplodeLayout {
  const centers = new Map<string, Vec3>();
  for (const part of doc.allParts()) {
    const strokes = doc.strokesInPart(part.id);
    if (strokes.length === 0) continue;
    const c = { x: 0, y: 0, z: 0 };
    for (const s of strokes) {
      c.x += s.transform.position.x;
      c.y += s.transform.position.y;
      c.z += s.transform.position.z;
    }
    centers.set(part.id, {
      x: c.x / strokes.length,
      y: c.y / strokes.length,
      z: c.z / strokes.length,
    });
  }
  if (centers.size === 0) {
    return { offsets: new Map(), center: { x: 0, y: 0, z: 0 } };
  }

  const global = { x: 0, y: 0, z: 0 };
  for (const c of centers.values()) {
    global.x += c.x;
    global.y += c.y;
    global.z += c.z;
  }
  global.x /= centers.size;
  global.y /= centers.size;
  global.z /= centers.size;

  let radius = 0;
  for (const c of centers.values()) {
    radius = Math.max(
      radius,
      Math.hypot(c.x - global.x, c.y - global.y, c.z - global.z),
    );
  }
  const spread = Math.max(1.5, radius * 1.2);

  const offsets = new Map<string, Vec3>();
  let i = 0;
  for (const [partId, c] of centers) {
    const d = {
      x: c.x - global.x,
      y: c.y - global.y,
      z: c.z - global.z,
    };
    const len = Math.hypot(d.x, d.y, d.z);
    if (len < 1e-6) {
      // coincident with the center (e.g. a single part): fan out on a
      // horizontal circle so parts still separate deterministically
      const angle = (i / centers.size) * Math.PI * 2;
      offsets.set(partId, {
        x: Math.cos(angle) * spread,
        y: 0,
        z: Math.sin(angle) * spread,
      });
    } else {
      offsets.set(partId, {
        x: (d.x / len) * spread,
        y: (d.y / len) * spread,
        z: (d.z / len) * spread,
      });
    }
    i++;
  }
  return { offsets, center: global };
}
