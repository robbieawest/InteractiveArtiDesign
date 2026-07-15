import type { Vec3 } from "./types";

/**
 * Rebase points (xyz triplets) so their bounding-box center becomes the
 * origin. Returns the shifted points and the center. Strokes are stored
 * local to their pivot; the pivot goes into `stroke.transform.position`
 * so selection/rotation later behaves sensibly.
 */
export function recenterPoints(points: Float32Array): {
  points: Float32Array;
  center: Vec3;
} {
  if (points.length === 0) {
    return { points, center: { x: 0, y: 0, z: 0 } };
  }

  const min = { x: Infinity, y: Infinity, z: Infinity };
  const max = { x: -Infinity, y: -Infinity, z: -Infinity };
  for (let i = 0; i < points.length; i += 3) {
    min.x = Math.min(min.x, points[i]);
    min.y = Math.min(min.y, points[i + 1]);
    min.z = Math.min(min.z, points[i + 2]);
    max.x = Math.max(max.x, points[i]);
    max.y = Math.max(max.y, points[i + 1]);
    max.z = Math.max(max.z, points[i + 2]);
  }
  const center = {
    x: (min.x + max.x) / 2,
    y: (min.y + max.y) / 2,
    z: (min.z + max.z) / 2,
  };

  const local = new Float32Array(points.length);
  for (let i = 0; i < points.length; i += 3) {
    local[i] = points[i] - center.x;
    local[i + 1] = points[i + 1] - center.y;
    local[i + 2] = points[i + 2] - center.z;
  }
  return { points: local, center };
}
