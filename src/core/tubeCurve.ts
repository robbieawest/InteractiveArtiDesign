// Recovers a stroke centerline from a SketchLab "Projectable_Polyline" tube
// mesh. SketchLab (Rapid Design of Articulated Objects) exports each stroke
// as a triangular tube: k rings of 3 vertices around the centerline plus one
// cap vertex at each end, so the vertex count is always 3k + 2. The layout
// (verified against the triangle indices of the excavator sample) is:
//
//   verts 0, 1, 3        first ring
//   vert 2               start-cap point
//   verts 4 .. 3k        remaining rings, 3 consecutive verts each
//   vert 3k + 1 (last)   end-cap point
//
// Each ring's centroid is a centerline point; the ring radius carries the
// stroke's width (tapered toward the ends, where SketchLab baked pressure in).

export interface TubeCenterline {
  /** Ring centroids as xyz triplets, one per centerline point. */
  points: Float32Array;
  /** Ring radius per centerline point, in the same units as the positions. */
  radii: Float32Array;
}

export function isTubeVertexCount(vertexCount: number): boolean {
  return vertexCount >= 5 && (vertexCount - 2) % 3 === 0;
}

/** `positions` are xyz triplets from the mesh's POSITION attribute, in any
 *  space — world-transform them first if world output is wanted. */
export function reconstructTubeCenterline(
  positions: Float32Array,
): TubeCenterline {
  const vertexCount = positions.length / 3;
  if (!isTubeVertexCount(vertexCount)) {
    throw new Error(
      `not a SketchLab tube: ${vertexCount} vertices (expected 3k + 2)`,
    );
  }

  const ringCount = (vertexCount - 2) / 3;
  const points = new Float32Array(ringCount * 3);
  const radii = new Float32Array(ringCount);

  for (let ring = 0; ring < ringCount; ring++) {
    const verts: [number, number, number] =
      ring === 0 ? [0, 1, 3] : [ring * 3 + 1, ring * 3 + 2, ring * 3 + 3];

    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const v of verts) {
      cx += positions[v * 3];
      cy += positions[v * 3 + 1];
      cz += positions[v * 3 + 2];
    }
    cx /= 3;
    cy /= 3;
    cz /= 3;
    points[ring * 3] = cx;
    points[ring * 3 + 1] = cy;
    points[ring * 3 + 2] = cz;

    let radius = 0;
    for (const v of verts) {
      radius += Math.hypot(
        positions[v * 3] - cx,
        positions[v * 3 + 1] - cy,
        positions[v * 3 + 2] - cz,
      );
    }
    radii[ring] = radius / 3;
  }

  return { points, radii };
}
