import * as THREE from "three";
import type { SketchDocument } from "../core/SketchDocument";
import type { StrokeRenderer } from "./StrokeRenderer";
import type { Viewport } from "./Viewport";

// Ribbon meshes are zero-width on the CPU (the shader gives them width), so
// triangle raycasting can never hit them. Strokes are picked the way
// Penzil's MeshLineRaycast did it: distance from the ray to each centerline
// segment, in world space.

export interface PickableStroke {
  id: string;
  /** Centerline xyz triplets, local space. */
  points: Float32Array;
  matrixWorld: THREE.Matrix4;
}

/**
 * The stroke whose centerline passes within `threshold` (world units) of
 * the ray, nearest to the ray origin. Returns undefined if none is close
 * enough.
 */
export function pickStroke(
  ray: THREE.Ray,
  strokes: Iterable<PickableStroke>,
  threshold = 0.08,
): string | undefined {
  const thresholdSq = threshold * threshold;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const onRay = new THREE.Vector3();
  const onSegment = new THREE.Vector3();

  let bestId: string | undefined;
  let bestAlong = Infinity;

  for (const stroke of strokes) {
    const p = stroke.points;
    for (let i = 0; i + 5 < p.length; i += 3) {
      a.fromArray(p, i).applyMatrix4(stroke.matrixWorld);
      b.fromArray(p, i + 3).applyMatrix4(stroke.matrixWorld);
      const distSq = ray.distanceSqToSegment(a, b, onRay, onSegment);
      if (distSq < thresholdSq) {
        const along = onRay.distanceToSquared(ray.origin);
        if (along < bestAlong) {
          bestAlong = along;
          bestId = stroke.id;
        }
      }
    }
  }
  return bestId;
}

/**
 * Full cursor pick used by the select/erase tools: centerline distance
 * first, then the fill surfaces (fills are real triangle meshes, so a plain
 * raycast works — clicking anywhere inside a filled stroke hits it).
 */
export function pickStrokeAtCursor(
  ndc: THREE.Vector2,
  viewport: Viewport,
  doc: SketchDocument,
  strokeRenderer: StrokeRenderer,
): string | undefined {
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(ndc, viewport.camera);
  viewport.scene.updateMatrixWorld();

  const candidates: PickableStroke[] = [];
  const fillGroups: THREE.Object3D[] = [];
  for (const stroke of doc.allStrokes()) {
    const group = strokeRenderer.groupFor(stroke.id);
    if (!group) continue;
    candidates.push({
      id: stroke.id,
      points: stroke.points,
      matrixWorld: group.matrixWorld,
    });
    if (stroke.fill.visible) fillGroups.push(group);
  }

  const byCenterline = pickStroke(raycaster.ray, candidates);
  if (byCenterline) return byCenterline;

  // ribbon meshes in these groups are zero-width, so only fills can hit
  const hit = raycaster.intersectObjects(fillGroups, true)[0];
  return hit ? strokeRenderer.strokeIdFor(hit.object) : undefined;
}
