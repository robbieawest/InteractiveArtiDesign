import type { Stroke, Transform } from "./types";
import { identityTransform, newStrokeId } from "./types";

// Importer for sketches saved by the original Penzil app: a JSON array of
// lines, each { vertices, stroke, fill, position, quaternion, scale, matrix }.
// Notes on that format:
//  - `vertices` is a flat xyz number array, local to the stroke's pivot.
//  - `stroke.force` holds per-vertex pen pressure; its length can drift from
//    the vertex count because of Penzil's input smoothing, so we clamp/pad.
//  - `quaternion` serialized THREE.Quaternion's private fields, so the keys
//    are `_x`, `_y`, `_z`, `_w`.
//  - The canvas the stroke was drawn on was NOT saved, only used at runtime,
//    so imported strokes get a default plane surface.

interface LegacyLine {
  vertices: number[];
  stroke: {
    show_stroke: boolean;
    color: string;
    lineWidth: number;
    force?: number[];
  };
  fill: { show_fill: boolean; color: string };
  position?: { x: number; y: number; z: number };
  quaternion?: { _x: number; _y: number; _z: number; _w: number };
  scale?: { x: number; y: number; z: number };
}

export function isLegacyPenzilJson(json: unknown): json is LegacyLine[] {
  return (
    Array.isArray(json) &&
    json.length > 0 &&
    typeof json[0] === "object" &&
    json[0] !== null &&
    "vertices" in json[0] &&
    "stroke" in json[0]
  );
}

export function importLegacyPenzil(json: LegacyLine[]): Stroke[] {
  return json.map((line) => {
    const points = new Float32Array(line.vertices);
    const pointCount = points.length / 3;

    const force = line.stroke.force ?? [];
    const pressure = new Float32Array(pointCount);
    for (let i = 0; i < pointCount; i++) {
      pressure[i] = force[Math.min(i, force.length - 1)] ?? 0;
    }

    const transform: Transform = identityTransform();
    if (line.position) transform.position = { ...line.position };
    if (line.quaternion) {
      transform.quaternion = {
        x: line.quaternion._x,
        y: line.quaternion._y,
        z: line.quaternion._z,
        w: line.quaternion._w,
      };
    }
    if (line.scale) transform.scale = { ...line.scale };

    return {
      id: newStrokeId(),
      points,
      pressure,
      style: {
        visible: line.stroke.show_stroke,
        color: line.stroke.color,
        width: line.stroke.lineWidth,
      },
      fill: {
        visible: line.fill.show_fill,
        color: line.fill.color,
      },
      transform,
      surface: { shape: "plane", transform: identityTransform() },
    } satisfies Stroke;
  });
}
