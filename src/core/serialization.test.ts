import { describe, expect, it } from "vitest";
import { SketchDocument } from "./SketchDocument";
import {
  FORMAT_VERSION,
  deserializeDocument,
  serializeDocument,
} from "./serialization";
import { importLegacyPenzil, isLegacyPenzilJson } from "./legacyPenzil";
import type { Stroke } from "./types";
import { identityTransform, newStrokeId } from "./types";

function makeStroke(): Stroke {
  return {
    id: newStrokeId(),
    points: new Float32Array([0, 0, 0, 1, 0.5, 0, 2, 1, 0.25]),
    pressure: new Float32Array([0, 0.5, 1]),
    style: { visible: true, color: "#ff0000", width: 0.012 },
    fill: { visible: true, color: "#00ff00" },
    transform: {
      position: { x: 1, y: 2, z: 3 },
      quaternion: { x: 0, y: 0.7071, z: 0, w: 0.7071 },
      scale: { x: 1, y: 1, z: 2 },
    },
    surface: { shape: "sphere", transform: identityTransform() },
  };
}

describe("serialization", () => {
  it("round-trips a document through JSON", () => {
    const doc = new SketchDocument();
    const stroke = makeStroke();
    doc.addStroke(stroke);

    // Through actual JSON text, to catch anything not JSON-representable.
    const restored = deserializeDocument(
      JSON.parse(JSON.stringify(serializeDocument(doc))),
    );

    expect(restored.strokeCount).toBe(1);
    const back = restored.getStroke(stroke.id)!;
    expect(Array.from(back.points)).toEqual(Array.from(stroke.points));
    expect(Array.from(back.pressure)).toEqual(Array.from(stroke.pressure));
    expect(back.style).toEqual(stroke.style);
    expect(back.fill).toEqual(stroke.fill);
    expect(back.transform).toEqual(stroke.transform);
    expect(back.surface).toEqual(stroke.surface);
  });

  it("rejects foreign or too-new files", () => {
    expect(() => deserializeDocument({ some: "junk" })).toThrow(/not an/);
    expect(() =>
      deserializeDocument({
        format: "interactive-arti-design",
        version: FORMAT_VERSION + 1,
        strokes: [],
      }),
    ).toThrow(/newer/);
  });
});

describe("legacy Penzil import", () => {
  // Shaped exactly like Penzil's Save.vue output.
  const legacy = [
    {
      vertices: [0, 0, 0, 0.5, 0.1, 0, 1, 0.2, 0],
      stroke: {
        show_stroke: true,
        color: "#1c1c1e",
        lineWidth: 0.006,
        force: [0, 0.4], // shorter than the vertex count, as Penzil produces
      },
      fill: { show_fill: false, color: "#1c1c1e" },
      mirrorOn: false,
      position: { x: 1, y: 0, z: -2 },
      quaternion: { _x: 0, _y: 0.7071, _z: 0, _w: 0.7071 },
      scale: { x: 1, y: 1, z: 1 },
      matrix: { elements: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] },
    },
  ];

  it("detects the legacy format", () => {
    expect(isLegacyPenzilJson(legacy)).toBe(true);
    expect(isLegacyPenzilJson({ format: "interactive-arti-design" })).toBe(
      false,
    );
    expect(isLegacyPenzilJson([])).toBe(false);
  });

  it("imports vertices, pressure, style and transform", () => {
    const strokes = importLegacyPenzil(legacy);
    expect(strokes).toHaveLength(1);
    const s = strokes[0];

    expect(s.points.length).toBe(9);
    // pressure padded to one entry per point, repeating the last force value
    expect(Array.from(s.pressure)).toEqual(Array.from(new Float32Array([0, 0.4, 0.4])));
    expect(s.style).toEqual({
      visible: true,
      color: "#1c1c1e",
      width: 0.006,
    });
    expect(s.transform.position).toEqual({ x: 1, y: 0, z: -2 });
    expect(s.transform.quaternion.y).toBeCloseTo(0.7071);
    expect(s.surface.shape).toBe("plane");
  });
});
