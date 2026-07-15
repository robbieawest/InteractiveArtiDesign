import type {
  FillStyle,
  Stroke,
  StrokeStyle,
  SurfaceShape,
  Transform,
} from "./types";
import { SketchDocument } from "./SketchDocument";

// Versioned on-disk format. Typed arrays become plain number arrays in JSON;
// everything else is already plain data.

export const FORMAT_NAME = "interactive-arti-design";
export const FORMAT_VERSION = 2; // 1 = legacy Penzil (see legacyPenzil.ts)

interface StrokeJson {
  id: string;
  points: number[];
  pressure: number[];
  style: StrokeStyle;
  fill: FillStyle;
  transform: Transform;
  surface: { shape: SurfaceShape; transform: Transform };
  partId?: string;
}

export interface DocumentJson {
  format: typeof FORMAT_NAME;
  version: number;
  strokes: StrokeJson[];
}

export function serializeDocument(doc: SketchDocument): DocumentJson {
  return {
    format: FORMAT_NAME,
    version: FORMAT_VERSION,
    strokes: doc.allStrokes().map((s) => ({
      id: s.id,
      points: Array.from(s.points),
      pressure: Array.from(s.pressure),
      style: { ...s.style },
      fill: { ...s.fill },
      transform: s.transform,
      surface: s.surface,
      ...(s.partId !== undefined && { partId: s.partId }),
    })),
  };
}

export function deserializeDocument(json: unknown): SketchDocument {
  const data = json as DocumentJson;
  if (data?.format !== FORMAT_NAME) {
    throw new Error("not an InteractiveArtiDesign document");
  }
  if (data.version > FORMAT_VERSION) {
    throw new Error(`document version ${data.version} is newer than this app`);
  }
  const doc = new SketchDocument();
  for (const s of data.strokes) {
    const stroke: Stroke = {
      id: s.id,
      points: new Float32Array(s.points),
      pressure: new Float32Array(s.pressure),
      style: s.style,
      fill: s.fill,
      transform: s.transform,
      surface: s.surface,
      ...(s.partId !== undefined && { partId: s.partId }),
    };
    doc.addStroke(stroke);
  }
  return doc;
}
