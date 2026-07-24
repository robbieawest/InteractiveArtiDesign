import type {
  FillStyle,
  Joint,
  Part,
  Pose,
  Stroke,
  StrokeStyle,
  SurfaceShape,
  Transform,
  Vec3,
} from "./types";
import { lockedDofs } from "./types";
import { SketchDocument } from "./SketchDocument";

// Versioned on-disk format. Typed arrays become plain number arrays in JSON;
// everything else is already plain data.

export const FORMAT_NAME = "interactive-arti-design";
// 1 = legacy Penzil (see legacyPenzil.ts); 2 = strokes only;
// 3 = adds parts, poses, exploded state; 4 = adds typed joints;
// 5 = joints become screws (per-DoF ranges/values instead of a type);
// 6 = optional embedded surfacing result (`surface`, written by the ui)
export const FORMAT_VERSION = 6;

/** Version-4 joint shape, kept for migration. */
interface JointJsonV4 {
  id: string;
  name: string;
  parentPartId: string;
  childPartId: string;
  type: "fixed" | "revolute" | "prismatic";
  pivot: Vec3;
  axis: Vec3;
  range: [number, number];
  value: number;
}

function migrateJoint(json: Joint | JointJsonV4): Joint {
  if (!("type" in json)) return json;
  const dofs = lockedDofs();
  if (json.type === "revolute") {
    dofs.twist = { range: [...json.range], value: json.value };
  } else if (json.type === "prismatic") {
    dofs.translation = { range: [...json.range], value: json.value };
  }
  return {
    id: json.id,
    name: json.name,
    parentPartId: json.parentPartId,
    childPartId: json.childPartId,
    pivot: json.pivot,
    axis: json.axis,
    dofs,
  };
}

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

/** The last surfacing result, embedded so a saved sketch reopens with its
 *  surface. Written and read by the ui layer only: the mesh is derived
 *  output and never enters the SketchDocument or the undo stack. */
export interface SurfaceJson {
  /** Adapter name that produced the mesh (e.g. "vns"). */
  method: string;
  /** The result .glb, base64-encoded. */
  glb: string;
}

export interface DocumentJson {
  format: typeof FORMAT_NAME;
  version: number;
  strokes: StrokeJson[];
  /** Since version 3. */
  parts?: Part[];
  poses?: Pose[];
  exploded?: boolean;
  /** Since version 4; version-5 screw shape or version-4 typed shape. */
  joints?: (Joint | JointJsonV4)[];
  /** Since version 6; absent when nothing was surfaced. */
  surface?: SurfaceJson;
}

export function serializeDocument(doc: SketchDocument): DocumentJson {
  return {
    format: FORMAT_NAME,
    version: FORMAT_VERSION,
    parts: doc.allParts(),
    poses: doc.allPoses(),
    joints: doc.allJoints(),
    exploded: doc.exploded,
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
  for (const part of data.parts ?? []) doc.addPart(part);
  for (const pose of data.poses ?? []) doc.addPose(pose);
  for (const joint of data.joints ?? []) doc.addJoint(migrateJoint(joint));
  doc.exploded = data.exploded ?? false;
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
