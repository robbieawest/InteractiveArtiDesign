// Plain-data types for the document model. No three.js here: the engine
// converts these to/from THREE.Vector3 etc. at its own boundary.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quat {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Transform {
  position: Vec3;
  quaternion: Quat;
  scale: Vec3;
}

export function identityTransform(): Transform {
  return {
    position: { x: 0, y: 0, z: 0 },
    quaternion: { x: 0, y: 0, z: 0, w: 1 },
    scale: { x: 1, y: 1, z: 1 },
  };
}

export function cloneTransform(t: Transform): Transform {
  return {
    position: { ...t.position },
    quaternion: { ...t.quaternion },
    scale: { ...t.scale },
  };
}

/** The shape of the canvas surface a stroke was drawn on. */
export type SurfaceShape = "plane" | "cube" | "cylinder" | "sphere";

export interface StrokeStyle {
  visible: boolean;
  color: string; // CSS color
  /** Base line width in world units (matches legacy Penzil's lineWidth). */
  width: number;
}

export interface FillStyle {
  visible: boolean;
  color: string;
}

export interface Stroke {
  id: string;
  /** Centerline as xyz triplets, local to `transform`. */
  points: Float32Array;
  /** One weight per centerline point, 0..1. Modulates width along the line. */
  pressure: Float32Array;
  style: StrokeStyle;
  fill: FillStyle;
  transform: Transform;
  /** The surface the stroke was drawn on, so it can be restored later. */
  surface: { shape: SurfaceShape; transform: Transform };
  /** Reserved for segmentation (phase 6). */
  partId?: string;
}

export function newStrokeId(): string {
  return crypto.randomUUID();
}

/** A named group of strokes (via `stroke.partId`) that poses and explodes
 *  as a unit. */
export interface Part {
  id: string;
  name: string;
  /** While the document is exploded: the world offset this part was moved
   *  by, so collapsing (including strokes segmented in the meantime) is the
   *  exact reverse. */
  explodeOffset?: Vec3;
}

/** A snapshot of stroke transforms, applied on demand from the poses panel. */
export interface Pose {
  id: string;
  name: string;
  /** Downscaled render of the viewport when the pose was saved (data URL). */
  thumbnail: string;
  transforms: Record<string, Transform>;
}
