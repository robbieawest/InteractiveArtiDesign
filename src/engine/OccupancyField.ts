import * as THREE from "three";
import type { Viewport } from "./Viewport";
import { DEFAULT_VOLUME_STYLE, VolumeGrid } from "./VolumeGrid";
import type { VolumeStyle } from "./VolumeGrid";
import type { OccupancyVolume } from "../surfacing/ns2sVolume";
import { blurVolume } from "./volumeBlur";

/**
 * The occupancy probabilities a surfacing method predicted, raymarched over
 * the sketch that produced them.
 *
 * Unlike the TRELLIS flow view this builds no regions and hides nothing: NS2S
 * normalizes the sketch itself and the bundle carries the way back, so the
 * field belongs exactly where the strokes already are. Laying it over them in
 * the live document is the whole point — the threshold marching cubes would
 * have baked in becomes something you move while looking at what it keeps.
 *
 * A derived view, like SurfacePreview: it renders a finished run and owns
 * none of it. Nothing here touches the document.
 */

/** NS2S's own default surface threshold, so the red region starts out as the
 *  mesh the same run would have produced. */
export const DEFAULT_OCCUPANCY_THRESHOLD = 0.6;

const STYLE: VolumeStyle = {
  ...DEFAULT_VOLUME_STYLE,
  threshold: DEFAULT_OCCUPANCY_THRESHOLD,
};

export class OccupancyFieldView {
  private readonly viewport: Viewport;
  private readonly root = new THREE.Group();
  private volumes: VolumeGrid[] = [];
  /** The fields as they arrived. Blur always recomputes from these, so
   *  dragging the slider back and forth cannot compound. */
  private fields: OccupancyVolume[] = [];
  private blur = 0;
  /** A blur asked for but not yet computed. One 112^3 pass is tens of
   *  milliseconds, and a slider drag asks for dozens — so requests coalesce
   *  onto the next frame and every superseded sigma is simply dropped. */
  private pendingBlur: number | null = null;
  private blurHandle = 0;
  private active = false;

  constructor(viewport: Viewport) {
    this.viewport = viewport;
    this.root.visible = false;
    this.viewport.scene.add(this.root);
  }

  get isActive(): boolean {
    return this.active;
  }

  /** How many fields are showing — one per part in a part-based run. */
  get count(): number {
    return this.volumes.length;
  }

  /** Show one run's fields. Replaces anything already up. */
  show(fields: OccupancyVolume[]): void {
    this.clear();
    for (const field of fields) {
      // Without an alignment nothing says where the grid sits against the
      // drawing, and a cloud at an invented pose reads as a wrong prediction
      // rather than as a missing number.
      if (!field.align) continue;
      const volume = new VolumeGrid(STYLE);
      volume.setVolume(field.voxels, field.grid);
      place(volume.object, field.align);
      this.volumes.push(volume);
      this.fields.push(field);
      this.root.add(volume.object);
    }
    this.root.visible = this.volumes.length > 0;
    this.active = this.volumes.length > 0;
    this.viewport.invalidate();
  }

  setStyle(style: Partial<VolumeStyle>): void {
    for (const volume of this.volumes) volume.setStyle(style);
    this.viewport.invalidate();
  }

  /** Smooth the field itself by `sigma` voxels, 0 for the field as predicted.
   *
   *  Not a style: this rewrites what is in the texture, so the threshold reads
   *  the smoothed values and the red region moves — which is the whole reason
   *  to blur an occupancy field rather than the image of one. */
  setBlur(sigma: number): void {
    if (sigma === this.blur && this.pendingBlur === null) return;
    this.pendingBlur = sigma;
    if (this.blurHandle) return;
    this.blurHandle = requestAnimationFrame(() => {
      this.blurHandle = 0;
      const wanted = this.pendingBlur;
      this.pendingBlur = null;
      if (wanted === null || wanted === this.blur) return;
      this.blur = wanted;
      this.fields.forEach((field, index) => {
        this.volumes[index]?.setVolume(
          blurVolume(field.voxels, field.grid, wanted),
          field.grid,
        );
      });
      this.viewport.invalidate();
    });
  }

  setVisible(visible: boolean): void {
    this.root.visible = visible && this.volumes.length > 0;
    this.viewport.invalidate();
  }

  clear(): void {
    if (this.blurHandle) cancelAnimationFrame(this.blurHandle);
    this.blurHandle = 0;
    this.pendingBlur = null;
    this.blur = 0;
    for (const volume of this.volumes) {
      this.root.remove(volume.object);
      volume.dispose();
    }
    this.volumes = [];
    this.fields = [];
    this.root.visible = false;
    this.active = false;
    this.viewport.invalidate();
  }

  dispose(): void {
    this.clear();
    this.viewport.scene.remove(this.root);
  }
}

/** world = scale * rotation * v + translation, with the box geometry already
 *  the unit cube the texture addresses. */
function place(object: THREE.Object3D, align: OccupancyVolume["align"]): void {
  if (!align) return;
  const rotation = new THREE.Matrix4().set(
    align.rotation[0], align.rotation[1], align.rotation[2], 0,
    align.rotation[3], align.rotation[4], align.rotation[5], 0,
    align.rotation[6], align.rotation[7], align.rotation[8], 0,
    0, 0, 0, 1,
  );
  object.quaternion.setFromRotationMatrix(rotation);
  object.scale.setScalar(align.scale);
  object.position.fromArray(align.translation);
}
