import * as THREE from "three";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack } from "../core/undo";
import type { FillStyle, Stroke, StrokeStyle } from "../core/types";
import { newStrokeId } from "../core/types";
import { recenterPoints } from "../core/geometry";
import type { Viewport } from "../engine/Viewport";
import type { CanvasSurface } from "../engine/CanvasSurface";
import {
  buildRibbonGeometry,
  computeVertexWidths,
  createRibbonMaterial,
} from "../engine/ribbon";
import { HIGHLIGHT_COLORS } from "../engine/StrokeRenderer";

const SMOOTHING_WINDOW = 4;

/**
 * Left-drag draws a stroke onto the canvas surface: each pointer move is
 * raycast through the camera onto the surface mesh, smoothed with a small
 * moving average (Penzil's approach), and shown via a live preview mesh.
 * On release the points are recentered on their pivot and committed to the
 * document through the undo stack.
 */
export type MirrorAxis = "off" | "x" | "y" | "z";

export class DrawTool {
  strokeStyle: StrokeStyle = { visible: true, color: "#1c1c1e", width: 0.016 };
  fillStyle: FillStyle = { visible: false, color: "#1c1c1e" };
  /** When set, each stroke also commits a twin reflected across the world
   *  plane perpendicular to this axis. The twin is an independent stroke
   *  (unlike Penzil's live-linked mirrors). */
  mirrorAxis: MirrorAxis = "off";
  /** While a part is selected, new strokes join it automatically; the live
   *  preview glows purple so the user knows. */
  targetPartId?: string;

  private drawing = false;
  private points: number[] = [];
  private smoothingBuffer: THREE.Vector3[] = [];
  private previewMesh?: THREE.Mesh;
  private readonly raycaster = new THREE.Raycaster();

  constructor(
    private readonly viewport: Viewport,
    private readonly surface: CanvasSurface,
    private readonly doc: SketchDocument,
    private readonly undo: UndoStack,
  ) {}

  attach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.addEventListener("pointerdown", this.onPointerDown);
    dom.addEventListener("pointermove", this.onPointerMove);
    dom.addEventListener("pointerup", this.onPointerUp);
    dom.addEventListener("pointercancel", this.onPointerCancel);
  }

  detach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.removeEventListener("pointerdown", this.onPointerDown);
    dom.removeEventListener("pointermove", this.onPointerMove);
    dom.removeEventListener("pointerup", this.onPointerUp);
    dom.removeEventListener("pointercancel", this.onPointerCancel);
    this.drawing = false;
    this.surface.suppressGizmo("drawing", false);
    this.discardPreview();
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (
      event.button !== 0 ||
      this.viewport.cameraOwnsPointer ||
      this.surface.gizmoActive
    ) {
      return;
    }
    this.drawing = true;
    this.points = [];
    this.smoothingBuffer = [];
    this.surface.suppressGizmo("drawing", true);
    this.viewport.renderer.domElement.setPointerCapture(event.pointerId);
    this.addPoint(event);
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (!this.drawing) return;
    this.addPoint(event);
    this.updatePreview();
  };

  private onPointerUp = (event: PointerEvent): void => {
    if (!this.drawing || event.button !== 0) return;
    this.drawing = false;
    this.viewport.renderer.domElement.releasePointerCapture(event.pointerId);
    this.commit();
  };

  private onPointerCancel = (): void => {
    if (!this.drawing) return;
    this.drawing = false;
    this.surface.suppressGizmo("drawing", false);
    this.discardPreview();
    this.viewport.invalidate();
  };

  /** Raycast the cursor onto the drawing surface; misses are skipped, so
   *  strokes pause at the surface's edge exactly like Penzil. */
  private addPoint(event: PointerEvent): void {
    const dom = this.viewport.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.viewport.camera);
    const hit = this.raycaster.intersectObject(this.surface.mesh, false)[0];
    if (!hit) return;

    this.smoothingBuffer.push(hit.point);
    if (this.smoothingBuffer.length > SMOOTHING_WINDOW) {
      this.smoothingBuffer.shift();
    }
    const smoothed = new THREE.Vector3();
    for (const p of this.smoothingBuffer) smoothed.add(p);
    smoothed.divideScalar(this.smoothingBuffer.length);

    this.points.push(smoothed.x, smoothed.y, smoothed.z);
  }

  private updatePreview(): void {
    const pointCount = this.points.length / 3;
    if (pointCount < 2) return;

    const points = new Float32Array(this.points);
    const geometry = buildRibbonGeometry(
      points,
      computeVertexWidths(new Float32Array(pointCount)), // mouse: no pressure
    );

    if (!this.previewMesh) {
      const material = createRibbonMaterial(
        this.strokeStyle.color,
        this.strokeStyle.width,
        this.viewport.resolution,
      );
      if (this.targetPartId) {
        material.uniforms.highlight.value = 1;
        (material.uniforms.highlightColor.value as THREE.Color).copy(
          HIGHLIGHT_COLORS.part,
        );
      }
      this.previewMesh = new THREE.Mesh(geometry, material);
      this.viewport.scene.add(this.previewMesh);
    } else {
      this.previewMesh.geometry.dispose();
      this.previewMesh.geometry = geometry;
    }
    this.viewport.invalidate();
  }

  private commit(): void {
    this.surface.suppressGizmo("drawing", false);
    this.discardPreview();

    if (this.points.length / 3 >= 2) {
      const strokes = [this.makeStroke(this.points)];
      if (this.mirrorAxis !== "off") {
        strokes.push(this.makeStroke(mirrorPoints(this.points, this.mirrorAxis)));
      }

      const doc = this.doc;
      this.undo.push({
        label: "Draw stroke",
        execute: () => strokes.forEach((s) => doc.addStroke(s)),
        undo: () => strokes.forEach((s) => doc.removeStroke(s.id)),
      });
    }

    this.points = [];
    this.smoothingBuffer = [];
    this.viewport.invalidate();
  }

  private makeStroke(worldPoints: number[]): Stroke {
    const { points, center } = recenterPoints(new Float32Array(worldPoints));
    return {
      id: newStrokeId(),
      points,
      pressure: new Float32Array(worldPoints.length / 3),
      style: { ...this.strokeStyle },
      fill: { ...this.fillStyle },
      transform: {
        position: center,
        quaternion: { x: 0, y: 0, z: 0, w: 1 },
        scale: { x: 1, y: 1, z: 1 },
      },
      surface: this.surface.snapshot(),
      ...(this.targetPartId && { partId: this.targetPartId }),
    };
  }

  private discardPreview(): void {
    if (!this.previewMesh) return;
    this.viewport.scene.remove(this.previewMesh);
    this.previewMesh.geometry.dispose();
    (this.previewMesh.material as THREE.Material).dispose();
    this.previewMesh = undefined;
  }
}

function mirrorPoints(points: number[], axis: "x" | "y" | "z"): number[] {
  const offset = axis === "x" ? 0 : axis === "y" ? 1 : 2;
  const mirrored = [...points];
  for (let i = offset; i < mirrored.length; i += 3) {
    mirrored[i] = -mirrored[i];
  }
  return mirrored;
}
