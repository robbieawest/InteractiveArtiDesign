import * as THREE from "three";
import CameraControls from "camera-controls";
import { createInfiniteGrid } from "./InfiniteGrid";

CameraControls.install({ THREE });

/**
 * Owns the WebGL renderer, camera, scene and camera controls.
 *
 * Rendering is on-demand: nothing is drawn unless `invalidate()` was called
 * or the camera moved. A single requestAnimationFrame loop checks both, so
 * the rest of the app never calls `renderer.render` directly.
 *
 * Camera bindings (left button is deliberately unbound — it belongs to
 * whichever tool is active):
 *   - wheel          → dolly (zoom)
 *   - right drag     → dolly (zoom)
 *   - middle drag    → pan
 *   - space + drag   → pan
 *   - alt + drag     → orbit
 */
export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera: THREE.PerspectiveCamera;
  readonly scene: THREE.Scene;
  readonly controls: CameraControls;
  /** Drawing-buffer size in pixels; shared with stroke shaders as a uniform. */
  readonly resolution = new THREE.Vector2(1, 1);

  private readonly container: HTMLElement;
  private readonly clock = new THREE.Clock();
  private readonly resizeObserver: ResizeObserver;
  private needsRender = true;
  private rafHandle = 0;
  private disposed = false;

  // Camera-interaction tracking, derived from raw input rather than
  // camera-controls events (those fire "controlstart" even for unbound
  // buttons and only report rest after damping settles).
  private cameraPointerActive = false;
  private wheelActive = false;
  private wheelTimer: ReturnType<typeof setTimeout> | undefined;
  private cameraActivityReported = false;
  private readonly cameraActivityListeners = new Set<(active: boolean) => void>();

  constructor(container: HTMLElement) {
    this.container = container;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    const backgroundColor = 0xdddddd;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(backgroundColor);
    this.scene.fog = new THREE.Fog(backgroundColor, 10, 25);

    this.camera = new THREE.PerspectiveCamera(
      100,
      container.clientWidth / Math.max(container.clientHeight, 1),
      0.1,
      1000,
    );
    this.camera.position.set(5, 5, 10);
    this.camera.zoom = 3;

    this.scene.add(createInfiniteGrid());

    const light = new THREE.HemisphereLight(0xffffff, 0x222222, 0.8);
    light.position.set(0, 50, 150);
    this.scene.add(light);

    this.controls = new CameraControls(this.camera, this.renderer.domElement);
    this.controls.dampingFactor = 20;
    this.controls.draggingDampingFactor = 200;
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 100;
    this.controls.mouseButtons.left = CameraControls.ACTION.NONE;
    this.controls.mouseButtons.wheel = CameraControls.ACTION.DOLLY;
    this.controls.mouseButtons.middle = CameraControls.ACTION.TRUCK;
    this.controls.mouseButtons.right = CameraControls.ACTION.DOLLY;

    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);

    this.renderer.domElement.addEventListener("pointerdown", this.onCameraPointerDown);
    window.addEventListener("pointerup", this.onCameraPointerUp);
    window.addEventListener("pointercancel", this.onCameraPointerUp);
    this.renderer.domElement.addEventListener("wheel", this.onCameraWheel, {
      passive: true,
    });

    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(container);
    this.onResize();

    this.rafHandle = requestAnimationFrame(this.frame);
  }

  /** Request a redraw on the next animation frame. */
  invalidate(): void {
    this.needsRender = true;
  }

  /** Downscaled snapshot of the current view (JPEG data URL). Renders
   *  synchronously first so the drawing buffer is fresh to read. */
  captureThumbnail(width = 192): string {
    this.renderer.render(this.scene, this.camera);
    const source = this.renderer.domElement;
    const height = Math.max(
      1,
      Math.round((width * source.height) / Math.max(source.width, 1)),
    );
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d")!.drawImage(source, 0, 0, width, height);
    return canvas.toDataURL("image/jpeg", 0.75);
  }

  /** True while a temporary key (space/alt) reroutes the left mouse button
   *  to the camera; tools should ignore pointer input in that state. */
  get cameraOwnsPointer(): boolean {
    return this.controls.mouseButtons.left !== CameraControls.ACTION.NONE;
  }

  /** Notifies when the user starts/stops driving the camera (drag with a
   *  camera button, or wheel — the latter settles after a short pause).
   *  Returns an unsubscribe function. */
  onCameraActivity(listener: (active: boolean) => void): () => void {
    this.cameraActivityListeners.add(listener);
    return () => this.cameraActivityListeners.delete(listener);
  }

  private emitCameraActivity(): void {
    const active = this.cameraPointerActive || this.wheelActive;
    if (active === this.cameraActivityReported) return;
    this.cameraActivityReported = active;
    for (const listener of this.cameraActivityListeners) listener(active);
  }

  private onCameraPointerDown = (event: PointerEvent): void => {
    const cameraButton =
      event.button === 1 ||
      event.button === 2 ||
      (event.button === 0 && this.cameraOwnsPointer);
    if (cameraButton) {
      this.cameraPointerActive = true;
      this.emitCameraActivity();
    }
  };

  private onCameraPointerUp = (): void => {
    if (!this.cameraPointerActive) return;
    this.cameraPointerActive = false;
    this.emitCameraActivity();
  };

  private onCameraWheel = (): void => {
    this.wheelActive = true;
    this.emitCameraActivity();
    clearTimeout(this.wheelTimer);
    this.wheelTimer = setTimeout(() => {
      this.wheelActive = false;
      this.emitCameraActivity();
    }, 250);
  };

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.rafHandle);
    clearTimeout(this.wheelTimer);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    this.renderer.domElement.removeEventListener(
      "pointerdown",
      this.onCameraPointerDown,
    );
    window.removeEventListener("pointerup", this.onCameraPointerUp);
    window.removeEventListener("pointercancel", this.onCameraPointerUp);
    this.renderer.domElement.removeEventListener("wheel", this.onCameraWheel);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private frame = (): void => {
    if (this.disposed) return;

    const cameraMoved = this.controls.update(this.clock.getDelta());
    if (cameraMoved || this.needsRender) {
      this.needsRender = false;
      this.renderer.render(this.scene, this.camera);
    }
    this.rafHandle = requestAnimationFrame(this.frame);
  };

  private onResize(): void {
    const width = this.container.clientWidth;
    const height = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.renderer.getDrawingBufferSize(this.resolution);
    this.invalidate();
  }

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.repeat) return;
    if (event.code === "Space") {
      this.controls.mouseButtons.left = CameraControls.ACTION.TRUCK;
    } else if (event.code === "AltLeft") {
      event.preventDefault(); // keep the browser from focusing its menu bar
      this.controls.mouseButtons.left = CameraControls.ACTION.ROTATE;
    }
  };

  private onKeyUp = (event: KeyboardEvent): void => {
    if (event.code === "Space" || event.code === "AltLeft") {
      this.controls.mouseButtons.left = CameraControls.ACTION.NONE;
    }
  };
}
