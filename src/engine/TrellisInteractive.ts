import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { Viewport } from "./Viewport";
import { buildStrokes, disposeSketchContent } from "./sketchRender";
import type { SketchRenderStyle } from "./sketchRender";
import { DEFAULT_VOLUME_STYLE, VolumeGrid } from "./VolumeGrid";
import type { VolumeStyle } from "./VolumeGrid";
import type { SurfacingSketch } from "../surfacing/client";
import type { FlowFrames } from "../surfacing/trellisFrames";
import { stageLengths } from "../surfacing/trellisFrames";

/**
 * The interactive TRELLIS viewer: one run laid out as regions of the scene,
 * scrubbable step by step.
 *
 * Five regions along X, all in the same scene under one camera, because the
 * point is to look between them — the conditioning views the model actually
 * saw, the sketch they came from, the occupancy grid the structure stage
 * sampled, where the latent stage was still moving, and the mesh that came
 * out. Everything is normalized into a slot of the same size: the sketch's
 * own units and placement carry no information here, and matching apparent
 * sizes is what makes the regions comparable.
 *
 * This is a derived view, like SurfacePreview — it renders a finished run and
 * owns none of it. Nothing here touches the document, and the whole layout is
 * built and thrown away as a unit.
 */

/** How the sketch copies are drawn inside the regions. Tubes rather than
 *  hairlines so they read against the volume, which is the thing they have
 *  to be legible *through*. */
const SKETCH_STYLE: SketchRenderStyle = {
  size: 1,
  strokeColor: 0x222222,
  strokeThickness: 0.006,
  surfaceColor: 0xffaa3c,
  surfaceOpacity: 1,
  margin: 1,
  headlight: false,
  ambientIntensity: 0.6,
  keyIntensity: 1.0,
};

/** Edge of the box each region's content is scaled to fit. */
const REGION_SIZE = 1;
/** Space between regions, as a fraction of one region. */
const REGION_GAP = 0.4;
/** Views stack four to a column before starting the next, so a 12-view run
 *  is three columns — the arrangement the run configurations actually use. */
const VIEWS_PER_COLUMN = 4;
/** The strokes drawn over the result: the surface overlay's own colour, so
 *  the two readings of "this is the sketch" agree across the app. */
const OVERLAY_COLOR = 0xffaa3c;

export interface TrellisRun {
  sketch: SurfacingSketch;
  /** Conditioning views as data URLs, in the order the model consumed them. */
  views: string[];
  frames: FlowFrames | null;
  /** The delivered mesh, and the same mesh before simplification and
   *  fragment removal — when the run was asked to keep it. */
  processedGlb: ArrayBuffer | null;
  rawGlb: ArrayBuffer | null;
}

export interface TrellisViewState {
  /** Position on the concatenated timeline: structure steps, then latent. */
  position: number;
  /** Show the unprocessed mesh instead of the delivered one. */
  showRaw: boolean;
  /** Draw the strokes over the result region, to compare them by eye. */
  showSketchOverlay: boolean;
  volume: Partial<VolumeStyle>;
}

export class TrellisInteractiveView {
  private readonly viewport: Viewport;
  private readonly loader = new GLTFLoader();
  private readonly root = new THREE.Group();

  private structure: VolumeGrid | null = null;
  private latent: VolumeGrid | null = null;
  private processedMesh: THREE.Object3D | null = null;
  private rawMesh: THREE.Object3D | null = null;
  private frames: FlowFrames | null = null;
  private sketchCopies: THREE.Object3D[] = [];
  private labels: THREE.Sprite[] = [];
  private viewQuads: THREE.Object3D | null = null;
  private sketchOverlay: THREE.Object3D | null = null;
  private active = false;

  constructor(viewport: Viewport) {
    this.viewport = viewport;
    this.root.visible = false;
    this.viewport.scene.add(this.root);
  }

  /** Total scrubbable positions. 0 when this run captured no frames. */
  get timelineLength(): number {
    return this.frames ? stageLengths(this.frames).total : 0;
  }

  /** How many of those belong to the structure stage. */
  get structureSteps(): number {
    return this.frames ? stageLengths(this.frames).structure : 0;
  }

  get isActive(): boolean {
    return this.active;
  }

  /** Build the layout for one finished run. Replaces anything showing. */
  async show(run: TrellisRun): Promise<void> {
    this.clear();
    this.frames = run.frames;

    const regions: { label: string; content: THREE.Object3D }[] = [];

    const views = this.buildViews(run.views);
    if (views) regions.push({ label: "conditioning views", content: views });

    regions.push({ label: "sketch", content: this.sketchCopy(run.sketch) });

    if (run.frames) {
      // occupancy: the field in blue, and red where it crosses the cut the
      // pipeline itself makes
      this.structure = new VolumeGrid();
      // convergence: one continuous quantity with no threshold in it, so a
      // plain ramp in a colour that cannot be mistaken for occupancy
      this.latent = new VolumeGrid({
        ...DEFAULT_VOLUME_STYLE,
        splitAtThreshold: false,
        hazeColor: new THREE.Color(0x7a4fd0),
      });
      regions.push({
        label: "structure flow — occupancy (red = above threshold)",
        content: this.buildVolumeRegion(this.structure, run),
      });
      regions.push({
        label: "latent flow — distance from final",
        content: this.buildVolumeRegion(this.latent, run),
      });
    }

    this.processedMesh = await this.loadMesh(run.processedGlb);
    this.rawMesh = await this.loadMesh(run.rawGlb);
    const meshRegion = new THREE.Group();
    if (this.processedMesh) meshRegion.add(this.processedMesh);
    if (this.rawMesh) meshRegion.add(this.rawMesh);
    if (this.processedMesh || this.rawMesh) {
      // The strokes over the result, for reading off by eye where the model
      // added volume the drawing never had and where it missed the drawing
      // entirely. Built now rather than on demand so it is inside the box
      // `fitInto` measures — appearing later must not reframe the region.
      // Only meaningful once the mesh has been registered onto the strokes;
      // without a fit the two are in unrelated frames.
      if (run.frames?.align) {
        this.sketchOverlay = this.sketchCopy(run.sketch, OVERLAY_COLOR);
        this.sketchOverlay.visible = false;
        meshRegion.add(this.sketchOverlay);
      }
      regions.push({ label: "result", content: meshRegion });
    }

    // Normalization happens per region *before* placement, so a sketch drawn
    // huge and a unit-cube lattice end up the same apparent size. Inside a
    // region the relative placement is preserved — that is the part that
    // carries meaning.
    const span = REGION_SIZE * (1 + REGION_GAP);
    const offset = ((regions.length - 1) * span) / 2;
    regions.forEach((region, index) => {
      const slot = fitInto(region.content, REGION_SIZE);
      slot.position.x = index * span - offset;
      this.root.add(slot);
      const label = makeLabel(region.label);
      label.position.set(slot.position.x, REGION_SIZE * 0.75, 0);
      this.labels.push(label);
      this.root.add(label);
    });

    this.root.visible = true;
    this.active = true;
    this.setState({
      position: 0,
      showRaw: false,
      showSketchOverlay: false,
      volume: {},
    });
    this.frameCamera();
  }

  /** Move the scrubber and apply the view toggles. */
  setState(state: Partial<TrellisViewState>): void {
    if (state.volume) {
      this.structure?.setStyle(state.volume);
      // the threshold is a fact about occupancy — the latent region measures
      // distance from the final latent, which has no such cut in it
      const { threshold: _ignored, ...shared } = state.volume;
      this.latent?.setStyle(shared);
    }
    if (state.showSketchOverlay !== undefined && this.sketchOverlay) {
      this.sketchOverlay.visible = state.showSketchOverlay;
    }
    if (state.showRaw !== undefined) {
      // falls back to whichever mesh this run actually has, so the toggle
      // can never blank the region
      const useRaw = (state.showRaw || !this.processedMesh) && !!this.rawMesh;
      if (this.rawMesh) this.rawMesh.visible = useRaw;
      if (this.processedMesh) this.processedMesh.visible = !useRaw;
    }
    if (state.position !== undefined && this.frames) {
      const { structure, latent } = stageLengths(this.frames);
      const position = Math.max(0, Math.min(this.timelineLength - 1, state.position));

      // The two stages get a region each rather than sharing one that swaps
      // content mid-scrub. Each holds its own last state once its stage is
      // over, so the occupancy stays on screen while the latent stage runs —
      // which is true of the pipeline too: SLAT never changes the voxel set.
      const structureStep = Math.min(position, structure - 1);
      const structureFrame = this.frames.stages.structure[structureStep];
      if (structureFrame && this.structure) {
        this.structure.setVolume(structureFrame, this.frames.grid);
      }

      const latentStep = position - structure;
      if (this.latent) {
        const showLatent = latentStep >= 0 && latent > 0;
        this.latent.setVisible(showLatent);
        const frame = this.frames.stages.latent[Math.max(0, Math.min(latentStep, latent - 1))];
        if (showLatent && frame) this.latent.setVolume(frame, this.frames.grid);
      }
    }
    this.viewport.invalidate();
  }

  /** Point the camera at the whole layout. */
  frameCamera(): void {
    const box = new THREE.Box3().setFromObject(this.root);
    if (box.isEmpty()) return;
    void this.viewport.controls.fitToBox(box, true, {
      paddingTop: 0.1,
      paddingBottom: 0.1,
      paddingLeft: 0.1,
      paddingRight: 0.1,
    });
    this.viewport.invalidate();
  }

  /** Tear the layout down and free everything it holds. */
  clear(): void {
    // order matters: the volumes own their own geometry and textures and
    // dispose themselves, so they come out before the blanket sweep
    this.structure?.dispose();
    this.latent?.dispose();
    this.structure = null;
    this.latent = null;

    for (const copy of this.sketchCopies) disposeSketchContent(copy);
    this.sketchCopies = [];
    for (const mesh of [this.processedMesh, this.rawMesh]) {
      if (mesh) disposeSketchContent(mesh);
    }
    this.processedMesh = null;
    this.rawMesh = null;
    this.sketchOverlay = null; // freed with the rest of the sketch copies

    if (this.viewQuads) {
      disposeQuads(this.viewQuads);
      this.viewQuads = null;
    }
    for (const label of this.labels) {
      label.material.map?.dispose();
      label.material.dispose();
    }
    this.labels = [];

    this.frames = null;
    for (const child of [...this.root.children]) this.root.remove(child);
    this.root.visible = false;
    this.active = false;
    this.viewport.invalidate();
  }

  dispose(): void {
    this.clear();
    this.viewport.scene.remove(this.root);
  }

  /** The lattice over a copy of the sketch, in one group so `fitInto` scales
   *  them together and their relative placement survives. */
  private buildVolumeRegion(volume: VolumeGrid, run: TrellisRun): THREE.Object3D {
    const group = new THREE.Group();
    const align = run.frames?.align ?? null;

    if (align) {
      // world = scale * rotation * v + translation, and the box geometry is
      // already the unit cube the generation happened in
      const rotation = new THREE.Matrix4().set(
        align.rotation[0], align.rotation[1], align.rotation[2], 0,
        align.rotation[3], align.rotation[4], align.rotation[5], 0,
        align.rotation[6], align.rotation[7], align.rotation[8], 0,
        0, 0, 0, 1,
      );
      volume.object.quaternion.setFromRotationMatrix(rotation);
      volume.object.scale.setScalar(align.scale);
      volume.object.position.fromArray(align.translation);

      group.add(this.sketchCopy(run.sketch));
    }
    // Without a fit there is nothing that says where the lattice sits
    // relative to the drawing, so the drawing is left out rather than
    // overlaid at an invented pose.

    group.add(volume.object);
    return group;
  }

  /** A standalone copy of the strokes, tracked so `clear` can free it. */
  private sketchCopy(sketch: SurfacingSketch, color?: number): THREE.Object3D {
    const copy = buildStrokes(
      sketch,
      color === undefined ? SKETCH_STYLE : { ...SKETCH_STYLE, strokeColor: color },
    );
    this.sketchCopies.push(copy);
    return copy;
  }

  private buildViews(urls: string[]): THREE.Object3D | null {
    if (urls.length === 0) return null;
    const group = new THREE.Group();
    this.viewQuads = group;
    const loader = new THREE.TextureLoader();
    const cell = 1;
    const gap = 0.08;

    urls.forEach((url, index) => {
      const column = Math.floor(index / VIEWS_PER_COLUMN);
      const row = index % VIEWS_PER_COLUMN;
      const texture = loader.load(url, () => this.viewport.invalidate());
      texture.colorSpace = THREE.SRGBColorSpace;
      const quad = new THREE.Mesh(
        new THREE.PlaneGeometry(cell, cell),
        new THREE.MeshBasicMaterial({
          map: texture,
          transparent: true,
          // the camera orbits the whole layout, so half the time these are
          // seen from behind; edge-on is accepted, invisible is not
          side: THREE.DoubleSide,
          toneMapped: false,
        }),
      );
      quad.position.set(column * (cell + gap), -row * (cell + gap), 0);
      group.add(quad);
    });
    return group;
  }

  private async loadMesh(glb: ArrayBuffer | null): Promise<THREE.Object3D | null> {
    if (!glb) return null;
    // parseAsync detaches its input; the caller keeps the bundle for redraws
    const gltf = await this.loader.parseAsync(glb.slice(0), "");
    gltf.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.material = new THREE.MeshStandardMaterial({
        color: 0xffaa3c,
        roughness: 0.6,
        metalness: 0,
        side: THREE.DoubleSide,
      });
    });
    return gltf.scene;
  }
}

/** Centre `object` on the origin and scale it to fit a box of `size`. */
function fitInto(object: THREE.Object3D, size: number): THREE.Group {
  const wrapper = new THREE.Group();
  wrapper.add(object);
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return wrapper;
  const extent = box.getSize(new THREE.Vector3());
  const scale = size / Math.max(extent.x, extent.y, extent.z, 1e-6);
  const center = box.getCenter(new THREE.Vector3());
  object.scale.multiplyScalar(scale);
  object.position.copy(center).multiplyScalar(-scale);
  return wrapper;
}

/** A region caption as a camera-facing sprite. Five unlabelled regions is a
 *  puzzle; a text mesh would mean shipping a font. */
function makeLabel(text: string): THREE.Sprite {
  const scale = 4;
  const canvas = document.createElement("canvas");
  canvas.width = 512 * scale;
  canvas.height = 64 * scale;
  const context = canvas.getContext("2d");
  if (context) {
    context.scale(scale, scale);
    context.font = "500 34px system-ui, sans-serif";
    context.fillStyle = "#333";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, 256, 32);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }),
  );
  sprite.scale.set(1.0, 0.125, 1);
  sprite.renderOrder = 20;
  return sprite;
}

/** Free the plane geometries and textures the view quads hold. */
function disposeQuads(root: THREE.Object3D): void {
  root.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry?.dispose();
    const material = mesh.material as THREE.MeshBasicMaterial;
    material?.map?.dispose();
    material?.dispose();
  });
}
