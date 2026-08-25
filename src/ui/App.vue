<template>
  <div ref="viewportEl" class="viewport"></div>

  <div class="toolbar left">
    <button
      :class="{ active: activeTool === 'draw' }"
      title="Draw (D) — left-drag on the canvas surface to sketch"
      @click="toggleTool('draw')"
    >
      Draw
    </button>
    <button
      :class="{ active: activeTool === 'erase' }"
      title="Erase (E) — left-drag over strokes to delete them"
      @click="toggleTool('erase')"
    >
      Erase
    </button>
    <button
      :class="{ active: activeTool === 'select' }"
      title="Select (V) — click a stroke, double-click a part; T/R/S set the gizmo, Delete removes, Esc deselects"
      @click="toggleTool('select')"
    >
      Select
    </button>
    <button
      :class="{ active: activeTool === 'segment' }"
      title="Segment (G) — drag a pen through strokes to add/remove them from the active part (see Parts panel)"
      @click="toggleTool('segment')"
    >
      Segment
    </button>
    <button
      :class="{ active: activeTool === 'articulate' }"
      title="Articulate (A) — click a part to drive its joint along its axis; toggle IK in the Articulations panel to drag parts freely (collapse the exploded view first)"
      @click="toggleTool('articulate')"
    >
      Articulate
    </button>
    <button
      :class="{ active: activeTool === 'joint' }"
      title="Joint (J) — drag from a parent part to a child part to create a joint; click a part to edit the joint that drives it"
      @click="toggleTool('joint')"
    >
      Joint
    </button>

    <span class="divider" />

    <template v-if="activeTool === 'draw'">
      <select
        v-model="shape"
        title="Canvas shape — the surface strokes are drawn onto"
      >
        <option value="plane">Plane</option>
        <option value="cube">Cube</option>
        <option value="cylinder">Cylinder</option>
        <option value="sphere">Sphere</option>
      </select>
      <button
        title="Gizmo mode — press T (translate), R (rotate) or S (scale), or click to cycle"
        @click="cycleGizmoMode"
      >
        {{ gizmoMode }}
      </button>
      <button
        title="Show or hide the canvas surface (drawing still works while hidden)"
        @click="toggleCanvasVisible"
      >
        {{ canvasVisible ? "Hide canvas" : "Show canvas" }}
      </button>
      <button
        title="Reset the canvas surface to the origin (position, rotation and scale)"
        @click="surface?.resetTransform()"
      >
        Reset canvas
      </button>
      <select
        v-model="mirror"
        title="Mirror — also draw a twin stroke reflected across a world axis"
      >
        <option value="off">No mirror</option>
        <option value="x">Mirror X</option>
        <option value="y">Mirror Y</option>
        <option value="z">Mirror Z</option>
      </select>

      <span class="divider" />

      <input v-model="strokeColor" type="color" title="Stroke color" />
      <input
        v-model.number="strokeWidth"
        type="range"
        min="1"
        max="30"
        title="Stroke width"
      />
      <label title="Fill the stroke's outline with a color">
        <input v-model="fillVisible" type="checkbox" /> Fill
      </label>
      <input
        v-if="fillVisible"
        v-model="fillColor"
        type="color"
        title="Fill color"
      />
      <button
        v-if="selectedPart"
        class="part-chip"
        title="New strokes join this part automatically (they preview purple). Click or press Esc to stop."
        @click="selectTool?.deselect()"
      >
        Drawing into {{ doc.getPart(selectedPart)?.name ?? "part" }} ✕
      </button>
    </template>
    <template v-else-if="activeTool === 'select'">
      <button
        title="Gizmo mode — press T (translate), R (rotate) or S (scale), or click to cycle"
        @click="cycleGizmoMode"
      >
        {{ gizmoMode }}
      </button>
      <button
        title="Deselect everything (Esc) — tip: ctrl+click adds/removes strokes from the selection, double-click selects a whole part"
        @click="selectTool?.deselect()"
      >
        Deselect
      </button>
    </template>
    <template v-else-if="activeTool === 'joint'">
      <span class="hint">{{ jointHint }}</span>
    </template>
    <template v-else-if="activeTool === 'none' && explodeMode">
      <span class="hint">
        Drag away from the model's center to explode it; other tools work on
        the exploded model — turn explode off in the Parts panel to restore
        the original pose
      </span>
    </template>
    <template v-else-if="activeTool === 'segment'">
      <select
        v-model="segmentMode"
        title="Whether the pen adds strokes to the active part or removes them from it"
      >
        <option value="add">Pen adds</option>
        <option value="remove">Pen removes</option>
      </select>
      <span class="hint">
        {{ activePart ? "Painting: " + activePart.name : "No active part" }}
      </span>
    </template>
  </div>

  <div class="toolbar right">
    <button
      :disabled="!canUndo"
      title="Undo (Ctrl+Z)"
      @click="undoStack.undo()"
    >
      Undo
    </button>
    <button
      :disabled="!canRedo"
      title="Redo (Ctrl+Shift+Z or Ctrl+Y)"
      @click="undoStack.redo()"
    >
      Redo
    </button>
    <button
      title="Show or hide all sketch strokes (drawing and selection still work)"
      @click="toggleSketchVisible"
    >
      {{ sketchVisible ? "Hide Sketch" : "Show Sketch" }}
    </button>
    <button
      :disabled="!hasSurface"
      title="Show or hide the surfacing result mesh"
      @click="toggleSurfaceVisible"
    >
      {{ surfaceVisible ? "Hide Surface" : "Show Surface" }}
    </button>
    <span class="divider" />
    <button title="Save the sketch as a JSON file" @click="saveFile">
      Save
    </button>
    <button
      title="Load a sketch (this app's JSON or a legacy Penzil file)"
      @click="loadFile"
    >
      Load
    </button>
    <button
      title="Export strokes as solid 3D tubes in a .glb file (Blender etc.)"
      @click="exportFile"
    >
      Export GLB
    </button>
    <button
      title="Delete all strokes and start over"
      @click="resetDialogOpen = true"
    >
      Reset strokes
    </button>
  </div>

  <bottom-panels
    :expanded="expandedPanel"
    :parts="partsList"
    :poses="posesList"
    :joints="jointsList"
    :ik="ikEnabled"
    :selected-joint-id="selectedJointId"
    :editing-joint-id="editingJointId"
    :armed-dof="armedDof"
    :mirror="mirrorRange"
    :active-part-id="activePartId"
    :stroke-counts="strokeCounts"
    :exploding="explodeMode"
    :methods="surfacingMethods"
    :surfacing-method="surfacingMethod"
    :method-params="currentMethodParams"
    :surfacing-options="currentOptions"
    :surfacing="surfacingBusy"
    :surfacing-progress="surfacingProgress"
    :surfacing-message="surfacingMessage"
    :has-surface="hasSurface"
    :surface-color="surfaceColor"
    :surface-opacity="surfaceOpacity"
    :surfacing-log="surfacingLog"
    @set-expanded="expandedPanel = $event"
    @set-active-part="setActivePart"
    @add-part="addPart"
    @remove-part="removePart"
    @toggle-explode="toggleExplode"
    @save-pose="savePose"
    @apply-pose="applyPose"
    @remove-pose="doc.removePose($event)"
    @toggle-ik="ikEnabled = !ikEnabled"
    @reset-articulation="resetArticulation"
    @select-joint="selectJoint"
    @edit-joint="editJoint"
    @delete-joint="deleteJoint"
    @new-joint="newJoint"
    @arm-dof="jointTool?.armDof($event)"
    @toggle-mirror="jointTool?.setMirror(!mirrorRange)"
    @set-surfacing-method="surfacingMethod = $event"
    @set-surfacing-option="setSurfacingOption"
    @reset-surfacing-options="resetSurfacingOptions"
    @set-surface-color="setSurfaceColor"
    @set-surface-opacity="setSurfaceOpacity"
    @surface="runSurfacing"
    @clear-surface="clearSurface"
    @open-benchmark="benchmarkOpen = true"
  />

  <FlowTimeline
    v-if="flowActive || flowHasRaw"
    :length="flowLength"
    :structure-steps="flowStructureSteps"
    :step-times="flowStepTimes"
    :position="flowPosition"
    :threshold="flowThreshold"
    :density="flowDensity"
    :show-raw="flowShowRaw"
    :has-raw="flowHasRaw"
    :has-frames="flowActive"
    :show-sketch-overlay="flowShowSketchOverlay"
    :can-overlay-sketch="flowCanOverlaySketch"
    :view-count="flowViewCount"
    @set-position="flowPosition = $event; applyFlowState()"
    @set-threshold="flowThreshold = $event; applyFlowState()"
    @set-density="flowDensity = $event; applyFlowState()"
    @set-show-raw="setFlowShowRaw($event)"
    @set-sketch-overlay="flowShowSketchOverlay = $event; applyFlowState()"
    @close="clearSurface"
  />

  <OccupancyControls
    v-if="occupancyActive"
    :count="occupancyCount"
    :grid="occupancyGrid"
    :max="occupancyMax"
    :threshold="occupancyThreshold"
    :density="occupancyDensity"
    :blur="occupancyBlur"
    @set-threshold="occupancyThreshold = $event; applyOccupancyStyle()"
    @set-density="occupancyDensity = $event; applyOccupancyStyle()"
    @set-blur="occupancyBlur = $event; applyOccupancyBlur()"
    @close="clearSurface"
  />

  <BenchmarkWindow
    v-if="benchmarkOpen"
    @close="benchmarkOpen = false"
    @open="openBenchmarkSketch"
  />

  <!-- an edit session borrows the whole editor, so it needs a way back -->
  <div v-if="editingBenchmarkSketch" class="benchmark-edit-bar">
    <span>
      Editing benchmark sketch <strong>{{ editingBenchmarkSketch }}</strong>
    </span>
    <button
      title="Save the edited sketch back into the benchmark, in its current pose. Any surfaces already made from it are discarded, since they no longer match."
      @click="saveBenchmarkEdit"
    >
      Save to benchmark
    </button>
    <button title="Leave the benchmark's copy untouched" @click="cancelBenchmarkEdit">
      Discard
    </button>
  </div>

  <div
    v-if="resetDialogOpen"
    class="modal-backdrop"
    @click.self="resetDialogOpen = false"
  >
    <div class="modal">
      <p>
        This deletes every stroke and cannot be undone. Save your sketch
        first?
      </p>
      <div class="modal-buttons">
        <button @click="saveThenReset">Save</button>
        <button class="danger" @click="resetStrokes">
          Reset without saving
        </button>
        <button @click="resetDialogOpen = false">Cancel</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, useTemplateRef, watch } from "vue";
import { Viewport } from "../engine/Viewport";
import { StrokeRenderer } from "../engine/StrokeRenderer";
import { JointLines } from "../engine/JointLines";
import { CanvasSurface, type GizmoMode } from "../engine/CanvasSurface";
import { exportGlb } from "../engine/exportGlb";
import { DrawTool, type MirrorAxis } from "../tools/DrawTool";
import { EraseTool } from "../tools/EraseTool";
import { SelectTool } from "../tools/SelectTool";
import { SegmentTool, type SegmentMode } from "../tools/SegmentTool";
import { ArticulateTool } from "../tools/ArticulateTool";
import { JointTool } from "../tools/JointTool";
import { ExplodeTool } from "../tools/ExplodeTool";
import BottomPanels, { type PanelName } from "./BottomPanels.vue";
import BenchmarkWindow from "./BenchmarkWindow.vue";
import * as benchmark from "../benchmark/store";
import { SketchDocument } from "../core/SketchDocument";
import {
  UndoStack,
  applyPoseCommand,
  collapseCommand,
  resetArticulationCommand,
} from "../core/undo";
import {
  deserializeDocument,
  serializeDocument,
  type DocumentJson,
} from "../core/serialization";
import { importLegacyPenzil, isLegacyPenzilJson } from "../core/legacyPenzil";
import { importSketchLabGltf, parseGlb } from "../engine/importSketchLab";
import { SurfacePreview } from "../engine/SurfacePreview";
import {
  buildSurfacingSketch,
  fetchMethods,
  runSurfacingJob,
  viewSpecFor,
  type MethodInfo,
  type MethodOptions,
} from "../surfacing/client";
import {
  renderConditioningViews,
  renderSurfacedViews,
} from "../engine/strokeViews";
import { TrellisInteractiveView } from "../engine/TrellisInteractive";
import { decodeFlowFrames, type FlowFrames } from "../surfacing/trellisFrames";
import FlowTimeline from "./trellis-interactive/FlowTimeline.vue";
import {
  DEFAULT_OCCUPANCY_THRESHOLD,
  OccupancyFieldView,
} from "../engine/OccupancyField";
import {
  decodeOccupancyVolume,
  type OccupancyVolume,
} from "../surfacing/ns2sVolume";
import OccupancyControls from "./occupancy-field/OccupancyControls.vue";
import type { JointDofName, SurfaceShape } from "../core/types";
import { jointPosed } from "../core/types";

/** "none" is the default idle state: no tool listens to the pointer —
 *  except in explode mode, where idle dragging adjusts the explosion. */
type ToolName =
  | "none"
  | "draw"
  | "erase"
  | "select"
  | "segment"
  | "articulate"
  | "joint";

const viewportEl = useTemplateRef("viewportEl");
const doc = new SketchDocument();
const undoStack = new UndoStack();

let viewport: Viewport | undefined;
let strokeRenderer: StrokeRenderer | undefined;
let surface: CanvasSurface | undefined;
interface ToolHandler {
  attach(): void;
  detach(): void;
}
let tools: Record<Exclude<ToolName, "none">, ToolHandler> | undefined;
/** The handler currently receiving pointer events (null in the idle state). */
let attachedHandler: ToolHandler | null = null;
let explodeTool: ExplodeTool | undefined;
let drawTool: DrawTool | undefined;
let selectTool: SelectTool | undefined;
let segmentTool: SegmentTool | undefined;
let articulateTool: ArticulateTool | undefined;
let jointTool: JointTool | undefined;
let jointLines: JointLines | undefined;
let surfacePreview: SurfacePreview | undefined;

const activeTool = ref<ToolName>("none");
/** Explode mode persists across tool switches so you can draw, select, etc.
 *  on the exploded model; turning it off restores the original pose. */
const explodeMode = ref(false);
const shape = ref<SurfaceShape>("plane");
const gizmoMode = ref<GizmoMode>("translate");
const canvasVisible = ref(true);
const mirror = ref<MirrorAxis>("off");
const strokeColor = ref("#1c1c1e");
const strokeWidth = ref(8);
const fillVisible = ref(false);
const fillColor = ref("#1c1c1e");
const canUndo = ref(false);
const canRedo = ref(false);
const resetDialogOpen = ref(false);
// the window is a view onto the benchmark store; closing it stops nothing
const benchmarkOpen = ref(false);
const editingBenchmarkSketch = computed(() => benchmark.state.editing);

// surfacing job state (the mesh itself lives in SurfacePreview, not here)
const surfacingBusy = ref(false);
const surfacingProgress = ref(0);
// The adapter's own label for what it is doing right now. Distinct from
// the log: it is one line that keeps being replaced, and for the flow
// samplers it carries the step's t range, which the percentage cannot —
// `rescale_t` makes the steps wildly non-uniform.
const surfacingMessage = ref("");
const hasSurface = ref(false);
const surfaceVisible = ref(true);
const sketchVisible = ref(true);
/** Surface overlay appearance, edited from the Surfacer panel. */
const surfaceColor = ref("#ffaa3c");
const surfaceOpacity = ref(0.55);
/** Methods from the server; empty while it is unreachable. */
const surfacingMethods = ref<MethodInfo[]>([]);
const surfacingMethod = ref("");
/** Edited parameter values, kept per method so switching loses nothing. */
const surfacingOptions = ref<Record<string, MethodOptions>>({});
const currentMethodParams = computed(
  () =>
    surfacingMethods.value.find((m) => m.name === surfacingMethod.value)
      ?.params ?? [],
);
const currentOptions = computed(
  () => surfacingOptions.value[surfacingMethod.value] ?? {},
);
const surfacingLog = ref<string[]>([]);
const SURFACING_LOG_CAP = 1000;

// --- interactive TRELLIS flow view ---
//
// A run asked to record its flow takes over the viewport when it finishes:
// the document's strokes and the ordinary surface overlay go out of sight and
// a laid-out copy of the run goes up in their place. Nothing here is part of
// the document — the frames are never serialized and never written to disk,
// so leaving the view (or any of the paths that already clear a surface)
// drops them for good and the run would have to be repeated.
let flowView: TrellisInteractiveView | undefined;
const flowActive = ref(false);
const flowLength = ref(0);
const flowStructureSteps = ref(0);
// The `t` behind each scrub position, when the capture recorded it.
const flowStepTimes = ref<number[]>([]);
const flowPosition = ref(0);
const flowThreshold = ref(0.5);
const flowDensity = ref(1);
const flowShowRaw = ref(false);
const flowHasRaw = ref(false);
const flowShowSketchOverlay = ref(false);
const flowCanOverlaySketch = ref(false);
const flowViewCount = ref(0);

// --- raymarched occupancy field (NS2S probability volume) ---
//
// The lighter sibling of the flow view: one field, sitting over the sketch it
// was predicted from, with the document left exactly as it is. Same lifetime
// though — never serialized, never written, dropped by anything that clears a
// surface.
let occupancyView: OccupancyFieldView | undefined;
const occupancyActive = ref(false);
const occupancyCount = ref(0);
const occupancyGrid = ref(0);
const occupancyMax = ref(0);
const occupancyThreshold = ref(DEFAULT_OCCUPANCY_THRESHOLD);
const occupancyDensity = ref(1);
const occupancyBlur = ref(0);
/** The shown surface's glb + producing method, kept for embedding in saves. */
let surfaceGlb: { method: string; bytes: ArrayBuffer } | null = null;
/** The same result before postprocessing, when the run kept it. Only ever in
 *  memory: `surfaceGlb` is what rides along in saves, and it is always the
 *  delivered mesh — the raw one is a diagnostic, not a deliverable. */
let surfaceRawGlb: ArrayBuffer | null = null;

// parts / poses / panels
const expandedPanel = ref<PanelName | null>(null);
const activePartId = ref<string | null>(null);
/** The part selected via double-click in select mode; drawing joins it. */
const selectedPart = ref<string | null>(null);
const segmentMode = ref<SegmentMode>("add");
const ikEnabled = ref(false);
/** The joint whose gizmo the Articulate tool is showing. */
const selectedJointId = ref<string | null>(null);
/** Joint-tool state: the joint being edited and the DoF being demonstrated. */
const editingJointId = ref<string | null>(null);
const armedDof = ref<JointDofName | null>(null);
const mirrorRange = ref(false);
const docVersion = ref(0);
doc.subscribe(() => docVersion.value++);

const partsList = computed(() => {
  void docVersion.value;
  return doc.allParts();
});
const posesList = computed(() => {
  void docVersion.value;
  return doc.allPoses();
});
const jointsList = computed(() => {
  void docVersion.value;
  return doc.allJoints();
});
const strokeCounts = computed(() => {
  void docVersion.value;
  const counts: Record<string, number> = {};
  for (const stroke of doc.allStrokes()) {
    if (stroke.partId) counts[stroke.partId] = (counts[stroke.partId] ?? 0) + 1;
  }
  return counts;
});
const activePart = computed(() =>
  activePartId.value ? doc.getPart(activePartId.value) : undefined,
);
let partCounter = 0;

undoStack.onChange = () => {
  canUndo.value = undoStack.canUndo;
  canRedo.value = undoStack.canRedo;
};

onMounted(() => {
  viewport = new Viewport(viewportEl.value!);
  strokeRenderer = new StrokeRenderer(doc, viewport);
  surface = new CanvasSurface(viewport);
  drawTool = new DrawTool(viewport, surface, doc, undoStack);
  selectTool = new SelectTool(viewport, doc, undoStack, strokeRenderer);
  selectTool.onPartSelectionChanged = (partId) => {
    selectedPart.value = partId;
  };
  segmentTool = new SegmentTool(viewport, doc, undoStack, strokeRenderer);
  articulateTool = new ArticulateTool(viewport, doc, undoStack, strokeRenderer);
  articulateTool.onJointSelected = (jointId) => {
    selectedJointId.value = jointId;
  };
  jointTool = new JointTool(viewport, doc, undoStack, strokeRenderer);
  jointTool.onStateChanged = (state) => {
    editingJointId.value = state.jointId;
    armedDof.value = state.armedDof;
    mirrorRange.value = state.mirror;
  };
  jointLines = new JointLines(doc, viewport);
  explodeTool = new ExplodeTool(viewport, doc, undoStack);
  surfacePreview = new SurfacePreview(viewport);
  tools = {
    draw: drawTool,
    erase: new EraseTool(viewport, doc, undoStack, strokeRenderer),
    select: selectTool,
    segment: segmentTool,
    articulate: articulateTool,
    joint: jointTool,
  };
  activate(activeTool.value);

  // gizmos get out of the way while the camera is being driven
  viewport.onCameraActivity((active) => {
    surface?.suppressGizmo("camera", active);
    selectTool?.suppressGizmo(active);
    articulateTool?.suppressGizmo(active);
    jointTool?.suppressGizmo(active);
  });

  window.addEventListener("keydown", onKeyDown);
  void refreshSurfacingMethods();
});

onUnmounted(() => {
  window.removeEventListener("keydown", onKeyDown);
  attachedHandler?.detach();
  attachedHandler = null;
  surfacePreview?.dispose();
  jointLines?.dispose();
  surface?.dispose();
  flowView?.dispose();
  occupancyView?.dispose();
  strokeRenderer?.dispose();
  viewport?.dispose();
});

watch(shape, (value) => surface?.setShape(value));

// while a part is selected, drawing targets it; keep its purple glow
// current in draw mode (including strokes just drawn into it)
watch([activeTool, selectedPart, docVersion], () => {
  if (drawTool) drawTool.targetPartId = selectedPart.value ?? undefined;
  if (activeTool.value !== "draw" || !strokeRenderer) return;
  if (selectedPart.value) {
    for (const stroke of doc.allStrokes()) {
      strokeRenderer.setHighlight(
        stroke.id,
        stroke.partId === selectedPart.value,
        "part",
      );
    }
  } else {
    strokeRenderer.clearHighlights();
  }
});
watch(mirror, (value) => drawTool && (drawTool.mirrorAxis = value));
// undo/redo can re-explode the document while the mode toggle is off; pick
// the mode back up so the panel button and idle drag stay truthful (never
// the reverse: mode-on-at-factor-0 is a valid state)
watch(docVersion, () => {
  if (doc.exploded && !explodeMode.value) {
    explodeMode.value = true;
    if (activeTool.value === "none") activate("none");
  }
});
// the skinned surface follows the rig: re-pose it whenever the document
// changes (articulation writes joint values). Cheap + skipped when the pose
// is unchanged or nothing is bound.
watch(docVersion, () => surfacePreview?.repose(doc.allJoints()));
watch(ikEnabled, (value) => articulateTool?.setIkMode(value));
watch(segmentMode, (value) => segmentTool && (segmentTool.mode = value));
watch([strokeColor, strokeWidth], ([color, width]) => {
  if (drawTool) {
    drawTool.strokeStyle.color = color;
    drawTool.strokeStyle.width = width / 500; // Penzil's slider scaling
  }
});
watch([fillVisible, fillColor], ([visible, color]) => {
  if (drawTool) {
    drawTool.fillStyle.visible = visible as boolean;
    drawTool.fillStyle.color = color as string;
  }
});

function activate(tool: ToolName): void {
  if (!tools) return;
  // the flow view is modal: the strokes on screen are copies laid out in
  // regions, so there is nothing here a tool could correctly act on
  if (flowActive.value && tool !== "none") return;
  attachedHandler?.detach();
  strokeRenderer?.clearHighlights(); // each tool re-applies its own
  activeTool.value = tool;
  attachedHandler =
    tool === "none"
      ? explodeMode.value
        ? (explodeTool ?? null)
        : null
      : tools[tool];
  attachedHandler?.attach();
  // the canvas surface only participates in drawing; elsewhere it would
  // obstruct clicks on strokes
  surface?.setVisible(tool === "draw" && canvasVisible.value);

  if (tool === "joint" || tool === "articulate") {
    expandedPanel.value = "articulations";
  } else if (tool === "segment") {
    // segmentation needs a target part: create one if none exists yet, and
    // surface the parts panel for orientation
    if (!activePartId.value) {
      addPart();
    } else {
      segmentTool?.setActivePart(activePartId.value);
    }
    expandedPanel.value = "parts";
  }
}

/** Explicit switch (panel actions etc.) — never toggles off. */
function setTool(tool: ToolName): void {
  if (tool !== activeTool.value) activate(tool);
}

/** Toolbar buttons and keybinds: picking the active tool again turns it
 *  off, back to the idle state. */
function toggleTool(tool: ToolName): void {
  activate(tool === activeTool.value ? "none" : tool);
}

function resetArticulation(): void {
  if (doc.allJoints().some(jointPosed)) {
    undoStack.push(resetArticulationCommand(doc));
  }
}

const jointHint = computed(() => {
  if (!editingJointId.value) {
    return "Drag from a parent part to a child part to create a joint; click a part to edit its joint";
  }
  if (armedDof.value) {
    return "Drag the gizmo through the motion — the extremes you reach become the range; release to snap back";
  }
  return "T moves the pivot, R aims the axis; arm a DoF in the panel to demonstrate its range";
});

function editJoint(jointId: string): void {
  setTool("joint");
  jointTool?.selectJoint(jointId);
}

function newJoint(): void {
  setTool("joint");
  jointTool?.deselect();
}

function deleteJoint(jointId: string): void {
  // the articulate gizmo may be sitting on this joint
  articulateTool?.deselect();
  jointTool?.deleteJoint(jointId);
}

function setActivePart(partId: string): void {
  activePartId.value = partId;
  segmentTool?.setActivePart(partId);
  // clicking a part also selects it (purple): drawing now targets it, and
  // in select mode its strokes share a gizmo
  selectTool?.select(
    doc.strokesInPart(partId).map((s) => s.id),
    partId,
  );
}

function addPart(): void {
  const part = {
    id: crypto.randomUUID(),
    name: `Part ${++partCounter}`,
  };
  doc.addPart(part);
  setActivePart(part.id);
}

function removePart(partId: string): void {
  if (selectedPart.value === partId) selectTool?.deselect();
  doc.removePart(partId);
  if (activePartId.value === partId) {
    activePartId.value = null;
    segmentTool?.setActivePart(undefined);
  }
}

function toggleExplode(): void {
  if (flowActive.value) return; // modal, same as the tools
  if (explodeMode.value) {
    explodeMode.value = false;
    // drop the drag handler if it was attached, then restore the pose
    if (activeTool.value === "none") activate("none");
    if (doc.allParts().some((p) => p.explodeOffset)) {
      undoStack.push(collapseCommand(doc));
    }
  } else {
    explodeMode.value = true;
    // enter the idle state, where dragging adjusts the explosion
    activate("none");
    // start with a small spread so toggling visibly does something
    if (!doc.exploded) explodeTool?.setFactor(0.5);
    expandedPanel.value = "parts";
  }
}

function selectJoint(jointId: string): void {
  setTool("articulate");
  articulateTool?.selectJoint(jointId);
}

function savePose(): void {
  if (!viewport) return;
  const name = prompt("Pose name", `Pose ${doc.allPoses().length + 1}`);
  if (name === null) return;
  const transforms: Record<string, import("../core/types").Transform> = {};
  for (const stroke of doc.allStrokes()) {
    transforms[stroke.id] = {
      position: { ...stroke.transform.position },
      quaternion: { ...stroke.transform.quaternion },
      scale: { ...stroke.transform.scale },
    };
  }
  doc.addPose({
    id: crypto.randomUUID(),
    name,
    thumbnail: viewport.captureThumbnail(),
    transforms,
  });
  expandedPanel.value = "poses";
}

function applyPose(poseId: string): void {
  const pose = doc.getPose(poseId);
  if (pose) undoStack.push(applyPoseCommand(doc, pose));
}

/** Ask the server which methods it offers; empty list = offline, which the
 *  Surfacer panel reports. Called when the panel opens (and on mount). */
async function refreshSurfacingMethods(): Promise<void> {
  try {
    surfacingMethods.value = await fetchMethods();
  } catch {
    surfacingMethods.value = [];
  }
  if (!surfacingMethods.value.some((m) => m.name === surfacingMethod.value)) {
    surfacingMethod.value = surfacingMethods.value[0]?.name ?? "";
  }
  // seed defaults for params the user hasn't touched yet (new methods, or
  // params added to an adapter since the last fetch)
  for (const method of surfacingMethods.value) {
    const options = (surfacingOptions.value[method.name] ??= {});
    for (const param of method.params) {
      if (!(param.name in options)) options[param.name] = param.default;
    }
  }
}

function setSurfacingOption(
  name: string,
  value: number | boolean | string,
): void {
  const options = surfacingOptions.value[surfacingMethod.value];
  if (options) options[name] = value;
}

function resetSurfacingOptions(): void {
  const method = surfacingMethods.value.find(
    (m) => m.name === surfacingMethod.value,
  );
  if (!method) return;
  surfacingOptions.value[method.name] = Object.fromEntries(
    method.params.map((p) => [p.name, p.default]),
  );
}
watch(expandedPanel, (panel) => {
  if (panel === "surfacer") void refreshSurfacingMethods();
});

async function runSurfacing(): Promise<void> {
  if (!surfacePreview || surfacingBusy.value || !surfacingMethod.value) return;
  const method = surfacingMethod.value;
  surfacingBusy.value = true;
  surfacingProgress.value = 0;
  surfacingMessage.value = "";
  surfacingLog.value = [];
  // the previous surface stays up until this job has something of its own to
  // show, so a job that fails before producing anything costs nothing
  let showingPartials = false;
  try {
    const sketch = buildSurfacingSketch(doc);
    const options: Record<string, unknown> = {
      ...surfacingOptions.value[method],
    };

    // Image-conditioned methods (trellis) consume renders of the sketch
    // rather than the strokes as geometry, and declare how they want them
    // rendered. Doing it here rather than in the adapter keeps one renderer
    // for the whole app — the editor already owns a three.js view of the
    // document, and a second rasterizer server-side would be another thing to
    // keep honest about stroke width, framing and pose.
    const info = surfacingMethods.value.find((m) => m.name === method);
    const spec = info
      ? viewSpecFor(info, surfacingOptions.value[method] ?? {})
      : null;
    if (spec) {
      // The surfaced condition shows the model a solid instead of line art,
      // which means a surface has to exist before this job is submitted — the
      // job protocol is one-way, so an adapter cannot hand the client geometry
      // mid-run and wait for renders of it. Hence a separate ns2s job first.
      // The field it predicts is kept server-side, so a run that also inpaints
      // with the surface reuses this same prediction instead of paying for a
      // second one.
      const surfaced = Boolean(options.surfaced_condition);
      if (surfaced && options.part_based) {
        surfacingLog.value = [
          ...surfacingLog.value,
          "surfaced image condition is whole-object only; conditioning on " +
            "stroke renders for this part-based run",
        ];
      }
      if (surfaced && !options.part_based) {
        surfacingLog.value = [
          ...surfacingLog.value,
          "surfacing the sketch for the image condition…",
        ];
        const surface = await runSurfacingJob({
          method: "ns2s",
          sketch,
          options: {
            part_based: false,
            probability_volume: false,
            // keep the field this mesh came from, for the TRELLIS job
            with_volume: true,
            threshold: Number(options.surface_threshold ?? 0.6),
            blur: Number(options.surface_blur ?? 1.6),
            // must match SURFACE_METHOD_MARGIN in adapters/trellis.py, or the
            // inpainting path predicts its own field rather than reusing this
            // one — a slower run, not a wrong one
            margin: 1.2,
            smooth: Boolean(options.surface_smooth),
          },
          onProgress: (status) => {
            surfacingProgress.value = 0.25 * status.progress;
            surfacingMessage.value = status.message;
          },
          onLog: (lines) => {
            surfacingLog.value = [...surfacingLog.value, ...lines].slice(
              -SURFACING_LOG_CAP,
            );
          },
        });
        surfacingLog.value = [
          ...surfacingLog.value,
          "rendering surfaced views…",
        ];
        options.views = await renderSurfacedViews(sketch, surface, spec);
      } else {
        surfacingLog.value = [...surfacingLog.value, "rendering sketch views…"];
        options.views = await renderConditioningViews(
          sketch,
          spec,
          Boolean(options.part_based),
        );
      }
    }

    // Artifacts a TRELLIS run was asked to keep. They arrive on the partial
    // channel during the run and are only assembled into a view once it has
    // finished — the whole capture is post-hoc by design.
    let flowFrames: FlowFrames | null = null;
    let rawGlb: ArrayBuffer | null = null;
    // NS2S in probability-volume mode publishes one field per unit and no
    // mesh at all, so these arrive instead of geometry rather than beside it
    const occupancyFields: OccupancyVolume[] = [];

    const glb = await runSurfacingJob({
      method,
      sketch,
      options,
      onProgress: (status) => {
        surfacingProgress.value = status.progress;
        surfacingMessage.value = status.message;
      },
      onLog: (lines) => {
        surfacingLog.value = [...surfacingLog.value, ...lines].slice(
          -SURFACING_LOG_CAP,
        );
      },
      // adapters that publish parts as they finish render straight into the
      // viewport; the final result below supersedes the lot
      onPartial: async (name, partial) => {
        if (!surfacePreview) return;
        if (!showingPartials) {
          showingPartials = true;
          clearSurface();
        }
        await surfacePreview.showPartial(name, partial);
        hasSurface.value = true;
        surfaceVisible.value = true;
      },
      onArtifact: (_name, kind, data) => {
        if (kind === "raw") {
          rawGlb = data;
        } else if (kind === "ns2s-volume") {
          try {
            occupancyFields.push(decodeOccupancyVolume(data));
          } catch (error) {
            surfacingLog.value = [
              ...surfacingLog.value,
              `probability volume unusable: ${
                error instanceof Error ? error.message : error
              }`,
            ];
          }
        } else if (kind === "trellis-frames") {
          try {
            flowFrames = decodeFlowFrames(data);
          } catch (error) {
            // a capture that cannot be read costs the view, not the run:
            // the mesh is still on its way and still correct
            surfacingLog.value = [
              ...surfacingLog.value,
              `flow capture unusable: ${
                error instanceof Error ? error.message : error
              }`,
            ];
          }
        }
      },
    });
    await surfacePreview.show(glb);
    // record the result first, so a later skinning hiccup can never lose the
    // mesh from saves (the glb is what rides along in the .json)
    surfaceGlb = { method, bytes: glb };
    hasSurface.value = true;
    surfaceVisible.value = true;
    surfacePreview.setStyle({
      color: surfaceColor.value,
      opacity: surfaceOpacity.value,
    });
    // skin the fresh surface to the rig so articulating deforms it; never let
    // a skinning failure abort the surfacing flow
    bindSurfaceSkin();

    surfaceRawGlb = rawGlb;
    flowHasRaw.value = rawGlb !== null;
    flowShowRaw.value = false;

    // Only a captured run takes the viewport over. Keeping the raw mesh on
    // its own is a question about one object, not about a process, so it
    // stays in the ordinary overlay and the toggle just swaps which mesh is
    // in it — no reason to hide the document for that.
    if (occupancyFields.length > 0) showOccupancyFields(occupancyFields);

    if (flowFrames) {
      await enterFlowView({
        sketch,
        views: Array.isArray(options.views) ? (options.views as string[]) : [],
        frames: flowFrames,
        processedGlb: glb,
        rawGlb,
      });
    }
  } catch (error) {
    alert(`Surfacing failed: ${error instanceof Error ? error.message : error}`);
  } finally {
    surfacingBusy.value = false;
  }
}

function clearSurface(): void {
  exitFlowView();
  clearOccupancyFields();
  surfacePreview?.clear();
  surfaceGlb = null;
  surfaceRawGlb = null;
  flowHasRaw.value = false;
  flowShowRaw.value = false;
  hasSurface.value = false;
}

/** Put a run's occupancy fields up over the sketch.
 *
 *  Nothing is hidden and no tool is switched off: the field is an overlay on
 *  the document, not a mode, and the run it came from delivered no mesh to
 *  compete with it. */
function showOccupancyFields(fields: OccupancyVolume[]): void {
  if (!viewport) return;
  occupancyView ??= new OccupancyFieldView(viewport);
  occupancyView.show(fields);
  occupancyActive.value = occupancyView.isActive;
  occupancyCount.value = occupancyView.count;
  occupancyGrid.value = fields[0]?.grid ?? 0;
  occupancyMax.value = Math.max(...fields.map((field) => field.max), 0);
  occupancyThreshold.value = DEFAULT_OCCUPANCY_THRESHOLD;
  occupancyDensity.value = 1;
  occupancyBlur.value = 0;
  applyOccupancyStyle();
  applyOccupancyBlur();
  if (!occupancyActive.value) {
    surfacingLog.value = [
      ...surfacingLog.value,
      "probability volume has no alignment — nothing says where it sits " +
        "against the sketch, so it is not drawn",
    ];
  }
}

function clearOccupancyFields(): void {
  if (!occupancyActive.value && !occupancyView?.isActive) return;
  occupancyView?.clear();
  occupancyActive.value = false;
  occupancyCount.value = 0;
}

/** Re-blur the grid. Separate from the style because it rewrites the texture
 *  rather than a uniform — see `OccupancyFieldView.setBlur`. */
function applyOccupancyBlur(): void {
  occupancyView?.setBlur(occupancyBlur.value);
}

function applyOccupancyStyle(): void {
  occupancyView?.setStyle({
    threshold: occupancyThreshold.value,
    density: occupancyDensity.value,
  });
}

/** Hand the viewport over to the flow view for a captured run.
 *
 *  The document's own strokes and the surface overlay are hidden rather than
 *  laid out alongside: the regions hold *copies*, and two sets of the same
 *  strokes in one scene — one of them the real, pickable, editable one — is
 *  exactly the confusion this mode exists to avoid. Tools are off for the
 *  same reason. */
async function enterFlowView(run: {
  sketch: ReturnType<typeof buildSurfacingSketch>;
  views: string[];
  frames: FlowFrames | null;
  processedGlb: ArrayBuffer | null;
  rawGlb: ArrayBuffer | null;
}): Promise<void> {
  if (!viewport) return;
  activate("none");
  flowView ??= new TrellisInteractiveView(viewport);
  await flowView.show(run);
  flowView.setSurfaceStyle({
    color: surfaceColor.value,
    opacity: surfaceOpacity.value,
  });

  strokeRenderer?.setVisible(false);
  surfacePreview?.setVisible(false);
  surface?.setVisible(false);

  flowActive.value = true;
  flowLength.value = flowView.timelineLength;
  flowStructureSteps.value = flowView.structureSteps;
  flowStepTimes.value = flowView.stepTimes;
  flowHasRaw.value = run.rawGlb !== null;
  flowViewCount.value = run.views.length;
  flowShowRaw.value = false;
  flowShowSketchOverlay.value = false;
  flowCanOverlaySketch.value = run.frames?.align != null;
  // the end of the timeline is the delivered result, so that is where a run
  // lands — scrubbing back is how you get to the flow
  flowPosition.value = Math.max(0, flowLength.value - 1);
  applyFlowState();
}

/** Give the viewport back. The capture is dropped, not parked. */
function exitFlowView(): void {
  if (!flowActive.value && !flowView?.isActive) return;
  flowView?.clear();
  flowActive.value = false;
  flowLength.value = 0;
  flowStepTimes.value = [];
  flowHasRaw.value = false;
  strokeRenderer?.setVisible(sketchVisible.value);
  surfacePreview?.setVisible(surfaceVisible.value);
  viewport?.invalidate();
}

function applyFlowState(): void {
  flowView?.setState({
    position: flowPosition.value,
    showRaw: flowShowRaw.value,
    showSketchOverlay: flowShowSketchOverlay.value,
    volume: { threshold: flowThreshold.value, density: flowDensity.value },
  });
}

/** The raw/processed switch, in whichever place the result is showing. */
async function setFlowShowRaw(showRaw: boolean): Promise<void> {
  flowShowRaw.value = showRaw;
  if (flowActive.value) {
    applyFlowState();
    return;
  }
  const chosen = showRaw ? surfaceRawGlb : surfaceGlb?.bytes;
  if (!surfacePreview || !chosen) return;
  await surfacePreview.show(chosen);
  surfacePreview.setStyle({
    color: surfaceColor.value,
    opacity: surfaceOpacity.value,
  });
  // the overlay was rebuilt, so the skin binding went with it
  bindSurfaceSkin();
}

/** Skin the current overlay to the rig. Non-fatal: a skinning failure logs
 *  and leaves the (still valid, still saveable) surface undeformed. */
function bindSurfaceSkin(): void {
  try {
    surfacePreview?.bindSkin(buildSurfacingSketch(doc), doc.allJoints());
  } catch (error) {
    console.error("surface skinning failed (surface kept, not skinned):", error);
  }
}

function toggleSketchVisible(): void {
  sketchVisible.value = !sketchVisible.value;
  strokeRenderer?.setVisible(sketchVisible.value);
}

function toggleSurfaceVisible(): void {
  surfaceVisible.value = !surfaceVisible.value;
  surfacePreview?.setVisible(surfaceVisible.value);
}

function setSurfaceColor(color: string): void {
  surfaceColor.value = color;
}

function setSurfaceOpacity(opacity: number): void {
  surfaceOpacity.value = opacity;
}

watch([surfaceColor, surfaceOpacity], ([color, opacity]) => {
  surfacePreview?.setStyle({ color, opacity });
  // the flow view's result region holds a copy of the same surface
  flowView?.setSurfaceStyle({ color, opacity });
});

function cycleGizmoMode(): void {
  const order: GizmoMode[] = ["translate", "rotate", "scale"];
  setGizmoMode(order[(order.indexOf(gizmoMode.value) + 1) % order.length]);
}

function setGizmoMode(mode: GizmoMode): void {
  gizmoMode.value = mode;
  surface?.setGizmoMode(mode);
  selectTool?.setGizmoMode(mode);
}

function toggleCanvasVisible(): void {
  canvasVisible.value = !canvasVisible.value;
  surface?.setVisible(canvasVisible.value);
}

function onKeyDown(event: KeyboardEvent): void {
  if (event.ctrlKey || event.metaKey) {
    if (event.code === "KeyZ") {
      event.preventDefault();
      event.shiftKey ? undoStack.redo() : undoStack.undo();
    } else if (event.code === "KeyY") {
      event.preventDefault();
      undoStack.redo();
    }
    return;
  }
  // avoid hijacking typing in inputs/selects
  if (event.target instanceof HTMLElement && event.target.tagName !== "BODY") {
    return;
  }
  if (event.code === "Escape") selectTool?.deselect();
  else if (event.code === "KeyT") setGizmoMode("translate");
  else if (event.code === "KeyR") setGizmoMode("rotate");
  else if (event.code === "KeyS") setGizmoMode("scale");
  else if (event.code === "KeyD") toggleTool("draw");
  else if (event.code === "KeyE") toggleTool("erase");
  else if (event.code === "KeyV") toggleTool("select");
  else if (event.code === "KeyG") toggleTool("segment");
  else if (event.code === "KeyA") toggleTool("articulate");
  else if (event.code === "KeyJ") toggleTool("joint");
}

function resetStrokes(): void {
  selectTool?.deselect();
  clearSurface();
  doc.clear();
  undoStack.clear();
  activePartId.value = null;
  segmentTool?.setActivePart(undefined);
  partCounter = 0;
  resetDialogOpen.value = false;
  syncExplodeMode();
}

/** Align explode mode with the (re)loaded document's exploded state. */
function syncExplodeMode(): void {
  if (explodeMode.value !== doc.exploded) {
    explodeMode.value = doc.exploded;
    if (activeTool.value === "none") activate("none");
  }
}

function saveThenReset(): void {
  saveFile();
  resetStrokes();
}

function saveFile(): void {
  const name = prompt("File name", "sketch");
  if (name === null) return;
  const json = serializeDocument(doc);
  // the surface rides along so the saved sketch reopens as last seen; it
  // stays out of the document itself (derived output, not sketch data)
  if (surfaceGlb) {
    const glb = bytesToBase64(surfaceGlb.bytes);
    if (!glb) {
      console.warn("surface glb is empty; saving without the surface");
    } else {
      json.surface = { method: surfaceGlb.method, glb };
    }
  }
  downloadBlob(
    new Blob([JSON.stringify(json)], { type: "application/json" }),
    `${name}.json`,
  );
}

function bytesToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000; // avoid call-stack limits on large meshes
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToBytes(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

async function exportFile(): Promise<void> {
  const name = prompt("File name", "sketch");
  if (name === null) return;
  try {
    downloadBlob(await exportGlb(doc), `${name}.glb`);
  } catch (error) {
    alert(`Export failed: ${error}`);
  }
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const el = document.createElement("a");
  el.href = url;
  el.download = filename;
  el.click();
  URL.revokeObjectURL(url);
}

function loadFile(): void {
  const input = document.createElement("input");
  input.type = "file";
  // .gltf/.glb (+ sidecar .bin): SketchLab exports; .json: our format or
  // legacy Penzil. Multiple selection exists so a .gltf and its .bin can be
  // picked together.
  input.accept = ".json,.gltf,.glb,.bin,application/json";
  input.multiple = true;
  input.onchange = async () => {
    const files = [...(input.files ?? [])];
    const byExt = (ext: string) =>
      files.find((f) => f.name.toLowerCase().endsWith(ext));
    const file = files[0];
    if (!file) return;
    try {
      const gltfFile = byExt(".gltf");
      const glbFile = byExt(".glb");
      if (gltfFile || glbFile) {
        const source = glbFile
          ? parseGlb(await glbFile.arrayBuffer())
          : {
              json: JSON.parse(await gltfFile!.text()),
              bin: await byExt(".bin")?.arrayBuffer(),
            };
        const imported = importSketchLabGltf(source.json, source.bin);
        selectTool?.deselect();
        clearSurface();
        doc.clear();
        undoStack.clear();
        activePartId.value = null;
        segmentTool?.setActivePart(undefined);
        for (const part of imported.parts) doc.addPart(part);
        for (const stroke of imported.strokes) doc.addStroke(stroke);
        for (const joint of imported.joints) doc.addJoint(joint);
        partCounter = imported.parts.length;
        syncExplodeMode();
        return;
      }

      const json = JSON.parse(await file.text());
      selectTool?.deselect();
      clearSurface();
      doc.clear();
      undoStack.clear();
      activePartId.value = null;
      segmentTool?.setActivePart(undefined);
      if (isLegacyPenzilJson(json)) {
        for (const stroke of importLegacyPenzil(json)) doc.addStroke(stroke);
      } else {
        const loaded = deserializeDocument(json);
        for (const part of loaded.allParts()) doc.addPart(part);
        for (const pose of loaded.allPoses()) doc.addPose(pose);
        for (const joint of loaded.allJoints()) doc.addJoint(joint);
        doc.exploded = loaded.exploded;
        for (const stroke of loaded.allStrokes()) doc.addStroke(stroke);
        partCounter = loaded.allParts().length;
        const surface = (json as DocumentJson).surface;
        if (surface?.glb && surfacePreview) {
          const bytes = base64ToBytes(surface.glb);
          await surfacePreview.show(bytes);
          surfaceGlb = { method: surface.method, bytes };
          hasSurface.value = true;
          surfaceVisible.value = true;
          surfacePreview.setStyle({
            color: surfaceColor.value,
            opacity: surfaceOpacity.value,
          });
          bindSurfaceSkin();
        }
      }
      syncExplodeMode();
    } catch (error) {
      alert(`Could not load file: ${error}`);
    }
  };
  input.click();
}

/** Open one of a benchmark's stored sketches in the editor.
 *
 *  The benchmark itself is untouched: its state and its jobs live in the
 *  store, not in the window, so surfacing carries on while you look around
 *  (and the window reopens exactly where it was). In edit mode the sketch
 *  comes back through `saveBenchmarkEdit`; otherwise this is just a look. */
async function openBenchmarkSketch(name: string): Promise<void> {
  const benchmarkId = benchmark.state.id;
  if (!benchmarkId) return;
  try {
    const json = await benchmark.readSketchDocument(name);
    selectTool?.deselect();
    clearSurface();
    doc.clear();
    undoStack.clear();
    activePartId.value = null;
    segmentTool?.setActivePart(undefined);
    const loaded = deserializeDocument(json);
    for (const part of loaded.allParts()) doc.addPart(part);
    for (const pose of loaded.allPoses()) doc.addPose(pose);
    for (const joint of loaded.allJoints()) doc.addJoint(joint);
    doc.exploded = loaded.exploded;
    for (const stroke of loaded.allStrokes()) doc.addStroke(stroke);
    partCounter = loaded.allParts().length;
    syncExplodeMode();

    if (benchmark.state.editMode) {
      benchmark.beginEdit(name);
      benchmarkOpen.value = false;
      return; // no surface overlay: the point is to see and change the strokes
    }

    // the viewed run's finished surface, read from disk on demand — the
    // benchmark holds no meshes in memory to hand over
    const bytes = await benchmark.readViewedSurface(name);
    if (bytes && surfacePreview) {
      await surfacePreview.show(bytes.slice(0)); // parse a throwaway copy
      surfaceGlb = { method: benchmark.state.viewing?.adapter ?? "", bytes };
      hasSurface.value = true;
      surfaceVisible.value = true;
      surfacePreview.setStyle({
        color: surfaceColor.value,
        opacity: surfaceOpacity.value,
      });
      // the sketch carries its joints, so bind the overlay to the rig here as
      // well — otherwise a benchmark surface sits still while the strokes move
      bindSurfaceSkin();
    }
    benchmarkOpen.value = false;
  } catch (error) {
    // capped: an alert is a modal the user has to dismiss, so whatever the
    // failure was it must not arrive as a screenful of text
    alert(`Could not open benchmark sketch: ${brief(error)}`);
  }
}

/** An error as one readable line. */
function brief(error: unknown): string {
  const text = String(error instanceof Error ? error.message : error)
    .trim()
    .replace(/\s+/g, " ");
  return text.length > 300 ? `${text.slice(0, 300)}…` : text;
}

/** Write the edited document back into the benchmark's sketches/ folder.
 *
 *  Articulation travels with it: strokes always hold absolute transforms, so
 *  a sketch saved while posed is stored posed, and that is the geometry the
 *  next run surfaces. Only the cached surface is dropped — it belongs to the
 *  strokes as they were, not as they are. */
async function saveBenchmarkEdit(): Promise<void> {
  const name = benchmark.state.editing;
  if (!name) return;
  try {
    const json = serializeDocument(doc);
    await benchmark.saveEdit(name, { ...json, surface: undefined });
    benchmarkOpen.value = true;
  } catch (error) {
    alert(`Could not save benchmark sketch: ${error}`);
  }
}

function cancelBenchmarkEdit(): void {
  benchmark.cancelEdit();
  benchmarkOpen.value = true;
}
</script>

<style scoped>
.viewport {
  width: 100%;
  height: 100%;
  touch-action: none;
}

.toolbar {
  position: absolute;
  top: 12px;
  display: flex;
  gap: 8px;
  align-items: center;
}

.toolbar.left {
  left: 12px;
}

.toolbar.right {
  right: 12px;
}

.toolbar button,
.toolbar select,
.toolbar label {
  height: 36px;
  padding: 0 14px;
  border: none;
  border-radius: 18px;
  background: #ffffff;
  font-weight: 900;
  cursor: pointer;
  filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.12));
  display: flex;
  align-items: center;
  gap: 4px;
}

.toolbar input[type="color"] {
  width: 36px;
  height: 36px;
  padding: 4px;
  border: none;
  border-radius: 18px;
  background: #ffffff;
  cursor: pointer;
  filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.12));
}

.toolbar input[type="range"] {
  width: 90px;
}

.toolbar button:hover:not(:disabled),
.toolbar select:hover {
  background: #ffe8b3;
}

.toolbar button.active {
  background: #ffd16b;
}

.toolbar button:disabled {
  opacity: 0.5;
  cursor: default;
}

.hint {
  font-size: 0.85em;
  font-weight: 900;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 18px;
  padding: 8px 14px;
}

.toolbar button.part-chip {
  background: #e3d1ff;
}

.toolbar button.part-chip:hover {
  background: #d0b3ff;
}

.divider {
  width: 1px;
  height: 24px;
  background: rgba(0, 0, 0, 0.15);
}

.benchmark-edit-bar {
  position: absolute;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  font-size: 12px;
  background: #f0ebfa;
  border: 1px solid #7b4bd8;
  border-radius: 4px;
  z-index: 20;
}

.modal-backdrop {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
}

.modal {
  background: #ffffff;
  border-radius: 12px;
  padding: 20px 24px;
  max-width: 420px;
  filter: drop-shadow(0 4px 24px rgba(0, 0, 0, 0.2));
}

.modal-buttons {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.modal-buttons button {
  height: 36px;
  padding: 0 14px;
  border: none;
  border-radius: 18px;
  background: #eeeeee;
  font-weight: 900;
  cursor: pointer;
}

.modal-buttons button:hover {
  background: #ffe8b3;
}

.modal-buttons button.danger {
  background: #ffd4d4;
}

.modal-buttons button.danger:hover {
  background: #ffb3b3;
}
</style>
