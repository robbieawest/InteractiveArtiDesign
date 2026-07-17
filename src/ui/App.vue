<template>
  <div ref="viewportEl" class="viewport"></div>

  <div class="toolbar left">
    <button
      :class="{ active: activeTool === 'draw' }"
      title="Draw (D) — left-drag on the canvas surface to sketch"
      @click="setTool('draw')"
    >
      Draw
    </button>
    <button
      :class="{ active: activeTool === 'erase' }"
      title="Erase (E) — left-drag over strokes to delete them"
      @click="setTool('erase')"
    >
      Erase
    </button>
    <button
      :class="{ active: activeTool === 'select' }"
      title="Select (V) — click a stroke, double-click a part; T/R/S set the gizmo, Delete removes, Esc deselects"
      @click="setTool('select')"
    >
      Select
    </button>
    <button
      :class="{ active: activeTool === 'segment' }"
      title="Segment (G) — drag a pen through strokes to add/remove them from the active part (see Parts panel)"
      @click="setTool('segment')"
    >
      Segment
    </button>
    <button
      :class="{ active: activeTool === 'articulate' }"
      title="Articulate (A) — click a part to drive its joint along its axis; toggle IK in the Articulations panel to drag parts freely (collapse the exploded view first)"
      @click="setTool('articulate')"
    >
      Articulate
    </button>
    <button
      :class="{ active: activeTool === 'joint' }"
      title="Joint (J) — drag from a parent part to a child part to create a joint; click a part to edit the joint that drives it"
      @click="setTool('joint')"
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
    <template v-else-if="activeTool === 'explode'">
      <span class="hint">
        Drag away from the model's center to explode it; turn explode off in
        the Parts panel to restore the original pose
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
    :exploding="activeTool === 'explode'"
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
  />

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
import { SketchDocument } from "../core/SketchDocument";
import {
  UndoStack,
  applyPoseCommand,
  resetArticulationCommand,
} from "../core/undo";
import { deserializeDocument, serializeDocument } from "../core/serialization";
import { importLegacyPenzil, isLegacyPenzilJson } from "../core/legacyPenzil";
import { importSketchLabGltf, parseGlb } from "../engine/importSketchLab";
import type { JointDofName, SurfaceShape } from "../core/types";
import { jointPosed } from "../core/types";

type ToolName =
  | "draw"
  | "erase"
  | "select"
  | "segment"
  | "articulate"
  | "joint"
  | "explode";

const viewportEl = useTemplateRef("viewportEl");
const doc = new SketchDocument();
const undoStack = new UndoStack();

let viewport: Viewport | undefined;
let strokeRenderer: StrokeRenderer | undefined;
let surface: CanvasSurface | undefined;
let tools: Record<ToolName, { attach(): void; detach(): void }> | undefined;
let drawTool: DrawTool | undefined;
let selectTool: SelectTool | undefined;
let segmentTool: SegmentTool | undefined;
let articulateTool: ArticulateTool | undefined;
let jointTool: JointTool | undefined;
let jointLines: JointLines | undefined;

const activeTool = ref<ToolName>("draw");
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
  tools = {
    draw: drawTool,
    erase: new EraseTool(viewport, doc, undoStack, strokeRenderer),
    select: selectTool,
    segment: segmentTool,
    articulate: articulateTool,
    joint: jointTool,
    explode: new ExplodeTool(viewport, doc, undoStack),
  };
  tools[activeTool.value].attach();

  // gizmos get out of the way while the camera is being driven
  viewport.onCameraActivity((active) => {
    surface?.suppressGizmo("camera", active);
    selectTool?.suppressGizmo(active);
    articulateTool?.suppressGizmo(active);
    jointTool?.suppressGizmo(active);
  });

  window.addEventListener("keydown", onKeyDown);
});

onUnmounted(() => {
  window.removeEventListener("keydown", onKeyDown);
  tools?.[activeTool.value].detach();
  jointLines?.dispose();
  surface?.dispose();
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

function setTool(tool: ToolName): void {
  if (!tools || tool === activeTool.value) return;
  tools[activeTool.value].detach();
  strokeRenderer?.clearHighlights(); // each tool re-applies its own
  activeTool.value = tool;
  tools[tool].attach();
  // the canvas surface only participates in drawing; elsewhere it would
  // obstruct clicks on strokes
  surface?.setVisible(tool === "draw" && canvasVisible.value);

  if (tool === "explode") {
    expandedPanel.value = "parts";
  } else if (tool === "joint") {
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
  } else if (tool === "articulate") {
    expandedPanel.value = "articulations";
  }
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
  // turning the tool off (or switching to any other tool) collapses the
  // parts back to their original pose
  setTool(activeTool.value === "explode" ? "select" : "explode");
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
  else if (event.code === "KeyD") setTool("draw");
  else if (event.code === "KeyE") setTool("erase");
  else if (event.code === "KeyV") setTool("select");
  else if (event.code === "KeyG") setTool("segment");
  else if (event.code === "KeyA") setTool("articulate");
  else if (event.code === "KeyJ") setTool("joint");
}

function resetStrokes(): void {
  selectTool?.deselect();
  doc.clear();
  undoStack.clear();
  activePartId.value = null;
  segmentTool?.setActivePart(undefined);
  partCounter = 0;
  resetDialogOpen.value = false;
}

function saveThenReset(): void {
  saveFile();
  resetStrokes();
}

function saveFile(): void {
  const name = prompt("File name", "sketch");
  if (name === null) return;
  downloadBlob(
    new Blob([JSON.stringify(serializeDocument(doc))], {
      type: "application/json",
    }),
    `${name}.json`,
  );
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
        doc.clear();
        undoStack.clear();
        activePartId.value = null;
        segmentTool?.setActivePart(undefined);
        for (const part of imported.parts) doc.addPart(part);
        for (const stroke of imported.strokes) doc.addStroke(stroke);
        for (const joint of imported.joints) doc.addJoint(joint);
        partCounter = imported.parts.length;
        return;
      }

      const json = JSON.parse(await file.text());
      selectTool?.deselect();
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
        doc.exploded = loaded.exploded;
        for (const stroke of loaded.allStrokes()) doc.addStroke(stroke);
        partCounter = loaded.allParts().length;
      }
    } catch (error) {
      alert(`Could not load file: ${error}`);
    }
  };
  input.click();
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
