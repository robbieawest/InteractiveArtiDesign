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
      title="Select (V) — click a stroke to select it; T/R/S set the gizmo, Delete removes it, Esc deselects"
      @click="setTool('select')"
    >
      Select
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
      <select v-model="mirror" title="Mirror — also draw a twin stroke reflected across a world axis">
        <option value="off">No mirror</option>
        <option value="x">Mirror X</option>
        <option value="y">Mirror Y</option>
        <option value="z">Mirror Z</option>
      </select>

      <span class="divider" />

      <input
        v-model="strokeColor"
        type="color"
        title="Stroke color"
      />
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
    </template>
    <template v-else-if="activeTool === 'select'">
      <button
        title="Gizmo mode — press T (translate), R (rotate) or S (scale), or click to cycle"
        @click="cycleGizmoMode"
      >
        {{ gizmoMode }}
      </button>
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

  <div v-if="resetDialogOpen" class="modal-backdrop" @click.self="resetDialogOpen = false">
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
import { onMounted, onUnmounted, ref, useTemplateRef, watch } from "vue";
import { Viewport } from "../engine/Viewport";
import { StrokeRenderer } from "../engine/StrokeRenderer";
import { CanvasSurface, type GizmoMode } from "../engine/CanvasSurface";
import { exportGlb } from "../engine/exportGlb";
import { DrawTool, type MirrorAxis } from "../tools/DrawTool";
import { EraseTool } from "../tools/EraseTool";
import { SelectTool } from "../tools/SelectTool";
import { SketchDocument } from "../core/SketchDocument";
import { UndoStack } from "../core/undo";
import { deserializeDocument, serializeDocument } from "../core/serialization";
import { importLegacyPenzil, isLegacyPenzilJson } from "../core/legacyPenzil";
import type { SurfaceShape } from "../core/types";

type ToolName = "draw" | "erase" | "select";

const viewportEl = useTemplateRef("viewportEl");
const doc = new SketchDocument();
const undoStack = new UndoStack();

let viewport: Viewport | undefined;
let strokeRenderer: StrokeRenderer | undefined;
let surface: CanvasSurface | undefined;
let tools: Record<ToolName, { attach(): void; detach(): void }> | undefined;
let drawTool: DrawTool | undefined;
let selectTool: SelectTool | undefined;

const activeTool = ref<ToolName>("draw");
const shape = ref<SurfaceShape>("plane");
const gizmoMode = ref<GizmoMode>("translate");
const canvasVisible = ref(true);
const mirror = ref<MirrorAxis>("off");
const strokeColor = ref("#1c1c1e");
const strokeWidth = ref(5);
const fillVisible = ref(false);
const fillColor = ref("#1c1c1e");
const canUndo = ref(false);
const canRedo = ref(false);
const resetDialogOpen = ref(false);

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
  tools = {
    draw: drawTool,
    erase: new EraseTool(viewport, doc, undoStack, strokeRenderer),
    select: selectTool,
  };
  tools[activeTool.value].attach();
  window.addEventListener("keydown", onKeyDown);
});

onUnmounted(() => {
  window.removeEventListener("keydown", onKeyDown);
  tools?.[activeTool.value].detach();
  surface?.dispose();
  strokeRenderer?.dispose();
  viewport?.dispose();
});

watch(shape, (value) => surface?.setShape(value));
watch(mirror, (value) => drawTool && (drawTool.mirrorAxis = value));
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
  activeTool.value = tool;
  tools[tool].attach();
  // the canvas surface only participates in drawing; elsewhere it would
  // obstruct clicks on strokes
  surface?.setVisible(tool === "draw" && canvasVisible.value);
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
  if (event.code === "KeyT") setGizmoMode("translate");
  else if (event.code === "KeyR") setGizmoMode("rotate");
  else if (event.code === "KeyS") setGizmoMode("scale");
  else if (event.code === "KeyD") setTool("draw");
  else if (event.code === "KeyE") setTool("erase");
  else if (event.code === "KeyV") setTool("select");
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

function resetStrokes(): void {
  doc.clear();
  undoStack.clear();
  resetDialogOpen.value = false;
}

function saveThenReset(): void {
  saveFile();
  resetStrokes();
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
  input.accept = ".json,application/json";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      const json = JSON.parse(await file.text());
      const strokes = isLegacyPenzilJson(json)
        ? importLegacyPenzil(json)
        : deserializeDocument(json).allStrokes();
      doc.clear();
      undoStack.clear();
      for (const stroke of strokes) doc.addStroke(stroke);
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
