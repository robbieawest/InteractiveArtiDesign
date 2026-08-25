<template>
  <div class="panels">
    <!-- Benchmark is a header only: it opens a full-screen window rather than
         expanding in place, since a grid of sketches needs the whole screen. -->
    <div class="panel">
      <button
        class="panel-header"
        title="Standardized surfacing comparison — run adapters and parameter permutations over a folder of sketches"
        @click="$emit('open-benchmark')"
      >
        <span>Benchmark</span>
        <span class="chevron">⤢</span>
      </button>
    </div>

    <div v-for="panel in panels" :key="panel" class="panel">
      <div
        v-if="expanded === panel"
        class="panel-body"
        :class="{ 'panel-body-tall': panel === 'surfacer' }"
      >
        <!-- Parts -->
        <template v-if="panel === 'parts'">
          <div v-if="parts.length === 0" class="empty">
            No parts yet. Create one, then paint strokes into it with the
            Segment tool.
          </div>
          <div
            v-for="part in parts"
            :key="part.id"
            class="row"
            :class="{ active: part.id === activePartId }"
            :title="'Click to make this the active part for the segmentation pen'"
            @click="$emit('set-active-part', part.id)"
          >
            <span class="row-label">{{ part.name }}</span>
            <span class="count">{{ strokeCounts[part.id] ?? 0 }}</span>
            <button
              class="mini danger"
              title="Delete the part (strokes stay, but are unassigned)"
              @click.stop="$emit('remove-part', part.id)"
            >
              ×
            </button>
          </div>
          <div class="panel-actions">
            <button @click="$emit('add-part')">New part</button>
            <button
              :disabled="parts.length === 0"
              :class="{ active: exploding }"
              :title="
                exploding
                  ? 'Turn explode off — every part returns to its original placement'
                  : 'Explode — with no tool active, left-drag away from the model center to spread the parts apart; other tools keep working on the exploded model'
              "
              @click="$emit('toggle-explode')"
            >
              {{ exploding ? "Turn off explode" : "Explode" }}
            </button>
          </div>
        </template>

        <!-- Poses -->
        <template v-else-if="panel === 'poses'">
          <div v-if="poses.length === 0" class="empty">
            No poses yet. Arrange your parts, then save the pose.
          </div>
          <div class="pose-grid">
            <div
              v-for="pose in poses"
              :key="pose.id"
              class="pose"
              :title="'Click to apply ' + pose.name"
              @click="$emit('apply-pose', pose.id)"
            >
              <img :src="pose.thumbnail" :alt="pose.name" />
              <span class="pose-name">{{ pose.name }}</span>
              <button
                class="mini danger pose-delete"
                title="Delete this pose"
                @click.stop="$emit('remove-pose', pose.id)"
              >
                ×
              </button>
            </div>
          </div>
          <div class="panel-actions">
            <button @click="$emit('save-pose')">Save pose</button>
          </div>
        </template>

        <!-- Surfacer -->
        <template v-else-if="panel === 'surfacer'">
          <div v-if="methods.length === 0" class="empty">
            Surfacing server offline — start it in surfacing-server/ (see its
            README), then reopen this panel.
          </div>
          <template v-else>
            <select
              class="method-select"
              :value="surfacingMethod"
              title="Surfacing method — one server adapter per method"
              @change="
                $emit(
                  'set-surfacing-method',
                  ($event.target as HTMLSelectElement).value,
                )
              "
            >
              <option
                v-for="method in methods"
                :key="method.name"
                :value="method.name"
              >
                {{ method.name }}
              </option>
            </select>
            <template v-if="methodParams.length > 0">
              <div
                v-for="param in methodParams"
                :key="param.name"
                class="row param-row"
                :class="{ disabled: !paramEnabled(param) }"
                :title="param.help"
              >
                <span class="row-label">{{ param.label }}</span>
                <input
                  v-if="param.type === 'bool'"
                  type="checkbox"
                  :checked="surfacingOptions[param.name] === true"
                  :disabled="!paramEnabled(param)"
                  @change="
                    $emit(
                      'set-surfacing-option',
                      param.name,
                      ($event.target as HTMLInputElement).checked,
                    )
                  "
                />
                <select
                  v-else-if="param.type === 'choice'"
                  class="param-input"
                  :value="surfacingOptions[param.name]"
                  :disabled="!paramEnabled(param)"
                  @change="
                    $emit(
                      'set-surfacing-option',
                      param.name,
                      ($event.target as HTMLSelectElement).value,
                    )
                  "
                >
                  <option
                    v-for="choice in param.choices"
                    :key="choice"
                    :value="choice"
                  >
                    {{ choice }}
                  </option>
                </select>
                <input
                  v-else
                  type="number"
                  class="param-input"
                  :value="surfacingOptions[param.name]"
                  :min="param.min"
                  :max="param.max"
                  :step="param.step ?? (param.type === 'int' ? 1 : 'any')"
                  :disabled="!paramEnabled(param)"
                  @change="commitNumber(param, $event)"
                />
              </div>
              <button
                class="mini defaults"
                title="Reset every parameter to the adapter's defaults"
                @click="$emit('reset-surfacing-options')"
              >
                Reset to defaults
              </button>
            </template>
          </template>

          <!-- general surface appearance (independent of the method) -->
          <div class="surf-appearance">
            <span class="surf-heading">Surface appearance</span>
            <div class="row param-row" title="Color of the surface overlay">
              <span class="row-label">Color</span>
              <input
                type="color"
                :value="surfaceColor"
                @input="
                  $emit(
                    'set-surface-color',
                    ($event.target as HTMLInputElement).value,
                  )
                "
              />
            </div>
            <div
              class="row param-row"
              title="Opacity of the surface overlay (lower = strokes show through)"
            >
              <span class="row-label">Opacity</span>
              <input
                type="range"
                min="0.05"
                max="1"
                step="0.05"
                :value="surfaceOpacity"
                @input="
                  $emit(
                    'set-surface-opacity',
                    Number(($event.target as HTMLInputElement).value),
                  )
                "
              />
            </div>
          </div>

          <pre
            v-if="surfacingLog.length > 0"
            ref="logEl"
            class="surf-log"
            @scroll="onLogScroll"
            >{{ surfacingLog.join("\n") }}</pre
          >
          <!-- One replaced-in-place line, not log history: the adapter's label
               for the step it is on. The percentage cannot stand in for it —
               `rescale_t` makes the flow steps wildly non-uniform, so "50%"
               and "t 0.500->0.313" are very different facts. -->
          <p v-if="surfacing && surfacingMessage" class="surf-status">
            {{ surfacingMessage }}
          </p>
          <div class="panel-actions">
            <button
              :disabled="surfacing || methods.length === 0"
              title="Send the sketch to the surfacing server and show the resulting mesh"
              @click="$emit('surface')"
            >
              {{
                surfacing
                  ? `Surfacing ${Math.round(surfacingProgress * 100)}%`
                  : "Surface"
              }}
            </button>
            <button
              v-if="hasSurface"
              title="Remove the surfacing result from the scene"
              @click="$emit('clear-surface')"
            >
              Clear
            </button>
          </div>
        </template>

        <!-- Articulations -->
        <template v-else>
          <div v-if="joints.length === 0" class="empty">
            No joints yet. Click "New joint" and drag from one part to
            another, or load a SketchLab model (.gltf) to bring in its rig.
          </div>
          <template v-for="joint in joints" :key="joint.id">
            <div
              class="row"
              :class="{ active: joint.id === selectedJointId || joint.id === editingJointId }"
              :title="jointTooltip(joint)"
              @click="$emit('select-joint', joint.id)"
            >
              <span class="row-label">{{ joint.name }}</span>
              <span class="count">{{ jointKindLabel(joint) }}</span>
              <span class="count value">{{ jointValueLabel(joint) }}</span>
              <button
                class="mini"
                title="Edit this joint (Joint tool): place its axis, demonstrate its ranges"
                @click.stop="$emit('edit-joint', joint.id)"
              >
                ✎
              </button>
              <button
                class="mini danger"
                title="Delete this joint (the parts stay)"
                @click.stop="$emit('delete-joint', joint.id)"
              >
                ×
              </button>
            </div>
            <template v-if="joint.id === editingJointId">
              <div
                v-for="dof in JOINT_DOF_NAMES"
                :key="dof"
                class="row dof-row"
                :class="{ active: dof === armedDof }"
                :title="'Click to arm this DoF, then drag the gizmo through the motion — the extremes you reach become its range. Click again to disarm.'"
                @click="$emit('arm-dof', dof)"
              >
                <span class="row-label">{{ DOF_LABELS[dof] }}</span>
                <span class="count value">{{ dofRangeLabel(joint, dof) }}</span>
                <span v-if="dof === armedDof" class="count recording">●</span>
              </div>
              <label
                class="row dof-row mirror-toggle"
                title="Commit demonstrated ranges symmetrically (± the largest extreme)"
              >
                <input
                  type="checkbox"
                  :checked="mirror"
                  @change="$emit('toggle-mirror')"
                />
                Mirror range
              </label>
            </template>
          </template>
          <div class="panel-actions">
            <button
              title="Create a joint (Joint tool): drag from the parent part to the child part in the viewport"
              @click="$emit('new-joint')"
            >
              New joint
            </button>
            <label
              v-if="joints.length > 0"
              class="ik-toggle"
              title="Inverse kinematics — drag a part freely with the Articulate tool and the joint chain solves to follow"
            >
              <input
                type="checkbox"
                :checked="ik"
                @change="$emit('toggle-ik')"
              />
              IK
            </label>
            <button
              v-if="joints.length > 0"
              title="Return every joint to its rest pose (undoable)"
              @click="$emit('reset-articulation')"
            >
              Reset pose
            </button>
          </div>
        </template>
      </div>

      <button
        class="panel-header"
        :title="headerTooltips[panel]"
        @click="$emit('set-expanded', expanded === panel ? null : panel)"
      >
        <span>{{ panelLabels[panel] }}</span>
        <span class="chevron">{{ expanded === panel ? "v" : "^" }}</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, useTemplateRef, watch } from "vue";
import type { Joint, JointDofName, Part, Pose } from "../core/types";
import { JOINT_DOF_NAMES, dofUnlocked, jointKindLabel } from "../core/types";
import type {
  MethodInfo,
  MethodOptions,
  MethodParam,
} from "../surfacing/client";

export type PanelName = "parts" | "poses" | "articulations" | "surfacer";

const props = defineProps<{
  expanded: PanelName | null;
  parts: Part[];
  poses: Pose[];
  joints: Joint[];
  ik: boolean;
  selectedJointId: string | null;
  editingJointId: string | null;
  armedDof: JointDofName | null;
  mirror: boolean;
  activePartId: string | null;
  strokeCounts: Record<string, number>;
  exploding: boolean;
  /** Methods from the surfacing server; empty = server offline. */
  methods: MethodInfo[];
  surfacingMethod: string;
  /** Parameter declarations of the selected method. */
  methodParams: MethodParam[];
  /** Current values for those parameters. */
  surfacingOptions: MethodOptions;
  surfacing: boolean;
  surfacingProgress: number;
  /** The adapter's label for the current step, shown while a job runs. */
  surfacingMessage: string;
  hasSurface: boolean;
  /** Surface overlay appearance (applies live to the shown mesh). */
  surfaceColor: string;
  surfaceOpacity: number;
  /** Free-form adapter output shown in the log window. */
  surfacingLog: string[];
}>();

const emit = defineEmits<{
  "set-expanded": [panel: PanelName | null];
  "open-benchmark": [];
  "set-active-part": [partId: string];
  "add-part": [];
  "remove-part": [partId: string];
  "toggle-explode": [];
  "save-pose": [];
  "apply-pose": [poseId: string];
  "remove-pose": [poseId: string];
  "toggle-ik": [];
  "reset-articulation": [];
  "select-joint": [jointId: string];
  "edit-joint": [jointId: string];
  "delete-joint": [jointId: string];
  "new-joint": [];
  "arm-dof": [dof: JointDofName];
  "toggle-mirror": [];
  "set-surfacing-method": [method: string];
  "set-surfacing-option": [name: string, value: number | boolean | string];
  "reset-surfacing-options": [];
  "set-surface-color": [color: string];
  "set-surface-opacity": [opacity: number];
  surface: [];
  "clear-surface": [];
}>();

/** A param with an `enabledWhen` clause is only editable while the referenced
 *  param currently holds the required value (e.g. per-part iterations unlock
 *  once part-based evaluation is ticked on).
 *
 *  `lockedWhileSurfaced` is the other gate: some params decide what a run has
 *  to *record*, so they cannot be applied to a surface that already exists.
 *  Clearing the surface releases them. */
function paramEnabled(param: MethodParam): boolean {
  if (param.lockedWhileSurfaced && props.hasSurface) return false;
  const cond = param.enabledWhen;
  if (!cond) return true;
  return props.surfacingOptions[cond.param] === cond.equals;
}

/** Number inputs: parse, clamp to the declared range, round ints, and fall
 *  back to the default when the field is left non-numeric. */
function commitNumber(param: MethodParam, event: Event): void {
  const input = event.target as HTMLInputElement;
  let value = Number(input.value);
  if (!Number.isFinite(value)) value = param.default as number;
  if (param.min !== undefined) value = Math.max(param.min, value);
  if (param.max !== undefined) value = Math.min(param.max, value);
  if (param.type === "int") value = Math.round(value);
  input.value = String(value); // reflect the sanitized value back
  emit("set-surfacing-option", param.name, value);
}

// the log window follows new lines unless the user scrolled up to read
const logEl = useTemplateRef<HTMLPreElement>("logEl");
let logFollowing = true;

function onLogScroll(): void {
  const el = logEl.value;
  if (!el) return;
  logFollowing = el.scrollTop + el.clientHeight >= el.scrollHeight - 12;
}

watch(
  () => props.surfacingLog.length,
  async () => {
    await nextTick();
    const el = logEl.value;
    if (el && logFollowing) el.scrollTop = el.scrollHeight;
  },
);

const DOF_LABELS: Record<JointDofName, string> = {
  translation: "Slide",
  twist: "Twist",
  swingU: "Swing U",
  swingV: "Swing V",
};

const toDegrees = (radians: number) => (radians * 180) / Math.PI;

const dofValueLabel = (dof: JointDofName, value: number) =>
  dof === "translation" ? value.toFixed(2) : `${toDegrees(value).toFixed(0)}°`;

/** Current values of the unlocked DoFs, compact. */
function jointValueLabel(joint: Joint): string {
  const parts = JOINT_DOF_NAMES.filter((d) => dofUnlocked(joint.dofs[d])).map(
    (d) => dofValueLabel(d, joint.dofs[d].value),
  );
  return parts.join(" ");
}

function dofRangeLabel(joint: Joint, dof: JointDofName): string {
  const [min, max] = joint.dofs[dof].range;
  if (min === 0 && max === 0) return "locked";
  return `${dofValueLabel(dof, min)} .. ${dofValueLabel(dof, max)}`;
}

function jointTooltip(joint: Joint): string {
  const ranges = JOINT_DOF_NAMES.filter((d) => dofUnlocked(joint.dofs[d]))
    .map((d) => `${DOF_LABELS[d]} ${dofRangeLabel(joint, d)}`)
    .join(", ");
  return `${jointKindLabel(joint)} joint${ranges ? ` — ${ranges}` : ""}. Click to drive it (Articulate tool); ✎ edits it.`;
}

const panels: PanelName[] = ["surfacer", "articulations", "poses", "parts"];
const panelLabels: Record<PanelName, string> = {
  parts: "Parts",
  poses: "Poses",
  articulations: "Articulations",
  surfacer: "Surfacer",
};
const headerTooltips: Record<PanelName, string> = {
  parts: "Part segmentation — group strokes into parts with the Segment tool",
  poses: "Saved poses — snapshots of every stroke's placement",
  articulations:
    "Joints between parts — articulate them with the Articulate tool (A), toggle IK here",
  surfacer:
    "Surface the sketch — pick a method and run it on the local surfacing server",
};
</script>

<style scoped>
.panels {
  position: absolute;
  right: 12px;
  /* spans the viewport so a tall panel is clamped by it instead of running off
     the top of the screen; the strip itself is click-through, only the panels
     take pointer events */
  top: 60px; /* clear of the right-hand toolbar */
  bottom: 12px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  gap: 8px;
  width: 260px;
  pointer-events: none;
}

.panel {
  display: flex;
  flex-direction: column;
  min-height: 0; /* let the stack shrink this panel when space runs out */
  pointer-events: auto;
}

.panel-header {
  flex: none;
  height: 36px;
  padding: 0 16px;
  border: none;
  border-radius: 18px;
  background: #ffffff;
  font-weight: 900;
  cursor: pointer;
  filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.12));
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header:hover {
  background: #ffe8b3;
}

.chevron {
  font-weight: 900;
}

.panel-body {
  background: #ffffff;
  border-radius: 12px;
  margin-bottom: 6px;
  padding: 10px;
  max-height: 40vh;
  min-height: 0;
  overflow-y: auto;
  filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.12));
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* the surfacer panel is the one you drag taller (the log), so it gets more
   room than a list panel before it starts scrolling internally */
.panel-body-tall {
  max-height: 82vh;
}

.empty {
  font-size: 0.85em;
  color: #666;
  padding: 4px;
}

.row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 8px;
  cursor: pointer;
}

.row:hover {
  background: #fff3d6;
}

.row.active {
  background: #ffe8b3;
}

.row-label {
  flex: 1;
  font-weight: 900;
}

.count {
  font-size: 0.8em;
  color: #666;
}

.mini {
  width: 22px;
  height: 22px;
  border: none;
  border-radius: 11px;
  background: #eeeeee;
  font-weight: 900;
  cursor: pointer;
  line-height: 1;
}

.mini.danger:hover {
  background: #ffb3b3;
}

.surf-status {
  /* same reason as .surf-log above: a flex child in the panel's column is
     shrunk to fit `max-height: 40vh`, and `overflow: hidden` below means the
     text cannot push back — without this the line renders at zero height */
  flex: none;
  margin: 4px 0 0;
  font-size: 0.85em;
  font-variant-numeric: tabular-nums;
  color: #555555;
  /* the flow labels carry a t range; keep it on one line and let the front
     of the string win, since that is the step number */
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.panel-actions {
  display: flex;
  gap: 6px;
  margin-top: 4px;
}

.panel-actions button {
  flex: 1;
  height: 32px;
  border: none;
  border-radius: 16px;
  background: #eeeeee;
  font-weight: 900;
  cursor: pointer;
}

.panel-actions button:hover:not(:disabled) {
  background: #ffe8b3;
}

.panel-actions button:disabled {
  opacity: 0.5;
  cursor: default;
}

.pose-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.pose {
  position: relative;
  cursor: pointer;
  border-radius: 8px;
  overflow: hidden;
  background: #f4f4f4;
}

.pose:hover {
  outline: 2px solid #ffd16b;
}

.pose img {
  width: 100%;
  display: block;
}

.pose-name {
  display: block;
  font-size: 0.75em;
  font-weight: 900;
  padding: 2px 6px 4px;
}

.pose-delete {
  position: absolute;
  top: 4px;
  right: 4px;
}

.panel-actions button.active {
  background: #ffd16b;
}

.dof-row {
  margin-left: 16px;
  padding-top: 3px;
  padding-bottom: 3px;
  font-size: 0.9em;
}

.dof-row .row-label {
  font-weight: 700;
}

.recording {
  color: #d03030;
}

.mirror-toggle {
  gap: 6px;
  font-weight: 700;
}

.count.value {
  min-width: 38px;
  text-align: right;
}

.ik-toggle {
  flex: 1;
  height: 32px;
  border-radius: 16px;
  background: #eeeeee;
  font-weight: 900;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  cursor: pointer;
}

.ik-toggle:hover {
  background: #ffe8b3;
}

.method-select {
  height: 32px;
  padding: 0 10px;
  border: none;
  border-radius: 16px;
  background: #eeeeee;
  font-weight: 900;
  cursor: pointer;
}

.method-select:hover {
  background: #ffe8b3;
}

.param-row {
  padding-top: 3px;
  padding-bottom: 3px;
  cursor: default;
  font-size: 0.9em;
}

.param-row .row-label {
  font-weight: 700;
}

/* an enabledWhen param whose condition isn't met: dimmed, input disabled */
.param-row.disabled {
  opacity: 0.4;
}

.param-input {
  width: 92px;
  height: 26px;
  padding: 0 8px;
  border: none;
  border-radius: 13px;
  background: #eeeeee;
  font-weight: 700;
  font-size: 0.95em;
}

.mini.defaults {
  width: auto;
  align-self: flex-end;
  padding: 0 10px;
  font-size: 0.75em;
}

.surf-appearance {
  border-top: 1px solid rgba(0, 0, 0, 0.1);
  margin-top: 6px;
  padding-top: 4px;
}

.surf-heading {
  font-size: 0.75em;
  font-weight: 700;
  opacity: 0.6;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.surf-appearance input[type="color"] {
  width: 44px;
  height: 24px;
  padding: 0;
  border: none;
  background: none;
  cursor: pointer;
}

.surf-log {
  height: 130px;
  /* drag the corner to make room for a chatty adapter. flex: none is what
     makes the drag stick: as a flex child in the panel's column it would
     otherwise be shrunk straight back to fit the body's max-height */
  resize: vertical;
  flex: none;
  min-height: 48px;
  max-height: 70vh;
  margin: 0;
  padding: 8px;
  border-radius: 8px;
  background: #f4f4f4;
  overflow-y: scroll;
  font-size: 0.68em;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
