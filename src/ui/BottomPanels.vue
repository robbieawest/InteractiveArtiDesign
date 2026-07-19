<template>
  <div class="panels">
    <div v-for="panel in panels" :key="panel" class="panel">
      <div v-if="expanded === panel" class="panel-body">
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
              <option v-for="method in methods" :key="method" :value="method">
                {{ method }}
              </option>
            </select>
          </template>
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
import type { Joint, JointDofName, Part, Pose } from "../core/types";
import { JOINT_DOF_NAMES, dofUnlocked, jointKindLabel } from "../core/types";

export type PanelName = "parts" | "poses" | "articulations" | "surfacer";

defineProps<{
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
  /** Adapter names from the surfacing server; empty = server offline. */
  methods: string[];
  surfacingMethod: string;
  surfacing: boolean;
  surfacingProgress: number;
  hasSurface: boolean;
}>();

defineEmits<{
  "set-expanded": [panel: PanelName | null];
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
  surface: [];
  "clear-surface": [];
}>();

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
  bottom: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 260px;
}

.panel {
  display: flex;
  flex-direction: column;
}

.panel-header {
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
  overflow-y: auto;
  filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.12));
  display: flex;
  flex-direction: column;
  gap: 6px;
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
</style>
