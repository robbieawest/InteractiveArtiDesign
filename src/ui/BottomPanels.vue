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
              :title="
                exploded
                  ? 'Bring all parts back to their original placement'
                  : 'Push all parts away from each other'
              "
              @click="$emit('toggle-explode')"
            >
              {{ exploded ? "Collapse" : "Explode" }}
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

        <!-- Articulations -->
        <template v-else>
          <div class="empty">
            Articulations (sliding/revolute joints) are not implemented yet.
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
import type { Part, Pose } from "../core/types";

export type PanelName = "parts" | "poses" | "articulations";

defineProps<{
  expanded: PanelName | null;
  parts: Part[];
  poses: Pose[];
  activePartId: string | null;
  strokeCounts: Record<string, number>;
  exploded: boolean;
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
}>();

const panels: PanelName[] = ["articulations", "poses", "parts"];
const panelLabels: Record<PanelName, string> = {
  parts: "Parts",
  poses: "Poses",
  articulations: "Articulations",
};
const headerTooltips: Record<PanelName, string> = {
  parts: "Part segmentation — group strokes into parts with the Segment tool",
  poses: "Saved poses — snapshots of every stroke's placement",
  articulations: "Joints between parts (coming later)",
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
</style>
