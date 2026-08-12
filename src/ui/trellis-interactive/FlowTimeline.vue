<!--
  Controls for the interactive TRELLIS viewer: a scrubber over the two flow
  stages, the threshold the occupancy is read at, and the raw/processed mesh
  switch. Docked at the bottom of the viewport rather than in the Surfacer
  panel, because everything here is about the thing on screen and wants to be
  reachable while looking at it.

  Deliberately coupled to TRELLIS — this is the one adapter with a flow to
  watch, and pretending otherwise would mean an abstraction over a single
  case. It lives in its own directory for that reason.
-->
<template>
  <div class="flow-timeline">
    <div class="row">
      <span class="title">{{ hasFrames ? "TRELLIS flow" : "TRELLIS result" }}</span>
      <span v-if="hasFrames" class="stage" :class="stage">{{ stageLabel }}</span>
      <span class="spacer" />
      <label
        v-if="hasFrames && canOverlaySketch"
        class="toggle"
        title="Draw the strokes over the result in orange, so you can see where the model added volume the drawing never had and where it missed the drawing"
      >
        <input
          type="checkbox"
          :checked="showSketchOverlay"
          @change="emit('set-sketch-overlay', ($event.target as HTMLInputElement).checked)"
        />
        sketch over result
      </label>
      <label
        v-if="hasRaw"
        class="toggle"
        title="Show the mesh as FlexiCubes produced it, before simplification, hole filling and fragment removal — the stages that delete geometry"
      >
        <input
          type="checkbox"
          :checked="showRaw"
          @change="emit('set-show-raw', ($event.target as HTMLInputElement).checked)"
        />
        unprocessed mesh
      </label>
      <button
        class="close"
        title="Clear the surface and leave this view. The captured frames are dropped — nothing about them is kept, so seeing them again means running again."
        @click="emit('close')"
      >
        Clear
      </button>
    </div>

    <div v-if="hasFrames" class="row">
      <button
        class="step"
        title="Previous step"
        :disabled="position <= 0"
        @click="emit('set-position', position - 1)"
      >
        ‹
      </button>
      <input
        class="scrub"
        type="range"
        min="0"
        :max="Math.max(0, length - 1)"
        :value="position"
        :title="scrubTitle"
        @input="emit('set-position', Number(($event.target as HTMLInputElement).value))"
      />
      <button
        class="step"
        title="Next step"
        :disabled="position >= length - 1"
        @click="emit('set-position', position + 1)"
      >
        ›
      </button>
      <span class="counter">{{ stepLabel }}</span>
    </div>

    <div v-if="hasFrames" class="row">
      <label
        class="slider"
        title="Voxels at or above this turn red — the ones the pipeline would keep. 0.50 is its own cut (the stored value is sigmoid(logit) and TRELLIS keeps logit > 0), so moving it shows what a different decision would have included. Occupancy only; the latent region measures something with no threshold in it."
      >
        threshold
        <input
          type="range"
          min="0.02"
          max="0.98"
          step="0.01"
          :value="threshold"
          @input="emit('set-threshold', Number(($event.target as HTMLInputElement).value))"
        />
        <span class="value">{{ threshold.toFixed(2) }}</span>
      </label>
      <label
        class="slider"
        title="How much light a unit of distance through a full-value voxel swallows. Higher makes faint structure visible and saturates the solid regions; lower lets the eye reach further into the object. Picture only — it does not touch the stored field or the threshold."
      >
        density
        <input
          type="range"
          min="0.2"
          max="3"
          step="0.05"
          :value="density"
          @input="emit('set-density', Number(($event.target as HTMLInputElement).value))"
        />
        <span class="value">{{ density.toFixed(2) }}</span>
      </label>
      <span class="hint" :title="viewHint">{{ viewHint }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  /** Total scrubbable positions: structure steps then latent steps. */
  length: number;
  /** How many of those belong to the structure stage. */
  structureSteps: number;
  position: number;
  threshold: number;
  density: number;
  showRaw: boolean;
  hasRaw: boolean;
  showSketchOverlay: boolean;
  /** False when the run was not fitted to the sketch: the mesh is then in
   *  the model's own frame and laying the strokes over it means nothing. */
  canOverlaySketch: boolean;
  /** False when the run kept only the unprocessed mesh: there is no flow to
   *  scrub, so this is just the mesh switch and the viewport is untouched. */
  hasFrames: boolean;
  /** Conditioning view count, for the which-view-guided-this-step readout. */
  viewCount: number;
}>();

const emit = defineEmits<{
  (event: "set-position", value: number): void;
  (event: "set-threshold", value: number): void;
  (event: "set-density", value: number): void;
  (event: "set-show-raw", value: boolean): void;
  (event: "set-sketch-overlay", value: boolean): void;
  (event: "close"): void;
}>();

const stage = computed(() =>
  props.position < props.structureSteps ? "structure" : "latent",
);

const stageLabel = computed(() =>
  stage.value === "structure"
    ? "sparse structure — sampling occupancy"
    : "structured latent — occupancy is now fixed",
);

const stepLabel = computed(() => {
  if (props.length === 0) return "—";
  const within =
    stage.value === "structure"
      ? props.position
      : props.position - props.structureSteps;
  const total =
    stage.value === "structure"
      ? props.structureSteps
      : props.length - props.structureSteps;
  return `step ${within + 1} / ${total}`;
});

/** Which conditioning view guided this step. In `stochastic` mode the
 *  sampler walks the views in order — `cond_indices = arange(steps) %
 *  num_images` — so this is exact, not a guess. */
const viewHint = computed(() => {
  if (props.viewCount === 0 || props.length === 0) return "";
  const within =
    stage.value === "structure"
      ? props.position
      : props.position - props.structureSteps;
  return `guided by view ${(within % props.viewCount) + 1} of ${props.viewCount}`;
});

const scrubTitle = computed(
  () =>
    `Scrub the flow. The first ${props.structureSteps} steps sample the ` +
    `occupancy grid; the rest sample the latent on the voxels it chose.`,
);
</script>

<style scoped>
.flow-timeline {
  position: absolute;
  left: 50%;
  bottom: 16px;
  transform: translateX(-50%);
  width: min(720px, calc(100% - 32px));
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(250, 250, 250, 0.94);
  border: 1px solid #c8c8c8;
  box-shadow: 0 3px 14px rgba(0, 0, 0, 0.18);
  font-size: 12px;
  color: #333;
}

.row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.title {
  font-weight: 600;
}

.stage {
  padding: 1px 7px;
  border-radius: 9px;
  font-size: 11px;
}

.stage.structure {
  background: #dce9fb;
  color: #1b4fd8;
}

.stage.latent {
  background: #e6e0f7;
  color: #5a3fbe;
}

.spacer {
  flex: 1;
}

.scrub {
  flex: 1;
}

.counter {
  min-width: 84px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.step {
  width: 24px;
}

.slider {
  display: flex;
  align-items: center;
  gap: 5px;
}

.slider input {
  width: 96px;
}

.value {
  font-variant-numeric: tabular-nums;
  min-width: 30px;
}

.hint {
  flex: 1;
  text-align: right;
  color: #777;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.toggle {
  display: flex;
  align-items: center;
  gap: 4px;
}
</style>
