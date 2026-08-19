<!--
  Controls for a raymarched occupancy field: the threshold it is read at, and
  how thick the haze is. Docked at the bottom of the viewport rather than in
  the Surfacer panel, because both are about the thing on screen and want to
  be reachable while looking at it.

  Separate from the TRELLIS flow timeline on purpose: there is no flow here
  and nothing to scrub — one field, sitting over the sketch it came from —
  and folding the two together would mean a panel of controls that are dead
  half the time.
-->
<template>
  <div class="occupancy-controls">
    <div class="row">
      <span class="title">NS2S occupancy</span>
      <span class="readout">{{ readout }}</span>
      <span class="spacer" />
      <button
        class="close"
        title="Drop the field. Nothing about it is kept, so seeing it again means running again."
        @click="emit('close')"
      >
        Clear
      </button>
    </div>

    <div class="row">
      <label
        class="slider"
        title="Voxels at or above this probability turn red — the ones marching cubes would have kept as surface. Moving it shows what a different surface threshold would have included, without running anything again."
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
        title="How much light a unit of distance through a full-probability voxel swallows. Higher makes faint structure visible and saturates the confident regions; lower lets the eye reach further in. Picture only — it does not touch the field or the threshold."
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
      <label
        class="slider"
        title="Gaussian blur over the voxel grid itself, in voxels — not over the picture. The threshold reads the smoothed field, so blurring fattens the surface and closes gaps the sharp field leaves open, at the cost of thin detail. 0 is the field exactly as the network predicted it."
      >
        blur
        <input
          type="range"
          min="0"
          :max="maxBlur"
          step="0.1"
          :value="blur"
          @input="emit('set-blur', Number(($event.target as HTMLInputElement).value))"
        />
        <span class="value">{{ blur > 0 ? `${blur.toFixed(1)}v` : "off" }}</span>
      </label>
    </div>

    <div v-if="belowThreshold" class="row">
      <span class="warn" :title="warning">{{ warning }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { MAX_BLUR_SIGMA } from "../../engine/volumeBlur";

const maxBlur = MAX_BLUR_SIGMA;

const props = defineProps<{
  /** Fields showing — one per part in a part-based run. */
  count: number;
  /** Edge of the grid, for the readout. */
  grid: number;
  /** Highest probability across the fields, 0..1. */
  max: number;
  threshold: number;
  density: number;
  /** Gaussian blur applied to the grid, in voxels; 0 is off. */
  blur: number;
}>();

const emit = defineEmits<{
  (event: "set-threshold", value: number): void;
  (event: "set-density", value: number): void;
  (event: "set-blur", value: number): void;
  (event: "close"): void;
}>();

const readout = computed(() => {
  const fields = props.count === 1 ? "field" : "fields";
  return `${props.count} ${fields} · ${props.grid}³ · max p ${props.max.toFixed(2)}`;
});

/** The case that would have failed as a mesh: no voxel reaches the cut, so
 *  marching cubes has no surface to walk. Here it is just an empty red
 *  region, which is easy to misread as a broken viewer. */
const belowThreshold = computed(() => props.max < props.threshold);
const warning = computed(
  () =>
    `nothing reaches ${props.threshold.toFixed(2)} — a mesh at this ` +
    `threshold would come out empty`,
);
</script>

<style scoped>
.occupancy-controls {
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

.readout {
  padding: 1px 7px;
  border-radius: 9px;
  background: #dce9fb;
  color: #1b4fd8;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.spacer {
  flex: 1;
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

.warn {
  flex: 1;
  color: #b8562a;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
