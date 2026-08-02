<template>
  <span class="tally" :class="{ wide }" :title="`${tally.done} of ${tally.total} surfaces completed`">
    <span class="track"><span class="fill" :style="{ width: `${percent}%` }" /></span>
    <span class="count">{{ tally.done }}/{{ tally.total }}</span>
  </span>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { Tally } from "../benchmark/store";

const props = defineProps<{ tally: Tally; wide?: boolean }>();

const percent = computed(() =>
  props.tally.total === 0 ? 0 : (props.tally.done / props.tally.total) * 100,
);
</script>

<style scoped>
.tally {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: #666;
}
.track {
  display: block;
  width: 38px;
  height: 4px;
  background: #e2e2e2;
  border-radius: 2px;
  overflow: hidden;
}
.wide .track {
  width: 100%;
  height: 6px;
}
.wide {
  display: flex;
  font-size: 11px;
}
.wide .track {
  flex: 1;
}
.fill {
  display: block;
  height: 100%;
  background: #7b4bd8;
  transition: width 0.2s;
}
.count {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
</style>
