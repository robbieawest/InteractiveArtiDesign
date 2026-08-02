<template>
  <div class="backdrop" @click.self="$emit('close')">
    <div class="window">
      <button class="close" title="Close the benchmark window (the run keeps going)"
              @click="$emit('close')">✕</button>

      <!-- left: what to run -->
      <aside class="config">
        <h2>Benchmark</h2>

        <section>
          <h3>Sketches</h3>
          <div class="row">
            <input v-model="dirInput" class="grow" placeholder="folder of sketches"
                   title="A path on the machine running the server. Press Enter to scan it; or use Select to pick a folder."
                   @keyup.enter="doScan" />
            <button :disabled="busy"
                    title="Pick a folder of .json sketches and/or subfolders each containing a .gltf"
                    @click="doSelect">Select</button>
          </div>
          <p v-if="store.state.scanned.length" class="hint">
            {{ store.state.scanned.length }} input(s):
            {{ jsonCount }} json, {{ gltfCount }} glTF
          </p>
          <button v-if="store.state.scanned.length" :disabled="busy"
                  title="Import the glTF inputs, articulation included, and write every sketch into the benchmark folder"
                  @click="doPrepare">
            Prepare {{ store.state.scanned.length }} sketch(es)
          </button>
          <p v-if="store.state.id" class="hint mono">{{ store.state.id }}</p>
        </section>

        <section>
          <div class="row">
            <h3 class="grow">Load</h3>
            <button class="small" :disabled="busy"
                    title="Rescan the benchmarks folder"
                    @click="doRefreshBenchmarks">↻</button>
          </div>
          <p v-if="!store.state.benchmarks.length" class="hint">
            No saved benchmarks yet.
          </p>
          <div v-for="bench in store.state.benchmarks" :key="bench.id"
               class="bench" :class="{ current: bench.id === store.state.id }">
            <span class="mono">{{ bench.id }}</span>
            <span class="hint">
              {{ bench.sketches }} sketch(es), {{ bench.results }} surface(s)
            </span>
            <div class="row">
              <button class="small" :disabled="busy"
                      :title="bench.hasProgress
                        ? 'Reopen this benchmark: its sketches, runs, per-cell status and finished surfaces. Resume continues where it stopped.'
                        : 'This folder has sketches but no saved run configuration — it reopens with an empty adapter list.'"
                      @click="doReopen(bench.id)">Open</button>
              <button class="small" :disabled="busy"
                      title="Start a new benchmark from these sketches and run configuration, with none of the surfaces — this folder is left untouched"
                      @click="doCleanCopy(bench.id)">Clean copy</button>
            </div>
          </div>
        </section>

        <section>
          <h3>Adapters</h3>
          <p v-if="!store.state.methods.length" class="hint">
            No surfacing server — start it and reopen this window.
          </p>
          <div v-for="method in store.state.methods" :key="method.name" class="adapter">
            <div class="row">
              <strong class="grow">{{ method.name }}</strong>
              <TallyBar v-if="store.state.runs[method.name]?.length"
                        :tally="store.adapterTally(method.name)" />
              <button class="small" :disabled="busy" @click="store.addRun(method.name)">
                + run
              </button>
            </div>
            <div v-for="run in store.state.runs[method.name] ?? []" :key="run.id"
                 class="run" :class="{ viewing: isViewing(method.name, run.id) }">
              <div class="row">
                <button class="small toggle"
                        :title="collapsed.has(runKey(method.name, run.id))
                          ? 'Show parameters' : 'Hide parameters'"
                        @click="toggle(method.name, run.id)">
                  {{ collapsed.has(runKey(method.name, run.id)) ? "▸" : "▾" }}
                </button>
                <button class="link" title="Show this run's results in the grid"
                        @click="view(method.name, run.id)">
                  {{ run.label }}
                </button>
                <TallyBar :tally="store.runTally(method.name, run.id)" />
                <button class="small" :disabled="busy || isComplete(method.name, run.id)"
                        :title="isComplete(method.name, run.id)
                          ? 'Every sketch of this run is done'
                          : 'Run only this run, from its first unfinished sketch — the rest of the benchmark is left where it is'"
                        @click="store.start({ adapter: method.name, run: run.id })">▶</button>
                <button class="small" :disabled="busy || !hasResults(method.name, run.id)"
                        title="Discard this run's surfaces and run it again from the first sketch — the other runs are left alone"
                        @click="doRerun(method.name, run.id)">↻</button>
                <button class="small" :disabled="busy"
                        @click="store.removeRun(method.name, run.id)">✕</button>
              </div>
              <template v-if="!collapsed.has(runKey(method.name, run.id))">
                <div v-for="param in method.params" :key="param.name" class="param"
                     :class="{ disabled: !paramEnabled(param, run) }" :title="param.help">
                  <label>{{ param.label }}</label>
                  <input v-if="param.type === 'bool'" type="checkbox"
                         :disabled="busy || !paramEnabled(param, run)"
                         :checked="Boolean(run.options[param.name])"
                         @change="setOption(run, param.name, ($event.target as HTMLInputElement).checked)" />
                  <select v-else-if="param.type === 'choice'"
                          :disabled="busy || !paramEnabled(param, run)"
                          :value="run.options[param.name]"
                          @change="setOption(run, param.name, ($event.target as HTMLSelectElement).value)">
                    <option v-for="choice in param.choices" :key="choice" :value="choice">
                      {{ choice }}
                    </option>
                  </select>
                  <input v-else type="number" :min="param.min" :max="param.max"
                         :step="param.step"
                         :disabled="busy || !paramEnabled(param, run)"
                         :value="run.options[param.name]"
                         @change="setOption(run, param.name, Number(($event.target as HTMLInputElement).value))" />
                </div>
              </template>
            </div>
          </div>
        </section>

        <section class="actions">
          <div class="row">
            <button v-if="!running" :disabled="!canStart"
                    :title="resumable
                      ? 'Continue this benchmark. Surfaces already finished are kept and skipped.'
                      : 'Run every sketch through every run of every adapter, one at a time'"
                    @click="store.start()">
              {{ resumable ? "Resume" : "Start" }}
            </button>
            <button v-if="!running" class="small"
                    :disabled="!canStart || overall.done === 0"
                    title="Discard every surface this benchmark has made and run the whole sweep again from the start"
                    @click="doRerun()">
              Rerun all
            </button>
            <template v-else>
              <button :disabled="store.state.phase === 'pausing'"
                      title="Stop after the surface being made now, keeping it. Everything stays loaded, and Resume continues from the next one."
                      @click="store.pause()">
                {{ store.state.phase === "pausing" ? "Pausing…" : "Pause" }}
              </button>
              <button class="small"
                      title="Stop now and kill the method: its processes and every resident worker are terminated, freeing the GPU immediately. The surface in flight is lost and its cell goes back to Queued, so prefer Pause when it is worth waiting for."
                      @click="store.stop()">Stop</button>
            </template>
          </div>
          <p class="hint">{{ store.state.message }}</p>
          <div class="overall" title="Finished surfaces across every run of every adapter">
            <TallyBar :tally="overall" wide />
          </div>
          <p v-if="store.state.error" class="error">{{ store.state.error }}</p>
        </section>
      </aside>

      <!-- right: the grid -->
      <main class="grid-pane">
        <header class="grid-head">
          <span v-if="store.state.viewing">
            {{ store.state.viewing.adapter }} / {{ store.state.viewing.run }}
          </span>
          <span v-else class="hint">Add a run to see per-sketch status</span>
          <label class="edit-toggle"
                 :class="{ disabled: running }"
                 :title="running
                   ? 'Not while a benchmark is running — editing an input mid-sweep would break the comparison'
                   : 'Click a thumbnail to edit that sketch in the editor. Saving discards any surfaces already made from it.'">
            <input type="checkbox" :checked="store.state.editMode"
                   :disabled="running"
                   @change="store.state.editMode = ($event.target as HTMLInputElement).checked" />
            Edit sketches
          </label>
        </header>
        <div class="grid">
          <figure v-for="sketch in store.state.sketches" :key="sketch.name"
                  class="cell"
                  :class="{ active: store.state.active === sketch.name,
                            editable: store.state.editMode }"
                  :title="store.state.editMode
                    ? `Edit ${sketch.name} in the editor`
                    : `Open ${sketch.name} in the editor`"
                  @click="$emit('open', sketch.name)">
            <img :src="sketch.thumbnail" :alt="sketch.name" />
            <figcaption>
              <span class="name">{{ sketch.name }}</span>
              <span class="state" :class="statusOf(sketch.name).state">
                {{ label(statusOf(sketch.name)) }}
              </span>
              <span v-if="statusOf(sketch.name).parts.length" class="hint">
                {{ statusOf(sketch.name).parts.length }} part(s)
              </span>
              <div v-if="statusOf(sketch.name).state === 'surfacing'" class="bar">
                <div :style="{ width: `${statusOf(sketch.name).progress * 100}%` }" />
              </div>
              <span v-if="statusOf(sketch.name).error" class="error"
                    :title="statusOf(sketch.name).error">failed</span>
            </figcaption>
          </figure>
          <p v-if="!store.state.sketches.length" class="hint empty">
            Select a folder and prepare it to populate the grid.
          </p>
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import * as store from "../benchmark/store";
import type { BenchRun, CellStatus } from "../benchmark/store";
import type { MethodParam } from "../surfacing/client";
import TallyBar from "./TallyBar.vue";

defineEmits<{ close: []; open: [sketch: string] }>();

const dirInput = ref("SampleModels");
// collapsed runs, by adapter+run; a sweep with many runs is unreadable
// otherwise, and the parameters only matter while you are editing them
const collapsed = reactive(new Set<string>());
const overall = computed(() => store.overallTally());
const running = computed(
  () => store.state.phase === "running" || store.state.phase === "pausing",
);
const busy = computed(() => running.value ||
  store.state.phase === "preparing" || store.state.phase === "loading");
const canStart = computed(
  () => store.state.sketches.length > 0 &&
    Object.values(store.state.runs).some((runs) => runs.length > 0),
);
// something has already been surfaced, so Start would continue rather than
// begin — worth saying, since it silently skips the finished cells
const resumable = computed(() => overall.value.done > 0 &&
  overall.value.done < overall.value.total);
const jsonCount = computed(
  () => store.state.scanned.filter((e) => e.kind === "json").length,
);
const gltfCount = computed(
  () => store.state.scanned.filter((e) => e.kind === "gltf").length,
);

onMounted(() => {
  if (!store.state.methods.length) void store.loadMethods().catch(() => {});
  void store.refreshBenchmarks().catch(() => {});
});

function doRefreshBenchmarks(): void {
  void store.refreshBenchmarks().catch((exc) => {
    store.state.error = exc instanceof Error ? exc.message : String(exc);
  });
}

function doReopen(id: string): void {
  void store.reopen(id);
}

/** Same sketches and run configuration, new folder, no surfaces — then open
 *  it, since a copy you cannot see is not much use. */
function doCleanCopy(id: string): void {
  void store
    .cleanCopy(id)
    .then((created) => store.reopen(created))
    .catch((exc) => {
      store.state.error = exc instanceof Error ? exc.message : String(exc);
    });
}

function hasResults(adapter: string, runId: string): boolean {
  return store.runTally(adapter, runId).done > 0;
}

function isComplete(adapter: string, runId: string): boolean {
  const { done, total } = store.runTally(adapter, runId);
  return total > 0 && done === total;
}

/** Rerun the whole bench, or one run of it. Confirmed first: the surfaces it
 *  drops took hours to make, and the button sits next to Resume. */
function doRerun(adapter?: string, runId?: string): void {
  const only = adapter && runId ? { adapter, run: runId } : undefined;
  const what = only ? `${only.adapter} / ${only.run}` : "every run";
  const count = only
    ? store.runTally(only.adapter, only.run).done
    : overall.value.done;
  if (!confirm(`Rerun ${what}? ${count} finished surface(s) will be redone.`)) {
    return;
  }
  void store.rerun(only);
}

/** Pick a folder. The browser hands over every file beneath it as a lazy
 *  handle and never reveals an absolute path, so these are read client-side
 *  rather than scanned by the server. */
function doSelect(): void {
  const input = document.createElement("input");
  input.type = "file";
  // non-standard but universally supported, and the only way to get a folder
  input.webkitdirectory = true;
  input.multiple = true;
  input.onchange = () => {
    const files = [...(input.files ?? [])];
    if (files.length === 0) return;
    try {
      store.selectFolder(files);
      dirInput.value = store.state.sourceDir;
    } catch (exc) {
      store.state.error = exc instanceof Error ? exc.message : String(exc);
    }
  };
  input.click();
}

/** Scan a path typed into the field — the server reads it, so this reaches
 *  folders outside anything the picker can offer. */
async function doScan(): Promise<void> {
  if (!dirInput.value.trim()) return;
  try {
    await store.scan(dirInput.value);
  } catch (exc) {
    store.state.error = exc instanceof Error ? exc.message : String(exc);
  }
}

function doPrepare(): void {
  void store.prepare();
}

function statusOf(sketch: string): CellStatus {
  const viewing = store.state.viewing;
  if (!viewing) {
    return { state: "pending", progress: 0, message: "", parts: [] };
  }
  return store.cellStatus(viewing.adapter, viewing.run, sketch);
}

function label(status: CellStatus): string {
  switch (status.state) {
    case "pending":
      return "Queued";
    case "initializing":
      return "Initializing…";
    case "surfacing":
      return `Surfacing… ${Math.round(status.progress * 100)}%`;
    case "done":
      return "Completed";
    case "error":
      return "Error";
  }
}

function runKey(adapter: string, run: string): string {
  return `${adapter} ${run}`;
}

function toggle(adapter: string, run: string): void {
  const key = runKey(adapter, run);
  if (collapsed.has(key)) collapsed.delete(key);
  else collapsed.add(key);
}

function isViewing(adapter: string, run: string): boolean {
  return (
    store.state.viewing?.adapter === adapter && store.state.viewing.run === run
  );
}

function view(adapter: string, run: string): void {
  void store.setViewing(adapter, run);
}

function paramEnabled(param: MethodParam, run: BenchRun): boolean {
  if (!param.enabledWhen) return true;
  return run.options[param.enabledWhen.param] === param.enabledWhen.equals;
}

function setOption(
  run: BenchRun,
  name: string,
  value: number | boolean | string,
): void {
  run.options[name] = value;
}
</script>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  z-index: 50;
}
.window {
  position: relative;
  margin: 3vh 3vw;
  flex: 1;
  display: flex;
  background: #f7f7f7;
  border: 1px solid #bbb;
  border-radius: 6px;
  overflow: hidden;
}
.close {
  position: absolute;
  top: 8px;
  left: 8px;
  width: 26px;
  height: 26px;
  border: 1px solid #bbb;
  background: #fff;
  border-radius: 4px;
  cursor: pointer;
  z-index: 1;
}
.config {
  width: 320px;
  flex: 0 0 320px;
  padding: 44px 12px 12px;
  border-right: 1px solid #ddd;
  /* min-height:0 is what actually lets a flex child scroll instead of
     stretching its parent — without it both panes grow and the window does
     the scrolling for them */
  min-height: 0;
  overflow-y: auto;
  font-size: 12px;
}
.config h2 {
  margin: 0 0 12px;
  font-size: 14px;
}
.config h3 {
  margin: 12px 0 6px;
  font-size: 12px;
  text-transform: uppercase;
  color: #666;
}
.row {
  display: flex;
  gap: 6px;
  align-items: center;
}
.grow {
  flex: 1;
  min-width: 0;
}
.adapter {
  margin-bottom: 10px;
}
.run {
  margin: 4px 0 4px 8px;
  padding: 4px 6px;
  border-left: 2px solid #ddd;
}
.run.viewing {
  border-left-color: #7b4bd8;
  background: #f0ebfa;
}
.param {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  margin: 2px 0;
}
.param.disabled {
  opacity: 0.45;
}
.param input[type="number"],
.param select {
  width: 80px;
}
.bench {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  width: 100%;
  margin: 2px 0;
  padding: 4px 6px;
  text-align: left;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 3px;
}
.bench.current {
  border-color: #7b4bd8;
  background: #f0ebfa;
}
.bench .hint {
  margin: 0;
}
.link {
  border: none;
  background: none;
  padding: 0;
  cursor: pointer;
  text-decoration: underline;
  flex: 1;
  text-align: left;
}
.small {
  padding: 0 5px;
  font-size: 11px;
}
.hint {
  color: #777;
  margin: 4px 0;
}
.mono {
  font-family: monospace;
}
.error {
  color: #b00;
}
.actions {
  border-top: 1px solid #ddd;
  padding-top: 10px;
}
.overall {
  margin-top: 6px;
}
.toggle {
  width: 18px;
  padding: 0;
}
.grid-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}
.grid-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 12px 8px;
  font-size: 12px;
  border-bottom: 1px solid #ddd;
}
.edit-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  user-select: none;
}
.edit-toggle.disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.cell.editable {
  border-style: dashed;
  border-color: #7b4bd8;
}
.cell.editable:hover {
  background: #f6f2fd;
}
.grid {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
  align-content: start;
}
.cell {
  margin: 0;
  border: 1px solid #ddd;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  overflow: hidden;
}
.cell.active {
  border-color: #7b4bd8;
  box-shadow: 0 0 0 2px #e5d9fb;
}
.cell img {
  display: block;
  width: 100%;
  aspect-ratio: 1;
  object-fit: contain;
  background: #fdfdfd;
}
figcaption {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px;
  font-size: 11px;
}
.name {
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.state.done {
  color: #227a22;
}
.state.error {
  color: #b00;
}
.state.surfacing {
  color: #7b4bd8;
}
.bar {
  height: 3px;
  background: #eee;
  border-radius: 2px;
  overflow: hidden;
}
.bar div {
  height: 100%;
  background: #7b4bd8;
}
.empty {
  grid-column: 1 / -1;
}
</style>
