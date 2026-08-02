// Benchmark state and the runner that walks it.
//
// This is a module-level singleton on purpose: the benchmark has to survive
// the window being closed, and clicking a thumbnail to inspect a sketch in the
// editor must not disturb a run in flight. Vue components read `state` and
// render; nothing about the run lives in a component.
//
// Execution is strictly serial — adapter, then run, then sketch — because the
// methods it drives are GPU-bound and two at once would just contend.

import { reactive } from "vue";
import { deserializeDocument } from "../core/serialization";
import type { DocumentJson } from "../core/serialization";
import { importSketchLabGltf, parseGlb } from "../engine/importSketchLab";
import { serializeDocument } from "../core/serialization";
import { SketchDocument } from "../core/SketchDocument";
import { buildSurfacingSketch, runSurfacingJob } from "../surfacing/client";
import type { MethodInfo, SurfacingSketch } from "../surfacing/client";
import * as api from "../surfacing/benchmarkClient";
import type { BenchmarkSummary, SourceEntry } from "../surfacing/benchmarkClient";
import { renderThumbnail } from "../engine/benchmarkThumbnail";

/** A scanned input, from either source.
 *
 *  A folder picked in the browser yields File handles and no absolute path —
 *  the browser will not reveal one — so those entries are read client-side and
 *  `path` is empty. A typed path is scanned by the server instead and has no
 *  Files. Either way nothing is read until `prepare()` needs it: a File is a
 *  lazy reference, not the bytes. */
export type BenchEntry = SourceEntry & { file?: File; bin?: File };

export type SketchState =
  | "pending"
  | "initializing"
  | "surfacing"
  | "done"
  | "error";

/** One (adapter, run, sketch) cell: what the grid renders under a thumbnail. */
export interface CellStatus {
  state: SketchState;
  progress: number;
  message: string;
  error?: string;
  /** Names of parts whose geometry has arrived, for the "3/8 parts" readout. */
  parts: string[];
}

/** One parameter permutation for one adapter, run and reported separately. */
export interface BenchRun {
  id: string;
  label: string;
  options: Record<string, number | boolean | string>;
}

export interface BenchSketch {
  name: string;
  /** Line drawing (in the pose it was stored in) plus parts and joints —
   *  also the surfacing payload. */
  sketch: SurfacingSketch;
  thumbnail: string;
}

export interface BenchmarkState {
  id: string | null;
  sourceDir: string;
  scanned: BenchEntry[];
  sketches: BenchSketch[];
  methods: MethodInfo[];
  /** Adapter name -> its runs. An adapter with no runs is not selected. */
  runs: Record<string, BenchRun[]>;
  /** Which adapter/run the grid is currently showing status for. */
  viewing: { adapter: string; run: string } | null;
  status: Record<string, CellStatus>;
  /** `paused` and `idle` differ only in what the button says: both hold a
   *  complete, resumable bench, and Start picks up from the first cell that
   *  is not already done. `pausing` is the gap between asking to pause and
   *  the surface in flight finishing. */
  phase:
    | "idle"
    | "preparing"
    | "loading"
    | "running"
    | "pausing"
    | "paused"
    | "finished"
    | "error";
  message: string;
  error: string | null;
  /** Sketch name currently being surfaced, so the grid can highlight it. */
  active: string | null;
  /** When on, clicking a thumbnail opens that sketch for editing rather than
   *  just viewing it. Off while a sweep runs — changing an input mid-sweep
   *  would leave the run comparing sketches that no longer match. */
  editMode: boolean;
  /** The sketch handed to the editor, while it is out there being edited. */
  editing: string | null;
  /** Benchmark folders on disk, for the reopen picker. */
  benchmarks: BenchmarkSummary[];
}

export const state = reactive<BenchmarkState>({
  id: null,
  sourceDir: "",
  scanned: [],
  sketches: [],
  methods: [],
  runs: {},
  viewing: null,
  status: {},
  phase: "idle",
  message: "",
  error: null,
  active: null,
  editMode: false,
  editing: null,
  benchmarks: [],
});

/** Geometry published by the cell being surfaced *right now*, and nothing
 *  else. A sweep produces hundreds of megabytes of mesh; none of it is cached
 *  here, because the server has already written every finished surface to
 *  benchmarks/<id>/<adapter>/<run>/<sketch>.glb. Anything that needs one reads
 *  it back, uses it, and drops it — a thumbnail is a 256px data URL, which is
 *  the only representation worth keeping alive.
 *
 *  Keyed by the name the adapter published under, which is what distinguishes
 *  the two kinds of partial: parts have distinct names and accumulate, while a
 *  whole-object method re-publishing its converging surface reuses one name
 *  and so replaces itself. Cleared when the cell ends, so at most one cell's
 *  worth of mesh is resident. Kept out of `state` because ArrayBuffers have no
 *  place in a reactive object (or in progress.json). */
const partials = new Map<string, ArrayBuffer>();
/** Which cell `partials` belongs to, so a stale one is never drawn. */
let partialsKey: string | null = null;

/** Release the in-flight geometry, optionally re-arming it for a new cell.
 *  Called on both ends of a cell and whenever the bench is torn down, so no
 *  mesh outlives the job that produced it. */
function clearPartials(key: string | null = null): void {
  partials.clear();
  partialsKey = key;
}

let abort: AbortController | null = null;
/** Set by pause(); read by the sweep between cells. Pausing is deliberately
 *  graceful — stopping kills the method outright, and a NeuVAS cell is three
 *  quarters of an hour, so the surface in flight is allowed to finish and be
 *  saved rather than thrown away. */
let pauseRequested = false;

export function cellKey(adapter: string, run: string, sketch: string): string {
  return `${adapter}\u0000${run}\u0000${sketch}`;
}

export function cellStatus(
  adapter: string,
  run: string,
  sketch: string,
): CellStatus {
  return (
    state.status[cellKey(adapter, run, sketch)] ?? {
      state: "pending",
      progress: 0,
      message: "",
      parts: [],
    }
  );
}

function setCell(key: string, patch: Partial<CellStatus>): void {
  const current = state.status[key] ?? {
    state: "pending" as SketchState,
    progress: 0,
    message: "",
    parts: [],
  };
  state.status[key] = { ...current, ...patch };
}

// --- source folder --------------------------------------------------------

export async function scan(dir: string): Promise<void> {
  state.error = null;
  const result = await api.scanSource(dir);
  state.sourceDir = result.dir;
  state.scanned = result.entries;
}

/** Take the contents of a folder picked in the browser (a directory input
 *  hands over every file beneath it, each tagged with its relative path) and
 *  pick out the same two shapes the server-side scan looks for: loose .json
 *  sketch documents at the top level, and subfolders holding a .gltf. */
export function selectFolder(files: File[]): void {
  state.error = null;
  const relative = (file: File) =>
    (file as File & { webkitRelativePath?: string }).webkitRelativePath ||
    file.name;

  const root = relative(files[0] ?? new File([], "")).split("/")[0] ?? "";
  const entries: BenchEntry[] = [];
  // subfolder name -> the model file and its sidecar buffer
  const folders = new Map<string, { model?: File; bin?: File }>();

  for (const file of files) {
    const segments = relative(file).split("/");
    const name = segments[segments.length - 1];
    const lower = name.toLowerCase();

    if (segments.length === 2 && lower.endsWith(".json")) {
      entries.push({ name: name.replace(/\.json$/i, ""), kind: "json", path: "", file });
      continue;
    }
    if (segments.length < 2) continue;
    const folder = segments[segments.length - 2];
    const slot = folders.get(folder) ?? {};
    if (lower.endsWith(".gltf") || lower.endsWith(".glb")) slot.model = file;
    else if (lower.endsWith(".bin")) slot.bin = file;
    folders.set(folder, slot);
  }

  for (const [folder, slot] of folders) {
    if (slot.model) {
      entries.push({ name: folder, kind: "gltf", path: "", file: slot.model, bin: slot.bin });
    }
  }

  entries.sort((a, b) => a.name.localeCompare(b.name));
  state.sourceDir = root;
  state.scanned = entries;
}

export async function loadMethods(): Promise<void> {
  const { fetchMethods } = await import("../surfacing/client");
  state.methods = await fetchMethods();
}

/** Turn every scanned entry into a stored sketch document.
 *
 *  glTF entries go through the same importer the editor uses, articulation
 *  included: a benchmark sketch is a full document, so it can be opened,
 *  reposed, and saved back with the pose it is meant to be surfaced in. json
 *  entries pass through as they are. Everything is written to
 *  benchmarks/<id>/sketches/, so re-selecting that folder later reruns the
 *  same set with no preprocessing at all. */
export async function prepare(): Promise<void> {
  state.phase = "preparing";
  state.error = null;
  state.sketches = [];
  clearPartials();
  state.status = {};
  state.id = api.newBenchmarkId();

  try {
    for (const entry of state.scanned) {
      state.message = `preparing ${entry.name}`;
      const json = await documentFor(entry);
      await api.saveSketch(state.id, entry.name, json);
      await addSketch(entry.name, json);
    }
    state.phase = "idle";
    state.message = `${state.sketches.length} sketch(es) ready`;
    await persist();
    await refreshBenchmarks().catch(() => {});
  } catch (exc) {
    state.phase = "error";
    state.error = exc instanceof Error ? exc.message : String(exc);
  }
}

async function documentFor(entry: BenchEntry): Promise<DocumentJson> {
  // one accessor for both origins: a picked File reads locally, a scanned
  // path comes back through the server
  const bytesOf = (e: BenchEntry) =>
    e.file ? e.file.arrayBuffer() : api.readSourceFile(e.path);

  if (entry.kind === "json") {
    const bytes = await bytesOf(entry);
    return JSON.parse(new TextDecoder().decode(bytes)) as DocumentJson;
  }

  const raw = await bytesOf(entry);
  const modelName = (entry.file?.name ?? entry.path).toLowerCase();
  const source = modelName.endsWith(".glb")
    ? parseGlb(raw)
    : {
        json: JSON.parse(new TextDecoder().decode(raw)) as unknown,
        bin: await binFor(entry, raw),
      };
  const imported = importSketchLabGltf(source.json, source.bin);

  const doc = new SketchDocument();
  for (const part of imported.parts) doc.addPart(part);
  for (const joint of imported.joints) doc.addJoint(joint);
  for (const stroke of imported.strokes) doc.addStroke(stroke);
  return serializeDocument(doc);
}

/** The .bin a .gltf references. A picked folder already handed us the sibling
 *  file; for a server path we read the buffer uri out of the glTF and ask for
 *  the file beside it. */
async function binFor(
  entry: BenchEntry,
  raw: ArrayBuffer,
): Promise<ArrayBuffer | undefined> {
  if (entry.file) return entry.bin?.arrayBuffer();

  const gltf = JSON.parse(new TextDecoder().decode(raw)) as {
    buffers?: { uri?: string }[];
  };
  const uri = gltf.buffers?.[0]?.uri;
  if (!uri || uri.startsWith("data:")) return undefined;
  const folder = entry.path.slice(0, entry.path.lastIndexOf("/"));
  return api.readSourceFile(`${folder}/${decodeURIComponent(uri)}`);
}

async function addSketch(name: string, json: DocumentJson): Promise<void> {
  const doc = deserializeDocument(json);
  const sketch = buildSurfacingSketch(doc);
  state.sketches.push({
    name,
    sketch,
    thumbnail: await renderThumbnail(sketch),
  });
}

/** The benchmark folders on disk, for the reopen picker. */
export async function refreshBenchmarks(): Promise<void> {
  state.benchmarks = await api.listBenchmarks();
}

/** Reopen a benchmark from disk: its sketches, its run configuration, how far
 *  it got, and the surfaces it already made.
 *
 *  Everything needed is already written as a run proceeds — sketches/ holds
 *  the preprocessed inputs, progress.json the runs and per-cell status,
 *  <adapter>/<run>/<sketch>.glb the finished surfaces — so this needs no
 *  cooperation from whatever session produced it. A bench whose server was
 *  killed mid-cell reopens with that cell back at Queued; nothing else is
 *  lost, because a cell is only ever recorded once it has finished.
 *
 *  Comes back paused when there is work left, so the button says Resume. */
export async function reopen(benchmarkId: string): Promise<void> {
  if (state.phase === "running" || state.phase === "pausing") return;
  state.phase = "loading";
  state.error = null;
  state.message = `loading ${benchmarkId}`;

  try {
    state.id = benchmarkId;
    state.sketches = [];
    state.scanned = [];
    state.active = null;
    state.editing = null;
    state.editMode = false;
    clearPartials();

    const saved = (await api.loadProgress(benchmarkId)) as Partial<
      BenchmarkState
    > | null;
    state.runs = saved?.runs ?? {};
    state.status = saved?.status ?? {};
    state.sourceDir = saved?.sourceDir ?? "";
    state.viewing = saved?.viewing ?? null;
    resetInFlight();

    for (const name of await api.listSketches(benchmarkId)) {
      state.message = `loading ${name}`;
      await addSketch(name, await api.readSketch(benchmarkId, name));
    }
    if (!state.viewing) state.viewing = firstViewing();
    await refreshViewedThumbnails();

    // `paused` says Resume, which only makes sense with something to resume
    // from — a clean copy has runs configured but nothing done, so it opens
    // idle and its button says Start
    const { done, total } = overallTally();
    state.phase =
      total > 0 && done === total ? "finished" : done > 0 ? "paused" : "idle";
    state.message =
      total === 0
        ? `${state.sketches.length} sketch(es), no runs configured`
        : done === 0
          ? `${state.sketches.length} sketch(es), ${total} surface(s) to make`
          : `${done}/${total} surfaces — ${
              done === total ? "complete" : "resume to continue"
            }`;
  } catch (exc) {
    state.phase = "error";
    state.error = exc instanceof Error ? exc.message : String(exc);
  }
}

/** Redraw every thumbnail against the viewed run, reading each finished
 *  surface back from disk one at a time.
 *
 *  Deliberately sequential and deliberately uncached: the mesh is alive only
 *  between the read and the render, so the peak cost of showing a run of any
 *  size is one glb, not the whole run. A cell with nothing (queued, failed,
 *  or a run never started) renders as bare strokes, which is what makes
 *  switching runs show that run rather than a mixture. */
async function refreshViewedThumbnails(): Promise<void> {
  const viewing = state.viewing;
  for (const sketch of state.sketches) {
    if (!viewing || !state.id) {
      await refreshThumbnail(sketch, []);
      continue;
    }
    const status = cellStatus(viewing.adapter, viewing.run, sketch.name);
    let surface: ArrayBuffer | null = null;
    if (status.state === "done") {
      surface = await readSurface(
        viewing.adapter,
        viewing.run,
        sketch.name,
      );
    }
    await refreshThumbnail(sketch, surface ? [surface] : []);
  }
}

/** The stored surface for one cell, or null if it is not on disk. Never
 *  cached: callers use the buffer and let it go. */
async function readSurface(
  adapter: string,
  run: string,
  sketch: string,
): Promise<ArrayBuffer | null> {
  if (!state.id) return null;
  try {
    return await api.readResult(state.id, adapter, run, sketch);
  } catch {
    // the file is gone from under us: show the cell without its surface
    // rather than failing the whole load
    return null;
  }
}

/** The viewed run's finished surface for one sketch, for the editor to show
 *  when a thumbnail is clicked. Read on demand, owned by the caller. */
export async function readViewedSurface(
  sketchName: string,
): Promise<ArrayBuffer | null> {
  const viewing = state.viewing;
  if (!viewing) return null;
  if (cellStatus(viewing.adapter, viewing.run, sketchName).state !== "done") {
    return null;
  }
  return readSurface(viewing.adapter, viewing.run, sketchName);
}

/** Show a run in the grid, fetching any surfaces of it not held in memory.
 *
 *  Every thumbnail is redrawn against this run alone — a cell with nothing in
 *  it goes back to bare strokes — so the grid always shows one run's results
 *  and never a mix of what happens to have been rendered last. */
export async function setViewing(adapter: string, run: string): Promise<void> {
  if (isViewed(adapter, run)) return;
  state.viewing = { adapter, run };
  await refreshViewedThumbnails();
}

/** Whether a cell belongs to the run the grid is showing. Thumbnails are
 *  per-sketch, not per-cell, so anything that draws one has to ask first. */
function isViewed(adapter: string, run: string): boolean {
  return state.viewing?.adapter === adapter && state.viewing.run === run;
}

// --- editing --------------------------------------------------------------

/** Hand a sketch to the editor. The window closes while this is out (the
 *  editor is the whole app), and the benchmark itself is untouched. */
export function beginEdit(name: string): void {
  if (state.phase === "running") return;
  state.editing = name;
}

export function cancelEdit(): void {
  state.editing = null;
}

/** Take an edited document back: overwrite the stored sketch, redraw its
 *  thumbnail, and drop whatever had already been surfaced from it.
 *
 *  The document is stored exactly as the editor left it, articulation and all,
 *  so the strokes it carries are in whatever pose it was saved in — that pose
 *  is what the next run surfaces.
 *
 *  Dropping the results is the point — a surface produced from the old strokes
 *  is no longer a result for this sketch, and silently keeping it would put
 *  two different inputs in the same comparison. The cells go back to Queued so
 *  the next run redoes them. */
export async function saveEdit(
  name: string,
  document: DocumentJson,
): Promise<void> {
  if (!state.id) return;
  await api.saveSketch(state.id, name, document);

  const doc = deserializeDocument(document);
  const sketch = state.sketches.find((s) => s.name === name);
  if (sketch) {
    sketch.sketch = buildSurfacingSketch(doc);
    sketch.thumbnail = await renderThumbnail(sketch.sketch);
  }

  for (const [adapter, runs] of Object.entries(state.runs)) {
    for (const run of runs) delete state.status[cellKey(adapter, run.id, name)];
  }
  state.editing = null;
  await persist();
}

// --- runs -----------------------------------------------------------------

export function addRun(adapter: string): BenchRun {
  const method = state.methods.find((m) => m.name === adapter);
  const options: Record<string, number | boolean | string> = {};
  for (const param of method?.params ?? []) options[param.name] = param.default;

  const runs = state.runs[adapter] ?? (state.runs[adapter] = []);
  const run: BenchRun = {
    id: `run-${runs.length + 1}`,
    label: `run-${runs.length + 1}`,
    options,
  };
  runs.push(run);
  if (!state.viewing) state.viewing = { adapter, run: run.id };
  return run;
}

export function removeRun(adapter: string, runId: string): void {
  const runs = state.runs[adapter];
  if (!runs) return;
  state.runs[adapter] = runs.filter((r) => r.id !== runId);
  if (state.runs[adapter].length === 0) delete state.runs[adapter];
  if (state.viewing?.adapter === adapter && state.viewing.run === runId) {
    state.viewing = firstViewing();
  }
}

function firstViewing(): { adapter: string; run: string } | null {
  for (const [adapter, runs] of Object.entries(state.runs)) {
    if (runs.length > 0) return { adapter, run: runs[0].id };
  }
  return null;
}

// --- execution ------------------------------------------------------------

/** One run of one adapter, as the rerun/scope controls name it. */
export interface RunRef {
  adapter: string;
  run: string;
}

/** Walk every adapter, every run, every sketch, one job at a time. Failures
 *  are recorded on the cell and the walk continues — one bad sketch should
 *  not cost the rest of the sweep.
 *
 *  Also the resume path: cells that are already done are skipped, so calling
 *  this on a paused or reopened bench continues where it left off.
 *
 *  `only` narrows the sweep to a single run and changes nothing else: its
 *  finished cells are still skipped, and the rest of the bench is left alone
 *  rather than reset. Pair it with `resetCells` to actually redo a run. */
export async function start(only?: RunRef): Promise<void> {
  if (state.phase === "running" || state.phase === "pausing" || !state.id) {
    return;
  }
  abort = new AbortController();
  pauseRequested = false;
  state.phase = "running";
  state.error = null;
  // a sweep and an open edit session cannot coexist, and the toggle is
  // disabled while running — so clear it here rather than stranding it on
  state.editMode = false;
  state.editing = null;

  try {
    for (const [adapter, runs] of Object.entries(state.runs)) {
      if (only && only.adapter !== adapter) continue;
      for (const run of runs) {
        if (only && only.run !== run.id) continue;
        // the grid follows the sweep: entering a run shows that run, which
        // clears every thumbnail it has nothing for and fills them back in as
        // its cells finish. Viewing another run in the meantime is fine — it
        // holds until the sweep moves on.
        await setViewing(adapter, run.id);
        for (const sketch of state.sketches) {
          if (abort.signal.aborted) return;
          if (pauseRequested) {
            state.phase = "paused";
            state.message = "paused";
            return;
          }
          await runOne(adapter, run, sketch);
          await persist();
        }
      }
    }
    // a single run finishing does not finish the bench: say so, and stay
    // resumable, so the runs left over are still one Start away
    const { done, total } = overallTally();
    state.phase = !only || done === total ? "finished" : "paused";
    state.message = only
      ? `${only.adapter} / ${only.run} complete — ${done}/${total} overall`
      : "benchmark complete";
  } catch (exc) {
    // stop() has already set the phase; a deliberate abort is not a failure
    if (!(exc instanceof DOMException && exc.name === "AbortError")) {
      state.phase = "error";
      state.error = exc instanceof Error ? exc.message : String(exc);
    }
  } finally {
    pauseRequested = false;
    state.active = null;
    resetInFlight();
    await persist();
  }
}

/** Put cells back to Queued so the next sweep redoes them — the whole bench,
 *  or one run of one adapter.
 *
 *  The .glb files on disk are left where they are and simply overwritten when
 *  the cell is redone: nothing reads a surface whose cell is not `done`, so a
 *  reset run shows bare strokes immediately and cannot serve stale geometry.
 *  Refused while a sweep is running, for the obvious reason. */
export async function resetCells(only?: RunRef): Promise<void> {
  if (state.phase === "running" || state.phase === "pausing") return;
  for (const [adapter, runs] of Object.entries(state.runs)) {
    if (only && only.adapter !== adapter) continue;
    for (const run of runs) {
      if (only && only.run !== run.id) continue;
      for (const sketch of state.sketches) {
        delete state.status[cellKey(adapter, run.id, sketch.name)];
      }
    }
  }
  clearPartials();
  state.phase = "idle";
  state.message = only ? `${only.adapter} / ${only.run} reset` : "reset";
  await refreshViewedThumbnails();
  await persist();
}

/** Discard results and run again from the top — the whole bench, or one run.
 *
 *  Distinct from Start, which resumes: this is for a bench that finished, was
 *  paused, or was stopped and should be redone rather than continued (new
 *  parameters, an edited sketch, a method that has changed underneath it). */
export async function rerun(only?: RunRef): Promise<void> {
  if (state.phase === "running" || state.phase === "pausing") return;
  await resetCells(only);
  await start(only);
}

/** Start a new benchmark from this one's sketches, carrying the run
 *  configuration but none of the results.
 *
 *  The source folder is left untouched — this is the non-destructive rerun:
 *  the old surfaces stay comparable on disk while the new ones are made
 *  beside them. Returns the new id; the caller decides whether to open it. */
export async function cleanCopy(sourceId: string): Promise<string> {
  const target = api.newBenchmarkId();
  await api.copySketches(sourceId, target);

  const saved = (await api
    .loadProgress(sourceId)
    .catch(() => null)) as Partial<BenchmarkState> | null;
  if (saved?.runs) {
    await api.saveProgress(target, {
      runs: saved.runs,
      status: {},
      viewing: saved.viewing ?? null,
      sourceDir: saved.sourceDir ?? "",
    });
  }
  await refreshBenchmarks().catch(() => {});
  return target;
}

/** Stop after the surface currently being made, keeping it.
 *
 *  Nothing is discarded: the whole bench stays in memory and Start resumes
 *  from the next cell. Progress is written to disk too, so a paused bench
 *  survives the server being killed as well as the tab being closed. */
export function pause(): void {
  if (state.phase !== "running") return;
  pauseRequested = true;
  state.phase = "pausing";
  state.message = "pausing — finishing the current surface";
}

/** Abandon the surface in flight and stop now.
 *
 *  The abort reaches the server: the job's method processes and every
 *  resident worker are killed, so the GPU is free the moment this returns
 *  rather than whenever the method would have finished. The work in flight is
 *  lost — prefer pause() when the current cell is worth waiting for.
 *  Resumable either way; the abandoned cell goes back to Queued. */
export function stop(): void {
  abort?.abort();
  abort = null;
  pauseRequested = false;
  state.phase = "idle";
  state.message = "stopped";
  state.active = null;
  resetInFlight();
  void persist();
}

/** Put any cell left mid-flight back to Queued.
 *
 *  A cell only ever reaches a terminal state through runOne, so anything
 *  still reading "initializing"/"surfacing" belongs to a job nobody is
 *  watching any more — after a stop, or after reopening a bench whose server
 *  died mid-cell. Deleting the entry is what makes it read Queued. */
function resetInFlight(): void {
  clearPartials();
  for (const [key, status] of Object.entries(state.status)) {
    if (status.state === "initializing" || status.state === "surfacing") {
      delete state.status[key];
    }
  }
}

async function runOne(
  adapter: string,
  run: BenchRun,
  sketch: BenchSketch,
): Promise<void> {
  const key = cellKey(adapter, run.id, sketch.name);
  const done = cellStatus(adapter, run.id, sketch.name);
  if (done.state === "done") return; // resumed benchmark: skip what finished

  state.active = sketch.name;
  clearPartials(key);
  setCell(key, {
    state: "initializing",
    progress: 0,
    message: "submitting",
    parts: [],
    error: undefined,
  });

  try {
    const result = await runSurfacingJob({
      method: adapter,
      sketch: sketch.sketch,
      options: run.options,
      save: {
        benchmarkId: state.id!,
        adapter,
        run: run.id,
        sketch: sketch.name,
      },
      signal: abort?.signal,
      // any word from the server means it is under way; a job sitting at 0%
      // is still surfacing, it just has nothing to report yet
      onProgress: (status) => {
        setCell(key, {
          state: "surfacing",
          progress: status.progress,
          message: status.message || "surfacing",
        });
      },
      onPartial: (name, glb) => {
        // a callback that outlived its cell (a stopped run still finishing on
        // the server) has nowhere to put its bytes
        if (partialsKey !== key) return;
        // same name = a re-published whole-object snapshot, so this overwrites
        // it, and the superseded buffer is dropped rather than accumulated
        const isNew = !partials.has(name);
        partials.set(name, glb);
        if (isNew) {
          const current = cellStatus(adapter, run.id, sketch.name);
          setCell(key, { parts: [...current.parts, name] });
        }
        // only when this cell is the one on screen: the thumbnail belongs to
        // the viewed run, and drawing another run's partials into it would
        // put two runs' surfaces in the same grid
        if (isViewed(adapter, run.id)) {
          void refreshThumbnail(sketch, [...partials.values()]);
        }
      },
    });

    // the finished object supersedes everything accumulated along the way,
    // and is itself let go once drawn — the server has written it to disk
    clearPartials();
    setCell(key, { state: "done", progress: 1, message: "completed" });
    if (isViewed(adapter, run.id)) await refreshThumbnail(sketch, [result]);
  } catch (exc) {
    if (exc instanceof DOMException && exc.name === "AbortError") throw exc;
    setCell(key, {
      state: "error",
      message: "failed",
      error: exc instanceof Error ? exc.message : String(exc),
    });
  }
}

async function refreshThumbnail(
  sketch: BenchSketch,
  surfaces: ArrayBuffer[],
): Promise<void> {
  sketch.thumbnail = await renderThumbnail(sketch.sketch, surfaces);
}

/** Finished surfaces out of surfaces expected. `done` counts only completed
 *  ones — a failure is neither done nor pending, so a sweep with errors ends
 *  below its total, which is the honest reading. */
export interface Tally {
  done: number;
  total: number;
}

function tally(runs: [string, BenchRun[]][]): Tally {
  let done = 0;
  let total = 0;
  for (const [adapter, list] of runs) {
    for (const run of list) {
      for (const sketch of state.sketches) {
        total++;
        if (cellStatus(adapter, run.id, sketch.name).state === "done") done++;
      }
    }
  }
  return { done, total };
}

export function runTally(adapter: string, runId: string): Tally {
  const run = state.runs[adapter]?.find((r) => r.id === runId);
  return tally(run ? [[adapter, [run]]] : []);
}

export function adapterTally(adapter: string): Tally {
  return tally([[adapter, state.runs[adapter] ?? []]]);
}

/** Across every run of every adapter — the whole sweep. */
export function overallTally(): Tally {
  return tally(Object.entries(state.runs));
}

let persistWarned = false;

/** Write the bench state to progress.json — what makes a benchmark
 *  reopenable, so it runs after every finished cell rather than at the end.
 *
 *  A failure must never stop a run, but it is not harmless either: it means
 *  the sweep in progress cannot be resumed later, so it is reported once
 *  instead of being swallowed entirely (silence here hid a server bug that
 *  rejected every save). */
async function persist(): Promise<void> {
  if (!state.id) return;
  try {
    await api.saveProgress(state.id, {
      runs: state.runs,
      status: state.status,
      viewing: state.viewing,
      sourceDir: state.sourceDir,
    });
    persistWarned = false;
  } catch (exc) {
    if (!persistWarned) {
      persistWarned = true;
      console.warn(
        "benchmark progress could not be saved — this run will not be " +
          "resumable after a reload:",
        exc,
      );
    }
  }
}
