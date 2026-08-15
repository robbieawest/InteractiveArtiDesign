// Benchmark-storage half of the surfacing server API: scanning a source
// folder, stashing preprocessed sketches, and reading results back. Pure TS,
// same layer as client.ts.
//
// The split of labour is fixed by what each side can do: only the server can
// read a folder of sketches or write next to the repo, and only the client can
// run the glTF importer (it is three.js). So the server hands over raw bytes,
// the client converts, and the converted documents go back to the server.

import type { DocumentJson } from "../core/serialization";
import { request } from "./http";

/** One surfaceable input found in a source folder. `gltf` entries still need
 *  client-side conversion; `json` entries are ready to run, which is what
 *  makes a previous benchmark's sketches/ folder reusable as a source. */
export interface SourceEntry {
  name: string;
  kind: "json" | "gltf";
  path: string;
}

export interface SourceScan {
  dir: string;
  entries: SourceEntry[];
}

export async function scanSource(dir: string): Promise<SourceScan> {
  return (await (
    await request(`/api/benchmark/scan?dir=${encodeURIComponent(dir)}`)
  ).json()) as SourceScan;
}

/** Raw bytes of a file inside a scanned folder — used to pull a .gltf and its
 *  sidecar .bin through to the importer. */
export async function readSourceFile(path: string): Promise<ArrayBuffer> {
  return (
    await request(`/api/benchmark/file?path=${encodeURIComponent(path)}`)
  ).arrayBuffer();
}

export async function saveSketch(
  benchmarkId: string,
  name: string,
  document: DocumentJson,
): Promise<void> {
  await request(`/api/benchmark/${encodeURIComponent(benchmarkId)}/sketches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, document }),
  });
}

export async function readSketch(
  benchmarkId: string,
  name: string,
): Promise<DocumentJson> {
  return (await (
    await request(
      `/api/benchmark/${encodeURIComponent(benchmarkId)}/sketches/${encodeURIComponent(name)}`,
    )
  ).json()) as DocumentJson;
}

/** Copy a benchmark's sketches into a new folder, results left behind. The
 *  target id is ours to pick, same as for a fresh benchmark. */
export async function copySketches(
  benchmarkId: string,
  target: string,
): Promise<void> {
  await request(`/api/benchmark/${encodeURIComponent(benchmarkId)}/copy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target }),
  });
}

export async function listSketches(benchmarkId: string): Promise<string[]> {
  return (
    (await (
      await request(
        `/api/benchmark/${encodeURIComponent(benchmarkId)}/sketches`,
      )
    ).json()) as { sketches: string[] }
  ).sketches;
}

/** Persist the whole bench state. The shape is the client's business; the
 *  server only has to hand it back after a reload. */
export async function saveProgress(
  benchmarkId: string,
  progress: unknown,
): Promise<void> {
  await request(`/api/benchmark/${encodeURIComponent(benchmarkId)}/progress`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(progress),
  });
}

export async function loadProgress(benchmarkId: string): Promise<unknown> {
  return (
    (await (
      await request(
        `/api/benchmark/${encodeURIComponent(benchmarkId)}/progress`,
      )
    ).json()) as { progress: unknown }
  ).progress;
}

/** One reopenable benchmark folder, as the picker describes it. */
export interface BenchmarkSummary {
  id: string;
  sketches: number;
  /** Finished surfaces on disk; 0 means prepared but never started. */
  results: number;
  /** False when there is no progress.json, i.e. no run configuration to
   *  restore — the folder reopens as a fresh bench over its sketches. */
  hasProgress: boolean;
}

export async function listBenchmarks(): Promise<BenchmarkSummary[]> {
  return (
    (await (await request("/api/benchmark")).json()) as {
      benchmarks: BenchmarkSummary[];
    }
  ).benchmarks;
}

export async function readResult(
  benchmarkId: string,
  adapter: string,
  run: string,
  sketch: string,
): Promise<ArrayBuffer> {
  return (
    await request(
      `/api/benchmark/${encodeURIComponent(benchmarkId)}/results/` +
        `${encodeURIComponent(adapter)}/${encodeURIComponent(run)}/${encodeURIComponent(sketch)}`,
    )
  ).arrayBuffer();
}

/** Local-time stamp used as the benchmark folder name: 2026-07-28T15-04-22.
 *  Colons would be legal here but make the folder awkward to type. */
export function newBenchmarkId(now = new Date()): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}` +
    `T${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`
  );
}

