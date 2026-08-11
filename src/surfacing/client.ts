// Client for the local surfacing job server (surfacing-server/). Pure TS:
// talks fetch + core types only, so it sits beside core in the layering and
// anything above may import it.

import type { SketchDocument } from "../core/SketchDocument";
import type { Joint } from "../core/types";
import { rotateVec } from "../core/rigid";

/** What a surfacing method receives: world-space stroke centerlines tagged
 *  with part ids, plus the part and joint tables. Baseline methods use the
 *  points; articulation-aware ones get the screws for free. */
export interface SurfacingSketch {
  strokes: {
    id: string;
    partId: string | null;
    /** Centerline as world-space [x, y, z] samples. */
    points: [number, number, number][];
  }[];
  parts: { id: string; name: string }[];
  joints: Joint[];
}

export interface JobStatus {
  id: string;
  method: string;
  status: "pending" | "running" | "done" | "error";
  progress: number;
  message: string;
  error: string | null;
  /** How many finished pieces the adapter has published so far; a cursor for
   *  /partials, absent on older servers. */
  partialCount?: number;
  /** Where the server wrote the result, when the job carried a save target. */
  savedPath?: string | null;
}

/** Tells the server to write a finished result to
 *  benchmarks/<benchmarkId>/<adapter>/<run>/<sketch>.glb itself, so results
 *  outlive the tab and never round-trip through the client. */
export interface SaveTarget {
  benchmarkId: string;
  adapter: string;
  run: string;
  sketch: string;
}

export interface SurfaceJobOptions {
  method: string;
  sketch: SurfacingSketch;
  options?: Record<string, unknown>;
  save?: SaveTarget;
  onProgress?: (status: JobStatus) => void;
  onLog?: (lines: string[]) => void;
  /** A part that has finished while the rest of the job continues. Awaited,
   *  so a handler that parses the glb sees the pieces strictly in order. */
  onPartial?: (name: string, glb: ArrayBuffer) => void | Promise<void>;
  /** Abort polling. The server-side job keeps running — this only detaches
   *  the client, which is what closing a window should do. */
  signal?: AbortSignal;
}

export function buildSurfacingSketch(doc: SketchDocument): SurfacingSketch {
  const strokes: SurfacingSketch["strokes"] = [];
  for (const stroke of doc.allStrokes()) {
    const { position, quaternion, scale } = stroke.transform;
    const points: [number, number, number][] = [];
    for (let i = 0; i + 2 < stroke.points.length; i += 3) {
      const rotated = rotateVec(quaternion, {
        x: stroke.points[i] * scale.x,
        y: stroke.points[i + 1] * scale.y,
        z: stroke.points[i + 2] * scale.z,
      });
      points.push([
        rotated.x + position.x,
        rotated.y + position.y,
        rotated.z + position.z,
      ]);
    }
    strokes.push({ id: stroke.id, partId: stroke.partId ?? null, points });
  }
  return {
    strokes,
    parts: doc.allParts().map((p) => ({ id: p.id, name: p.name })),
    joints: doc.allJoints(),
  };
}

/** A user-editable parameter one adapter declares; the Surfacer panel
 *  renders these generically and sends the values back in `options`. */
export interface MethodParam {
  name: string;
  label: string;
  type: "int" | "float" | "bool" | "choice";
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
  choices?: string[];
  help?: string;
  /** When set, the input is only enabled while another param equals a given
   *  value (e.g. a per-part control that unlocks when part-based is on). It
   *  still travels in `options`; the adapter decides whether to use it. */
  enabledWhen?: { param: string; equals: number | boolean | string };
}

/** How one method wants the sketch rendered for conditioning. Every field is
 *  optional: the server sends what it cares about and the renderer
 *  (engine/strokeViews.ts) falls back for the rest. A wire type, so it lives
 *  here with the rest of the protocol rather than with the renderer. */
export interface ViewSpec {
  /** Square output edge, in pixels. */
  size?: number;
  /** How many angles, evenly spaced in yaw. */
  count?: number;
  /** Elevation of the orbit, in radians above the horizon. For `helix` this
   *  is the elevation of the first view. */
  pitch?: number;
  /** Where the camera ends up. `ring` holds `pitch` for every view, so they
   *  differ in yaw alone; `helix` also climbs from `pitch` to `pitchMax`
   *  across the sequence, so the last view looks down on the sketch. */
  layout?: "ring" | "helix";
  /** Elevation of the last view under `helix`, in radians. Ignored by
   *  `ring`. Capped short of vertical by the renderer. */
  pitchMax?: number;
  /** CSS colour string, e.g. "#dcdcdc". */
  strokeColor?: string;
  /** Tube radius as a fraction of the sketch's bounding radius; 0 draws
   *  hairlines. */
  strokeThickness?: number;
  /** Camera pullback as a multiple of the bounding radius. */
  margin?: number;
  /** Which of this method's params the user may steer the render with, as
   *  `{ param name: spec field }`. The declaring method owns the mapping, so
   *  a knob like "how many views" reaches the renderer without the renderer
   *  learning any method's param names. Resolved and stripped by
   *  `viewSpecFor`; never reaches the renderer. */
  overrides?: Record<string, ViewSpecField>;
}

/** The `ViewSpec` fields a method may expose as a parameter. Deliberately not
 *  every field: the render style (colour, thickness, size) is a fact about
 *  what the model needs and stays the method's to decide. */
export type ViewSpecField = "count" | "pitch" | "pitchMax" | "layout";

/** A method that conditions on images of the sketch declares how it wants
 *  them rendered; the client obliges and sends the PNGs in `options.views`.
 *  Absent for every method that consumes the strokes as geometry.
 *
 *  `selector` names the param that chooses between specs (a method can offer
 *  several conditioning strategies wanting different renders); without one,
 *  `specs["*"]` applies. A strategy that needs no images has no entry, and
 *  nothing is rendered for it. */
export interface MethodViewSpec {
  selector?: string | null;
  specs: Record<string, ViewSpec>;
}

export interface MethodInfo {
  name: string;
  params: MethodParam[];
  viewSpec?: MethodViewSpec | null;
}

/** The spec a method's current options select, or null if it wants no
 *  renders. Kept here rather than in the caller so every submit path — the
 *  Surfacer panel, the benchmark runner — resolves it the same way.
 *
 *  Any `overrides` the spec declares are folded in here, so what comes back is
 *  a plain render spec and the renderer never sees a param name. An override
 *  whose param is absent from `options` leaves the spec's own value alone —
 *  the method's default stays the default. */
export function viewSpecFor(
  method: MethodInfo,
  options: MethodOptions,
): ViewSpec | null {
  const declared = method.viewSpec;
  if (!declared) return null;
  const key = declared.selector
    ? String(options[declared.selector])
    : DEFAULT_VIEW_SPEC_KEY;
  const spec = declared.specs[key];
  if (!spec) return null;

  const { overrides, ...resolved } = spec;
  for (const [param, field] of Object.entries(overrides ?? {})) {
    const value = options[param];
    if (value === undefined) continue;
    if (field === "layout") {
      if (value === "ring" || value === "helix") resolved.layout = value;
    } else if (typeof value === "number" && Number.isFinite(value)) {
      resolved[field] = value;
    }
  }
  return resolved;
}

const DEFAULT_VIEW_SPEC_KEY = "*";

export type MethodOptions = Record<string, number | boolean | string>;

/** The methods the surfacing server currently offers, with their parameter
 *  declarations. Throws with a start-the-server hint when unreachable. */
export async function fetchMethods(): Promise<MethodInfo[]> {
  const response = await request("/api/health");
  return ((await response.json()) as { methods: MethodInfo[] }).methods;
}

/** The full job lifecycle: submit, poll, stream logs and finished parts, and
 *  resolve with the result glb. */
export async function runSurfacingJob(
  opts: SurfaceJobOptions,
): Promise<ArrayBuffer> {
  const { method, sketch, options = {}, save, signal } = opts;
  const created = await request("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, sketch, options, save }),
  });
  const { jobId } = (await created.json()) as { jobId: string };

  // Aborting used to detach the client and leave the method running — hours
  // of GPU work with nobody waiting for it. Now the abort takes the job with
  // it: the server kills the method's processes and every resident worker,
  // so Stop actually frees the card. Fire-and-forget, and idempotent server
  // side, because nothing here may wait on (or fail because of) the cancel.
  let cancelSent = false;
  const cancel = () => {
    if (cancelSent) return;
    cancelSent = true;
    void fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }).catch(() => {});
  };
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel(); // aborted between submit and listener

  let logCursor = 0;
  const pullLog = async () => {
    if (!opts.onLog) return;
    const { lines, next } = (await (
      await request(`/api/jobs/${jobId}/log?after=${logCursor}`)
    ).json()) as { lines: string[]; next: number };
    logCursor = next;
    if (lines.length > 0) opts.onLog(lines);
  };

  // partials are fetched one blob at a time, so a caller that ignores them
  // (the Surfacer panel) never pays for the bytes
  let partialCursor = 0;
  const pullPartials = async (status: JobStatus) => {
    if (!opts.onPartial) return;
    if ((status.partialCount ?? 0) <= partialCursor) return;
    const { names } = (await (
      await request(`/api/jobs/${jobId}/partials?after=${partialCursor}`)
    ).json()) as { names: string[]; next: number };
    for (const name of names) {
      const index = partialCursor++;
      const glb = await (
        await request(`/api/jobs/${jobId}/partials/${index}`)
      ).arrayBuffer();
      await opts.onPartial(name, glb);
    }
  };

  try {
    for (;;) {
      await delay(500);
      if (signal?.aborted) throw new DOMException("aborted", "AbortError");
      const status = (await (
        await request(`/api/jobs/${jobId}`)
      ).json()) as JobStatus;
      opts.onProgress?.(status);
      await pullLog();
      await pullPartials(status);
      if (status.status === "done") {
        return (await request(`/api/jobs/${jobId}/result`)).arrayBuffer();
      }
      if (status.status === "error") {
        throw new Error(status.error ?? "surfacing failed");
      }
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
    // catch log lines emitted between the last poll and the terminal state
    await pullLog().catch(() => {});
  }
}

async function request(url: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error(OFFLINE_HINT);
  }
  if (response.status === 500 || response.status === 502 || response.status === 504) {
    // what the Vite proxy returns when nothing listens on the server port
    throw new Error(OFFLINE_HINT);
  }
  if (!response.ok) {
    throw new Error(await response.text().catch(() => response.statusText));
  }
  return response;
}

const OFFLINE_HINT =
  "surfacing server unreachable — start it with " +
  "`uvicorn server:app --port 8801` in surfacing-server/ (see its README)";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
