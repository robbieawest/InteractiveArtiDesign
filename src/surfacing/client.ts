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

/** The methods (adapter names) the surfacing server currently offers.
 *  Throws with a start-the-server hint when it is unreachable. */
export async function fetchMethods(): Promise<string[]> {
  const response = await request("/api/health");
  return ((await response.json()) as { methods: string[] }).methods;
}

/** Submit a job and poll it to completion; resolves with the result glb. */
export async function surfaceSketch(
  method: string,
  sketch: SurfacingSketch,
  options: Record<string, unknown> = {},
  onProgress?: (status: JobStatus) => void,
): Promise<ArrayBuffer> {
  const created = await request("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ method, sketch, options }),
  });
  const { jobId } = (await created.json()) as { jobId: string };

  for (;;) {
    await delay(500);
    const status = (await (
      await request(`/api/jobs/${jobId}`)
    ).json()) as JobStatus;
    onProgress?.(status);
    if (status.status === "done") {
      return (await request(`/api/jobs/${jobId}/result`)).arrayBuffer();
    }
    if (status.status === "error") {
      throw new Error(status.error ?? "surfacing failed");
    }
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
