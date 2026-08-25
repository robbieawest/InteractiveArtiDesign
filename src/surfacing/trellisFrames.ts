// Wire format for the TRELLIS interactive capture: what the two flow stages
// did, one volume per step. Pure TS over an ArrayBuffer, so it sits beside
// the rest of the surfacing protocol rather than in the engine — the viewer
// consumes these, but nothing here knows about three.js.
//
// The adapter writes it as: magic "TRLZ", u32 version, u32 header length,
// a JSON header, then every frame back to back. Frames are `grid ** 3` bytes
// of u8, already in WebGL 3D-texture order (x fastest) and already rotated
// into the document's y-up frame — see `to_texture_bytes` in the worker.

const MAGIC = "TRLZ";
const SUPPORTED_VERSION = 1;

/** What one voxel's byte means, per stage. These two are the flow itself,
 *  and they run in sequence — which is what makes one scrubber over both
 *  meaningful. */
export type FlowStage = "structure" | "latent";

/** Everything a bundle can carry, including tracks that are not stages of the
 *  flow. "constraint" is the inpainting signal as it stood at each structure
 *  step: it runs *alongside* those steps rather than after them, so it has a
 *  frame per structure step and adds nothing to the timeline. */
export type FlowTrack = FlowStage | "constraint";

/** The similarity that puts the generated unit cube onto the sketch:
 *  `world = scale * rotation * v + translation`, with `rotation` in
 *  row-major order. Absent when the run could not solve one (fit turned off,
 *  or degenerate strokes), in which case the lattice has no known place. */
export interface FlowAlign {
  rotation: number[];
  scale: number;
  translation: number[];
}

export interface FlowFrames {
  /** Edge of the cubic grid; 64 for every current TRELLIS checkpoint. */
  grid: number;
  align: FlowAlign | null;
  /** Per track, one u8 volume per sampling step, in step order. */
  stages: Record<FlowTrack, Uint8Array[]>;
  /** The flow time each frame was predicted at, per track, in step order.
   *  Empty for a capture that did not record them (TRELLIS 1, and TRELLIS.2
   *  runs from before the field existed) — the scrubber falls back to the
   *  step index alone. Not derivable on this side: `rescale_t` decides the
   *  spacing and it lives in the checkpoint. */
  times: Record<FlowTrack, number[]>;
}

/** Number of steps in each stage, in timeline order. */
export function stageLengths(frames: FlowFrames): {
  structure: number;
  latent: number;
  total: number;
} {
  const structure = frames.stages.structure.length;
  const latent = frames.stages.latent.length;
  return { structure, latent, total: structure + latent };
}

/** Where a position on the unified timeline lands: which stage, and which
 *  step within it. The two stages are concatenated because they run in
 *  sequence — structure first, then latent on the voxels it chose. */
export function locateFrame(
  frames: FlowFrames,
  position: number,
): { stage: FlowStage; step: number; volume: Uint8Array } | null {
  const { structure } = stageLengths(frames);
  const stage: FlowStage = position < structure ? "structure" : "latent";
  const step = position < structure ? position : position - structure;
  const volume = frames.stages[stage][step];
  return volume ? { stage, step, volume } : null;
}

/** Parse a bundle published by the TRELLIS adapter. Throws on anything it
 *  does not recognise rather than rendering a misread volume — a wrong
 *  stride here looks like a plausible shape, not like an error. */
export function decodeFlowFrames(buffer: ArrayBuffer): FlowFrames {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12) throw new Error("flow bundle is truncated");
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== MAGIC) throw new Error(`not a flow bundle (magic ${magic})`);

  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  if (version !== SUPPORTED_VERSION) {
    throw new Error(
      `flow bundle version ${version}, this build reads ${SUPPORTED_VERSION}`,
    );
  }

  const headerLength = view.getUint32(8, true);
  const headerStart = 12;
  const payloadStart = headerStart + headerLength;
  if (payloadStart > bytes.length) throw new Error("flow header is truncated");
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(headerStart, payloadStart)),
  ) as {
    grid?: number;
    frameBytes?: number;
    align?: FlowAlign | null;
    stages?: {
      name: string;
      offset: number;
      steps: number;
      times?: number[];
    }[];
  };

  const grid = header.grid ?? 64;
  const frameBytes = header.frameBytes ?? grid ** 3;
  const stages: Record<FlowTrack, Uint8Array[]> = {
    structure: [],
    latent: [],
    constraint: [],
  };
  const times: Record<FlowTrack, number[]> = {
    structure: [],
    latent: [],
    constraint: [],
  };
  const known = new Set<string>(Object.keys(stages));

  for (const stage of header.stages ?? []) {
    // an unknown track is a newer capture read by an older build: skip it and
    // show the rest, rather than losing the whole view over one extra name
    if (!known.has(stage.name)) continue;
    for (let step = 0; step < stage.steps; step++) {
      const start = payloadStart + stage.offset + step * frameBytes;
      const end = start + frameBytes;
      if (end > bytes.length) {
        throw new Error(
          `flow bundle is short: ${stage.name} step ${step} needs ` +
            `${end} bytes, got ${bytes.length}`,
        );
      }
      // a view, not a copy: the whole capture is already one buffer and the
      // frames are uploaded to the GPU one at a time
      stages[stage.name as FlowTrack].push(bytes.subarray(start, end));
    }
    // only when it lines up with the frames actually read — a mismatched
    // length would silently label the wrong step
    if (stage.times?.length === stage.steps) {
      times[stage.name as FlowTrack] = stage.times;
    }
  }

  return { grid, align: header.align ?? null, stages, times };
}
