// Wire format for the NeuralSketch2Surf probability volume: the occupancy
// field the network predicts, before marching cubes turns it into one
// threshold's worth of mesh. Pure TS over an ArrayBuffer, beside the rest of
// the surfacing protocol — the viewer consumes these, but nothing here knows
// about three.js.
//
// The adapter writes it as: magic "NSVX", u32 version, u32 header length, a
// JSON header, then `grid ** 3` bytes of u8 — already in WebGL 3D-texture
// order (x fastest), and already quantized as `round(p * 255)`, so a byte
// divided by 255 is the probability the network gave that voxel.

const MAGIC = "NSVX";
const SUPPORTED_VERSION = 1;

/** The similarity that puts the unit cube the texture addresses back onto the
 *  sketch: `world = scale * rotation * v + translation`, `rotation` row-major.
 *  NS2S normalizes the sketch itself and this inverts that, so unlike the
 *  TRELLIS capture it is always known — the rotation is the identity. */
export interface VolumeAlign {
  rotation: number[];
  scale: number;
  translation: number[];
}

export interface OccupancyVolume {
  /** Edge of the cubic grid; 112 for the released checkpoint. */
  grid: number;
  align: VolumeAlign | null;
  /** `grid ** 3` bytes, x fastest. */
  voxels: Uint8Array;
  /** Highest probability in the field, 0..1. A field whose maximum is below
   *  the threshold would have made marching cubes fail — worth saying so
   *  rather than showing an empty box. */
  max: number;
  /** Mean probability, 0..1 — how much of the grid the network filled. */
  mean: number;
}

/** Parse a volume published by the NS2S adapter. Throws on anything it does
 *  not recognise rather than rendering a misread field: a wrong stride here
 *  looks like a plausible shape, not like an error. */
export function decodeOccupancyVolume(buffer: ArrayBuffer): OccupancyVolume {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 12) throw new Error("probability volume is truncated");
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== MAGIC) {
    throw new Error(`not a probability volume (magic ${magic})`);
  }

  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  if (version !== SUPPORTED_VERSION) {
    throw new Error(
      `probability volume version ${version}, this build reads ${SUPPORTED_VERSION}`,
    );
  }

  const headerLength = view.getUint32(8, true);
  const payloadStart = 12 + headerLength;
  if (payloadStart > bytes.length) {
    throw new Error("probability volume header is truncated");
  }
  const header = JSON.parse(
    new TextDecoder().decode(bytes.subarray(12, payloadStart)),
  ) as {
    grid?: number;
    align?: VolumeAlign | null;
    max?: number;
    mean?: number;
  };

  const grid = header.grid ?? 112;
  const expected = grid ** 3;
  const voxels = bytes.subarray(payloadStart, payloadStart + expected);
  if (voxels.length !== expected) {
    throw new Error(
      `probability volume is short: a ${grid}^3 field needs ${expected} ` +
        `bytes, got ${voxels.length}`,
    );
  }

  return {
    grid,
    align: header.align ?? null,
    voxels,
    max: header.max ?? 1,
    mean: header.mean ?? 0,
  };
}
