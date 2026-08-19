/**
 * A separable Gaussian blur over a cubic u8 scalar volume.
 *
 * On the grid, not on the picture. Blurring the rendered image would only
 * soften edges on screen; blurring the field changes what the threshold reads,
 * which is the question worth asking of an occupancy prediction — how much of
 * the detail survives smoothing, and where a fatter, softer field would close
 * a gap the sharp one leaves open. It is also the shape an inpainting signal
 * built from this field would take, so the slider is a preview of that.
 *
 * Pure TypeScript over typed arrays: no three.js, no DOM. Separable because a
 * 3D Gaussian is three 1D passes — `grid**3 * 3 * taps` work instead of
 * `grid**3 * taps**3`, which at 112^3 is the difference between a frame and a
 * minute. Edges clamp: a voxel at the boundary borrows its own value rather
 * than fading into a zero that was never measured.
 */

/** Sigmas above this add nothing but cost — at 112^3 the field is already a
 *  blob well before here, and the kernel grows linearly. */
export const MAX_BLUR_SIGMA = 4;

/** Half a normalized Gaussian kernel: `[center, 1 away, 2 away, ...]`, with
 *  `center + 2 * sum(rest) === 1`. Truncated at 3σ, where the tail is under
 *  0.3% of the mass. */
export function gaussianKernel(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const kernel = new Float32Array(radius + 1);
  const denominator = 2 * sigma * sigma;
  let total = 0;
  for (let k = 0; k <= radius; k++) {
    kernel[k] = Math.exp(-(k * k) / denominator);
    total += k === 0 ? kernel[k] : 2 * kernel[k];
  }
  for (let k = 0; k <= radius; k++) kernel[k] /= total;
  return kernel;
}

/**
 * Blur `voxels` (a `grid ** 3` field, x fastest) by `sigma` voxels.
 *
 * `sigma <= 0` is the identity and returns the input itself — callers treat
 * these arrays as immutable, and copying 1.4 MB to change nothing is waste.
 * Anything else returns a fresh array, requantized to u8 the same way the
 * adapter quantized the probabilities in the first place.
 */
export function blurVolume(
  voxels: Uint8Array,
  grid: number,
  sigma: number,
): Uint8Array {
  if (!(sigma > 0)) return voxels;
  const total = grid ** 3;
  if (voxels.length !== total) {
    throw new Error(
      `volume is ${voxels.length} bytes, expected ${total} for ${grid}^3`,
    );
  }

  const kernel = gaussianKernel(Math.min(sigma, MAX_BLUR_SIGMA));
  let source = Float32Array.from(voxels);
  let target = new Float32Array(total);
  // x, y, z — the strides the flat index has along each axis
  for (const stride of [1, grid, grid * grid]) {
    blurAxis(source, target, grid, stride, kernel);
    [source, target] = [target, source];
  }

  const out = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    // the same rounding the adapter uses, so an unblurred voxel that survives
    // three identity passes comes back as the byte it went in as
    out[i] = Math.min(255, Math.max(0, Math.round(source[i])));
  }
  return out;
}

/** One 1D pass along the axis `stride` steps through. */
function blurAxis(
  source: Float32Array,
  target: Float32Array,
  grid: number,
  stride: number,
  kernel: Float32Array,
): void {
  const radius = kernel.length - 1;
  const total = source.length;
  for (let index = 0; index < total; index++) {
    // where this voxel sits along the axis being blurred
    const coordinate = ((index / stride) | 0) % grid;
    let sum = source[index] * kernel[0];
    for (let k = 1; k <= radius; k++) {
      // clamp: past the end, keep sampling the edge voxel
      const back = coordinate - k >= 0 ? k : coordinate;
      const forward = coordinate + k < grid ? k : grid - 1 - coordinate;
      sum +=
        (source[index - back * stride] + source[index + forward * stride]) *
        kernel[k];
    }
    target[index] = sum;
  }
}
