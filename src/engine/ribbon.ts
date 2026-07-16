import * as THREE from "three";

// Screen-space ribbon rendering for strokes, the same technique as
// THREE.MeshLine (https://github.com/spite/THREE.MeshLine), which Penzil
// used: each centerline point is emitted twice (side = ±1) at zero width,
// and the vertex shader pushes the two copies apart perpendicular to the
// line's screen-space direction. The ribbon therefore always faces the
// camera, which is visually indistinguishable from a round tube.

/**
 * Per-vertex width profile derived from pen pressure — Penzil's exact
 * formula: base 3 + pressure*16, with the first/last `tailLength` points
 * tapering to 0 so strokes get pointed tips. The shader multiplies this by
 * the material's `lineWidth`, so values here are unitless multipliers.
 */
export function computeVertexWidths(pressure: Float32Array): Float32Array {
  const n = pressure.length;
  const widths = new Float32Array(n);
  const base = 3;
  const pressureScale = 16;
  const tailLength = 3;

  for (let i = 0; i < n; i++) {
    const full = base + pressure[i] * pressureScale;
    if (i < tailLength) {
      widths[i] = (i / tailLength) * full;
    } else if (i > n - 1 - tailLength) {
      widths[i] = ((n - 1 - i) / tailLength) * full;
    } else {
      widths[i] = full;
    }
  }
  return widths;
}

/**
 * Expand a centerline (xyz triplets) into ribbon buffers: 2 vertices per
 * point, 2 triangles per segment. All positions stay on the centerline —
 * width only exists in the vertex shader.
 */
export function buildRibbonGeometry(
  points: Float32Array,
  widths: Float32Array,
): THREE.BufferGeometry {
  const n = points.length / 3;
  const position = new Float32Array(n * 2 * 3);
  const previous = new Float32Array(n * 2 * 3);
  const next = new Float32Array(n * 2 * 3);
  const side = new Float32Array(n * 2);
  const width = new Float32Array(n * 2);
  const index = new Uint32Array(Math.max(n - 1, 0) * 6);

  const copyPoint = (target: Float32Array, vertex: number, point: number) => {
    target[vertex * 3] = points[point * 3];
    target[vertex * 3 + 1] = points[point * 3 + 1];
    target[vertex * 3 + 2] = points[point * 3 + 2];
  };

  for (let i = 0; i < n; i++) {
    const prevI = Math.max(i - 1, 0);
    const nextI = Math.min(i + 1, n - 1);
    for (const [offset, sign] of [
      [0, 1],
      [1, -1],
    ] as const) {
      const v = i * 2 + offset;
      copyPoint(position, v, i);
      copyPoint(previous, v, prevI);
      copyPoint(next, v, nextI);
      side[v] = sign;
      width[v] = widths[i];
    }
  }

  for (let i = 0; i < n - 1; i++) {
    index.set(
      [i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 2, i * 2 + 1, i * 2 + 3],
      i * 6,
    );
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("previous", new THREE.BufferAttribute(previous, 3));
  geometry.setAttribute("next", new THREE.BufferAttribute(next, 3));
  geometry.setAttribute("side", new THREE.BufferAttribute(side, 1));
  geometry.setAttribute("width", new THREE.BufferAttribute(width, 1));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  geometry.computeBoundingSphere();
  return geometry;
}

const vertexShader = /* glsl */ `
  uniform float lineWidth;
  uniform vec2 resolution;

  attribute vec3 previous;
  attribute vec3 next;
  attribute float side;
  attribute float width;

  #include <fog_pars_vertex>

  // Perspective-divide to NDC, aspect-corrected so screen angles are true.
  vec2 toScreen(vec4 clip, float aspect) {
    vec2 ndc = clip.xy / clip.w;
    ndc.x *= aspect;
    return ndc;
  }

  void main() {
    float aspect = resolution.x / resolution.y;
    mat4 m = projectionMatrix * modelViewMatrix;

    vec4 finalPosition = m * vec4(position, 1.0);
    vec4 prevClip = m * vec4(previous, 1.0);
    vec4 nextClip = m * vec4(next, 1.0);

    vec2 currentP = toScreen(finalPosition, aspect);
    vec2 prevP = toScreen(prevClip, aspect);
    vec2 nextP = toScreen(nextClip, aspect);

    vec2 dir;
    if (nextP == currentP) {
      dir = normalize(currentP - prevP);
    } else if (prevP == currentP) {
      dir = normalize(nextP - currentP);
    } else {
      // interior points: average the two segment directions (miter)
      dir = normalize(normalize(currentP - prevP) + normalize(nextP - currentP));
    }

    vec2 normal = vec2(-dir.y, dir.x);
    normal.x /= aspect;
    normal *= 0.5 * lineWidth * width;

    // Offsetting clip.xy without scaling by clip.w makes the on-screen
    // width shrink with distance (world-space width, like Penzil's
    // sizeAttenuation: 1).
    finalPosition.xy += normal * side;
    gl_Position = finalPosition;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    #include <fog_vertex>
  }
`;

const fragmentShader = /* glsl */ `
  uniform vec3 color;
  uniform float opacity;
  uniform float highlight; // 0..1, blends toward highlightColor
  uniform vec3 highlightColor;

  #include <fog_pars_fragment>

  void main() {
    vec3 c = mix(color, highlightColor, highlight * 0.65);
    gl_FragColor = vec4(c, opacity);
    #include <fog_fragment>
  }
`;

export function createRibbonMaterial(
  color: string,
  lineWidth: number,
  resolution: THREE.Vector2,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    fog: true,
    side: THREE.DoubleSide,
    transparent: true,
    uniforms: {
      color: { value: new THREE.Color(color) },
      opacity: { value: 0.9 },
      highlight: { value: 0 },
      highlightColor: { value: new THREE.Color(1, 0.62, 0) },
      lineWidth: { value: lineWidth },
      resolution: { value: resolution }, // shared instance, updated on resize
      fogColor: { value: new THREE.Color() },
      fogNear: { value: 1 },
      fogFar: { value: 1000 },
      fogDensity: { value: 0.00025 },
    },
  });
}
