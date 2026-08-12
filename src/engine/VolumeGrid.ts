import * as THREE from "three";

/**
 * A raymarched u8 scalar volume, drawn as one box.
 *
 * Why not instanced cubes: the grid is 64^3, so a cube per occupied voxel is
 * tens of thousands of translucent meshes with no correct draw order, and
 * accumulating opacity through them is exactly the case alpha blending gets
 * wrong. Marching a 3D texture inside a single box is order-independent by
 * construction, uploads a whole timeline frame as one 256KB texture, and
 * makes the threshold a shader uniform instead of a rebuild.
 *
 * Why not three's own `VolumeRenderShader1`: it offers maximum-intensity and
 * first-hit-isosurface casting against a colormap texture. Neither is an
 * occupancy readout — what is wanted here is emission-absorption accumulation
 * with a hard split at the threshold, which is the loop below.
 *
 * Depth is deliberately approximate. The box is drawn transparent after the
 * opaque scene with `depthWrite` off, so the strokes and the mesh show
 * through wherever the accumulated alpha stops short of the cap — which is
 * what the cap is for. A stroke *inside* the volume is dimmed by the alpha of
 * the whole volume rather than by the part in front of it; correcting that
 * needs a depth prepass, and at these opacities it is not visible.
 */
export interface VolumeStyle {
  /** Split the field at `threshold` into two colours, or read it as one
   *  continuous ramp. Occupancy wants the split — the threshold is a real
   *  decision the pipeline makes. A field that measures something else (how
   *  far the latent still has to move) has no threshold to speak of, and
   *  drawing one would invent a boundary that does not exist. */
  splitAtThreshold: boolean;
  /** Where that split falls (0..1). The structure stage's own cut is exactly
   *  0.5 — the stored byte is `sigmoid(logit)` and the pipeline keeps
   *  `logit > 0`. Ignored entirely when `splitAtThreshold` is off. */
  threshold: number;
  /** The field itself, drawn in proportion to its value: faint where the
   *  sample is near zero, strong where it is confident. */
  hazeColor: THREE.Color;
  /** Only above the threshold, and only when splitting: the occupancy the
   *  pipeline would actually keep. */
  solidColor: THREE.Color;
  /** Highest opacity any ray may accumulate. Below 1 so the sketch stays
   *  visible through the densest part of the object. */
  maxOpacity: number;
  /** Multiplies the per-sample absorption — how much of the light a unit of
   *  distance through a full-value voxel swallows. Higher makes faint
   *  structure visible and saturates the solid regions; lower lets the eye
   *  reach further into the object. It changes only the picture, never the
   *  stored field or where the threshold falls. */
  density: number;
}

export const DEFAULT_VOLUME_STYLE: VolumeStyle = {
  splitAtThreshold: true,
  threshold: 0.5,
  hazeColor: new THREE.Color(0x2f7fe0),
  solidColor: new THREE.Color(0xe03b2f),
  maxOpacity: 0.8,
  density: 1.0,
};

const VERTEX_SHADER = /* glsl */ `
varying vec3 vLocal;
void main() {
  vLocal = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// The box is a unit cube centred on the origin, matching the frame the
// worker writes its volumes in: voxel (a, b, c) sits at (a + 0.5) / grid - 0.5.
const FRAGMENT_SHADER = /* glsl */ `
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform vec3 uCameraLocal;
uniform vec3 uHazeColor;
uniform vec3 uSolidColor;
uniform float uThreshold;
uniform float uMaxOpacity;
uniform float uDensity;
uniform float uSteps;
uniform float uSplit;

varying vec3 vLocal;

// GLSL3 through a ShaderMaterial gets no gl_FragColor from three — unlike the
// GLSL1 path, which defines it onto an injected output.
layout(location = 0) out vec4 fragColor;

// Slab test against the unit cube. Returns false when the ray misses.
bool intersectBox(vec3 origin, vec3 dir, out float near, out float far) {
  vec3 inverse = 1.0 / dir;
  vec3 a = (vec3(-0.5) - origin) * inverse;
  vec3 b = (vec3(0.5) - origin) * inverse;
  vec3 low = min(a, b);
  vec3 high = max(a, b);
  near = max(max(low.x, low.y), low.z);
  far = min(min(high.x, high.y), high.z);
  return far > max(near, 0.0);
}

void main() {
  vec3 dir = normalize(vLocal - uCameraLocal);
  float near, far;
  if (!intersectBox(uCameraLocal, dir, near, far)) discard;
  near = max(near, 0.0); // camera inside the volume starts at the camera

  float span = far - near;
  float stride = span / uSteps;
  // Absorption is per unit length, so a longer ray through the same voxel is
  // more opaque — the alpha readout is thickness, which is the point.
  float unit = stride * uDensity;

  vec3 accumulated = vec3(0.0);
  float alpha = 0.0;

  for (int i = 0; i < 512; i++) {
    if (float(i) >= uSteps) break;
    vec3 point = uCameraLocal + dir * (near + (float(i) + 0.5) * stride);
    float value = texture(uVolume, point + 0.5).r;

    // The field is drawn as it is stored — absorption proportional to the
    // value, so what you see is the decoder's own confidence rather than a
    // binarization of it. The split only recolours: at or above the
    // threshold a voxel is one the pipeline would keep, and it goes solid.
    // uSplit is 0 for fields that have no threshold to draw.
    float solid = step(uThreshold, value) * uSplit;
    vec3 tint = mix(uHazeColor, uSolidColor, solid);
    float strength = mix(value, 1.0, solid);

    float sampleAlpha = 1.0 - exp(-strength * unit * 24.0);
    // front-to-back compositing: what is already opaque cannot be relit
    accumulated += (1.0 - alpha) * sampleAlpha * tint;
    alpha += (1.0 - alpha) * sampleAlpha;
    if (alpha >= uMaxOpacity) break;
  }

  alpha = min(alpha, uMaxOpacity);
  if (alpha <= 0.001) discard;
  fragColor = vec4(accumulated, alpha);
}
`;

/** One box, one 3D texture, one material. `setVolume` swaps the frame. */
export class VolumeGrid {
  readonly object: THREE.Mesh;

  private readonly material: THREE.ShaderMaterial;
  private texture: THREE.Data3DTexture | null = null;
  private grid = 0;

  constructor(style: VolumeStyle = DEFAULT_VOLUME_STYLE) {
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      glslVersion: THREE.GLSL3,
      transparent: true,
      depthWrite: false,
      // the loop composites front-to-back, so colour comes out already
      // scaled by alpha; without this three would scale it a second time
      premultipliedAlpha: true,
      // the box has to keep drawing when the camera is inside it, and the
      // marching starts from the camera in that case anyway
      side: THREE.BackSide,
      uniforms: {
        uVolume: { value: null },
        uCameraLocal: { value: new THREE.Vector3() },
        uHazeColor: { value: style.hazeColor.clone() },
        uSolidColor: { value: style.solidColor.clone() },
        uThreshold: { value: style.threshold },
        uMaxOpacity: { value: style.maxOpacity },
        uDensity: { value: style.density },
        uSteps: { value: 192 },
        uSplit: { value: style.splitAtThreshold ? 1 : 0 },
      },
    });

    this.object = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.material);
    this.object.frustumCulled = false;
    // after the opaque scene, so the accumulated alpha composites over the
    // strokes rather than replacing them
    this.object.renderOrder = 10;
    this.object.onBeforeRender = (_r, _s, camera) => {
      this.object.worldToLocal(
        this.material.uniforms.uCameraLocal.value.setFromMatrixPosition(
          camera.matrixWorld,
        ),
      );
    };
  }

  /** Upload one timeline frame. `volume` is `grid ** 3` bytes in x-fastest
   *  order — exactly what `decodeFlowFrames` hands back. */
  setVolume(volume: Uint8Array, grid: number): void {
    if (!this.texture || this.grid !== grid) {
      this.texture?.dispose();
      this.texture = new THREE.Data3DTexture(
        new Uint8Array(volume),
        grid,
        grid,
        grid,
      );
      this.texture.format = THREE.RedFormat;
      this.texture.type = THREE.UnsignedByteType;
      this.texture.minFilter = THREE.LinearFilter;
      this.texture.magFilter = THREE.LinearFilter;
      this.texture.wrapS = THREE.ClampToEdgeWrapping;
      this.texture.wrapT = THREE.ClampToEdgeWrapping;
      this.texture.wrapR = THREE.ClampToEdgeWrapping;
      this.texture.unpackAlignment = 1;
      this.grid = grid;
      this.material.uniforms.uVolume.value = this.texture;
      // a step per voxel along the longest diagonal is enough to never skip
      // a cell, and the 512 cap in the shader keeps it bounded
      this.material.uniforms.uSteps.value = Math.min(512, grid * 3);
    } else {
      (this.texture.image.data as Uint8Array).set(volume);
    }
    this.texture.needsUpdate = true;
  }

  setStyle(style: Partial<VolumeStyle>): void {
    const uniforms = this.material.uniforms;
    if (style.splitAtThreshold !== undefined) {
      uniforms.uSplit.value = style.splitAtThreshold ? 1 : 0;
    }
    if (style.threshold !== undefined) uniforms.uThreshold.value = style.threshold;
    if (style.maxOpacity !== undefined) uniforms.uMaxOpacity.value = style.maxOpacity;
    if (style.density !== undefined) uniforms.uDensity.value = style.density;
    if (style.hazeColor) uniforms.uHazeColor.value.copy(style.hazeColor);
    if (style.solidColor) uniforms.uSolidColor.value.copy(style.solidColor);
  }

  setVisible(visible: boolean): void {
    this.object.visible = visible;
  }

  dispose(): void {
    this.object.geometry.dispose();
    this.material.dispose();
    this.texture?.dispose();
    this.texture = null;
  }
}
