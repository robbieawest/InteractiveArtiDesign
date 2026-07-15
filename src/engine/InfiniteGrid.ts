import * as THREE from "three";

// Infinite ground grid rendered by a single camera-following quad.
// Ported from Penzil's InfiniteGridHelper, originally by Fyrestar
// (https://github.com/Fyrestar/THREE.InfiniteGridHelper), adapted to
// TypeScript and current three.js (no OES_standard_derivatives needed
// under WebGL2).
export function createInfiniteGrid(
  size1 = 1,
  size2 = 10,
  color = new THREE.Color(0x0000ff),
  distance = 40,
): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(2, 2, 1, 1);

  const material = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    uniforms: {
      uSize1: { value: size1 },
      uSize2: { value: size2 },
      uColor: { value: color },
      uDistance: { value: distance },
    },
    vertexShader: /* glsl */ `
      varying vec3 worldPosition;

      uniform float uDistance;

      void main() {
        // The unit quad is stretched to cover the visible range and
        // re-centered on the camera every frame, so it never runs out.
        vec3 pos = position.xzy * uDistance;
        pos.xz += cameraPosition.xz;

        worldPosition = pos;

        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 worldPosition;

      uniform float uSize1;
      uniform float uSize2;
      uniform vec3 uColor;
      uniform float uDistance;

      float getGrid(float size) {
        vec2 r = worldPosition.xz / size;
        vec2 grid = abs(fract(r - 0.5) - 0.5) / fwidth(r);
        float line = min(grid.x, grid.y);
        return 1.0 - min(line, 1.0);
      }

      void main() {
        float d = 1.0 - min(distance(cameraPosition.xz, worldPosition.xz) / uDistance, 1.0);

        float g1 = getGrid(uSize1);
        float g2 = getGrid(uSize2);

        gl_FragColor = vec4(uColor.rgb, mix(g2, g1, g1) * pow(d, 3.0));
        gl_FragColor.a = mix(0.5 * gl_FragColor.a, gl_FragColor.a, g2);

        if (gl_FragColor.a <= 0.0) discard;
      }
    `,
  });

  const grid = new THREE.Mesh(geometry, material);
  grid.frustumCulled = false;
  return grid;
}
