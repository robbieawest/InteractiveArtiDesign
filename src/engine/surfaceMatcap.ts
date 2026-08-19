// The shading every surface in the app is drawn with.
//
// A matcap rather than a lit material: form comes from a baked sphere image,
// so it reads the same however the camera moves and needs no light rig. That
// matters twice over here — the viewport overlay orbits freely, and the
// conditioning renders point a camera at the object from several directions
// with no chance to relight it. A headlight, which is the obvious thing for
// the second case, is the one setup that flattens a surface completely: light
// arriving along the view direction leaves N·L near 1 everywhere facing the
// camera, so the object comes out as a silhouette with no interior form.
//
// Shared as one texture: it is 256x256 drawn once, and two renderers uploading
// their own GPU copy of the same canvas is cheaper than two canvases.

import * as THREE from "three";

let shared: THREE.Texture | null = null;

export function getSurfaceMatcap(): THREE.Texture {
  if (!shared) shared = makeMatcapTexture();
  return shared;
}

/** A soft studio matcap drawn procedurally (no asset dependency): a sphere lit
 *  from the upper-left, bright falling to dark, with a soft specular hotspot.
 *  Grayscale so the material colour tints it by multiplication. */
function makeMatcapTexture(): THREE.Texture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#2b2b2b";
  ctx.fillRect(0, 0, size, size);

  const diffuse = ctx.createRadialGradient(
    size * 0.36,
    size * 0.3,
    size * 0.04,
    size * 0.5,
    size * 0.5,
    size * 0.62,
  );
  diffuse.addColorStop(0, "#ffffff");
  diffuse.addColorStop(0.5, "#b8b8b8");
  diffuse.addColorStop(1, "#454545");
  ctx.fillStyle = diffuse;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
  ctx.fill();

  const spec = ctx.createRadialGradient(
    size * 0.33,
    size * 0.27,
    0,
    size * 0.33,
    size * 0.27,
    size * 0.2,
  );
  spec.addColorStop(0, "rgba(255,255,255,0.85)");
  spec.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = spec;
  ctx.beginPath();
  ctx.arc(size * 0.33, size * 0.27, size * 0.2, 0, Math.PI * 2);
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/** Mix a fresnel term into a matcap material's outgoing light: the silhouette
 *  and every surface turning away from the camera go toward `color`.
 *
 *  Both callers want the contour picked out and disagree only about which way.
 *  The overlay lightens it, because it is translucent over a busy scene and
 *  the silhouette is what separates it from the sketch. A conditioning render
 *  darkens it, because it is an opaque object on a flat ground and the
 *  contour is the only depth cue an image model gets: without it a light grey
 *  solid reads as a flat blob of one colour, whatever its geometry.
 *
 *  `strength` 0 leaves the matcap alone. */
export function injectFresnelRim(
  material: THREE.MeshMatcapMaterial,
  color: THREE.Color,
  strength: number,
  power: number,
  mode: "add" | "toward",
): void {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.rimColor = { value: color };
    shader.uniforms.rimPower = { value: power };
    shader.uniforms.rimStrength = { value: strength };
    const blend =
      mode === "add"
        ? "outgoingLight += rimColor * rim;"
        : "outgoingLight = mix(outgoingLight, rimColor, rim);";
    shader.fragmentShader =
      "uniform vec3 rimColor;\nuniform float rimPower;\nuniform float rimStrength;\n" +
      shader.fragmentShader.replace(
        "#include <opaque_fragment>",
        // abs() so back faces (double-sided) rim too; normal & vViewPosition
        // are both in view space here
        `float rimDot = 1.0 - abs(dot(normalize(normal), normalize(vViewPosition)));
         float rim = pow(rimDot, rimPower) * rimStrength;
         ${blend}
         #include <opaque_fragment>`,
      );
  };
}
