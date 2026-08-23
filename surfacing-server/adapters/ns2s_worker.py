"""Resident NeuralSketch2Surf inference process.

Run in the ns2s venv by adapters/ns2s.py and kept alive between jobs. The
point is the ~15s of setup a fresh interpreter pays before it can do ~1s of
work — almost all of it ROCm context creation on the first `.cuda()`. Loading
the model once and holding it turns a benchmark over N sketches from N*16s
into 15s + N*1s.

Protocol: one JSON request per line on stdin, one JSON event per line on
stdout. Requests are

    {"input": <dir>, "output": <dir>, "threshold": f, "margin": f,
     "img_size": i, "volume": bool, "blur": f}

and the reply is one {"event": "file", "name": ..., "ok": bool} per .obj
found, then {"event": "done", "count": n} — or {"event": "error", ...} if the
whole request failed. Anything the model prints goes to stderr so it cannot
corrupt the stream.

With "volume" set the run stops at the network's output: each input writes its
raw occupancy probabilities as `<stem>_prob.u8` plus a `<stem>_prob.json`
describing the grid, and no marching cubes runs. A mesh extracted at one
threshold has already thrown the probabilities away, and they are the thing
the client raymarches (and the thing an inpainting signal is made of).

With "volume" and "mesh" both set it writes the grid *and* meshes it from the
same forward pass — marching cubes at "threshold", largest component, normals
repaired, exactly what `process_and_save` does after its own sigmoid. That
combination exists because two consumers want the same prediction: TRELLIS
conditions on renders of the mesh and constrains sampling with the field, and
running the network twice for them is a minute of GPU and 13GB of residency for
an answer that is already in hand. "blur" applies to the written grid only —
the mesh is marched on the raw probabilities.
"""

import glob
import json
import os
import sys


def main() -> None:
    repo, model_path, img_size, feature_size = sys.argv[1:5]
    # inference.py imports train112TVloss and network/ relative to the repo,
    # and sys.path[0] is this script's directory, not the repo
    sys.path.insert(0, repo)
    os.chdir(repo)

    # everything the model layer prints must go to stderr; stdout is protocol
    stdout = sys.stdout
    sys.stdout = sys.stderr

    from inference import InferenceEngine  # noqa: E402  (needs sys.path first)
    import torch  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = InferenceEngine(
        model_path,
        device=device,
        img_size=int(img_size),
        feature_size=int(feature_size),
        eager=True,
    )

    def send(payload: dict) -> None:
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()

    send({"event": "ready", "device": device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handle(engine, request, send)
        except Exception as exc:  # never die on one bad request
            send({"event": "error", "message": f"{type(exc).__name__}: {exc}"})


def predict_field(engine, obj_path, img_size, margin):
    """One forward pass: the occupancy probabilities and the normalization.

    Returns `(field, params)`, or `(None, None)` when the sketch has too few
    points to voxelize. `field` is float in [0, 1] on the method's own
    `img_size ** 3` grid, exactly as the network produced it — smoothing is a
    separate step, applied by whoever wants it and not by whoever asked for
    the prediction.
    """
    import torch

    from inference import voxelize_strict_aligned

    input_np, params = voxelize_strict_aligned(obj_path, img_size, margin)
    if input_np is None:
        return None, None

    with torch.no_grad():
        logits = engine.model(torch.from_numpy(input_np).to(engine.device))
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()[0, 0], params


def blur_field(field, blur):
    """A Gaussian over the probabilities, in voxels. Clamped edges and a 3σ
    truncation, matching `engine/volumeBlur.ts`, so the client's own slider is
    the same operator applied twice over rather than a different one."""
    if blur <= 0:
        return field
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(field, sigma=blur, mode="nearest", truncate=3.0)


def write_probability_grid(field, params, out_stem, blur) -> None:
    """The field as the two files the client and the adapters read.

    Two of them, because the payload is a fixed-stride block and everything
    else is small:

    * `<stem>.u8`  — `img_size ** 3` bytes, `round(p * 255)`, ordered with x
      varying fastest so the client can hand it to a WebGL 3D texture with no
      further work (the same convention as the TRELLIS capture).
    * `<stem>.json` — the grid edge, the similarity that puts the unit cube
      the texture addresses back onto the sketch, and the blur that was
      applied.

    The alignment inverts the method's own normalization. `voxelize_strict_aligned`
    maps a vertex to `n = (v - center) * scale` and samples voxel i (of R) at
    `n = -margin + i * voxel_size`; a unit cube centred on the origin
    addresses that same voxel at `x = (i + 0.5) / R - 0.5`. Eliminating i
    gives `n = x * 2 * margin * R / (R - 1)` — no offset, since the grid is
    symmetric about the origin — and so
    `world = x * 2 * margin * R / ((R - 1) * scale) + center`, which is a
    scale and a translation with no rotation in it.
    """
    import numpy as np

    resolution = int(params["resolution"])
    quantized = np.clip(field * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # C order runs z fastest; a 3D texture wants x fastest
    with open(f"{out_stem}.u8", "wb") as handle:
        handle.write(np.ascontiguousarray(quantized.transpose(2, 1, 0)).tobytes())

    scale = float(params["scale"])
    with open(f"{out_stem}.json", "w") as handle:
        json.dump(
            {
                "grid": resolution,
                "align": {
                    "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "scale": 2.0 * float(params["margin"]) * resolution
                    / ((resolution - 1) * scale),
                    "translation": [float(v) for v in params["center"]],
                },
                "blur": float(blur),
                "max": float(field.max()),
                "mean": float(field.mean()),
            },
            handle,
        )


def mesh_field(field, params, threshold, obj_out, npz_out) -> bool:
    """Marching cubes on an already-predicted field, as `process_and_save`
    would have done it.

    Deliberately the same steps in the same order — cubes at `threshold`,
    largest connected component, `fix_normals` — because this is a substitute
    for that call, not a variant of it: a run that meshes from here should be
    indistinguishable from a normal ns2s run at the same threshold and no
    blur. The npz is written last and holds the field, so the adapter's
    "is it finished" check (which watches for it) still means what it meant.
    """
    import numpy as np
    import trimesh
    from skimage import measure

    if field.max() < threshold:
        return False
    verts, faces, _, _ = measure.marching_cubes(field, level=threshold)
    verts = verts * params["voxel_size"] + params["origin"]
    verts = verts / params["scale"] + params["center"]

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    components = mesh.split(only_watertight=False)
    if len(components) > 0:
        mesh = max(components, key=lambda part: len(part.vertices))
    mesh.fix_normals()
    mesh.export(obj_out)
    np.savez(
        npz_out,
        raw_probability_grid=field,
        center=params["center"],
        margin=float(params["margin"]),
        resolution=int(params["resolution"]),
        voxel_size=float(params["voxel_size"]),
        o3d_origin=params["origin"],
        scale=float(params["scale"]),
    )
    return True


def save_probability_grid(
    engine, obj_path, out_stem, img_size, margin, blur=0.0
) -> bool:
    """Run the network on one sketch and write its occupancy probabilities.

    Everything `process_and_save` does after the sigmoid — marching cubes,
    component pruning, normal repair — is a decision made at one threshold,
    so this stops before it and keeps the field itself.
    """
    field, params = predict_field(engine, obj_path, img_size, margin)
    if field is None:
        return False
    write_probability_grid(blur_field(field, blur), params, out_stem, blur)
    return True


def save_grid_and_mesh(
    engine, obj_path, out_stem, prob_stem, npz_path,
    img_size, margin, threshold, blur=0.0
) -> bool:
    """One prediction, published both ways: the field and a mesh of it.

    The blur reaches the grid only. It exists to soften an *inpainting
    constraint* — to close the pinholes a hard threshold on a noisy field
    leaves, at the cost of moving the level set — and none of that is a reason
    to smooth the geometry. Marching cubes runs on the raw probabilities, so
    the surface is the one ns2s would have produced on its own at this
    threshold, and the mesh post-process (`smooth.py`) stays the only thing
    that changes its shape.
    """
    field, params = predict_field(engine, obj_path, img_size, margin)
    if field is None:
        return False
    write_probability_grid(blur_field(field, blur), params, prob_stem, blur)
    # the grid is worth keeping even if the mesh comes out empty: a caller
    # constraining with it does not need the surface
    return mesh_field(field, params, threshold, out_stem, npz_path)


def handle(engine, request: dict, send) -> None:
    in_dir = request["input"]
    out_dir = request["output"]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(in_dir, "**/*.obj"), recursive=True))
    count = 0
    for path in files:
        name = os.path.basename(path)
        if "_recon.obj" in name:
            continue
        stem = name[: -len(".obj")]
        ok = False
        try:
            if request.get("volume") and request.get("mesh"):
                ok = save_grid_and_mesh(
                    engine,
                    path,
                    os.path.join(out_dir, f"{stem}_recon.obj"),
                    os.path.join(out_dir, f"{stem}_prob"),
                    os.path.join(out_dir, f"{stem}_data.npz"),
                    int(request.get("img_size", 112)),
                    float(request.get("margin", 1.2)),
                    float(request.get("threshold", 0.6)),
                    float(request.get("blur", 0.0)),
                )
            elif request.get("volume"):
                ok = save_probability_grid(
                    engine,
                    path,
                    os.path.join(out_dir, f"{stem}_prob"),
                    int(request.get("img_size", 112)),
                    float(request.get("margin", 1.2)),
                    float(request.get("blur", 0.0)),
                )
            else:
                ok = engine.process_and_save(
                    path,
                    os.path.join(out_dir, f"{stem}_recon.obj"),
                    os.path.join(out_dir, f"{stem}_data.npz"),
                    int(request.get("img_size", 112)),
                    float(request.get("threshold", 0.6)),
                    float(request.get("margin", 1.2)),
                )
        except Exception as exc:
            send({"event": "log", "message": f"{stem}: {exc}"})
        # one event per input, so the caller can publish geometry as it lands
        send({"event": "file", "name": stem, "ok": bool(ok)})
        count += 1
    send({"event": "done", "count": count})


if __name__ == "__main__":
    main()
