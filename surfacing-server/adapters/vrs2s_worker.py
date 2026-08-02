"""Resident VRSketch2Shape inference process.

Run in the vrs2s venv by adapters/vrs2s.py and kept alive between jobs, for the
same reason as ns2s_worker.py: setup (interpreter, transformers, the diffusion
UNet, the first `.cuda()`) costs far more than one generation does, and a
benchmark sweep pays it once instead of once per sketch.

Protocol: one JSON request per line on stdin, one JSON event per line on
stdout. Requests are

    {"lines": [[[x,y,z], ...], ...], "output": <path.obj>, "ddim_steps": i,
     "guidance": f, "eta": f, "seed": i, "iso": f, "simplify": f,
     "up_axis": "y"|"z", "fit_to_sketch": bool}

and the reply is a stream of {"event": "progress", "step": i, "total": n}
followed by {"event": "done", "output": ..., "info": {...}} — or
{"event": "error", "message": ...} if the request failed. Anything the model
prints goes to stderr so it cannot corrupt the stream.
"""

import json
import os
import sys


def main() -> None:
    repo, model_path = sys.argv[1:3]
    sys.path.insert(0, repo)
    os.chdir(repo)

    # everything the model layer prints must go to stderr; stdout is protocol
    stdout = sys.stdout
    sys.stdout = sys.stderr

    import mcubes  # noqa: E402
    import torch  # noqa: E402
    from headless import Sketch2ShapeEngine  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = Sketch2ShapeEngine(model_path, device=device)

    def send(payload: dict) -> None:
        stdout.write(json.dumps(payload) + "\n")
        stdout.flush()

    send({"event": "ready", "device": device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle(engine, mcubes, json.loads(line), send)
        except Exception as exc:  # never die on one bad request
            send({"event": "error", "message": f"{type(exc).__name__}: {exc}"})


def handle(engine, mcubes, request: dict, send) -> None:
    steps = int(request.get("ddim_steps", 100))

    def callback(index: int) -> None:
        send({"event": "progress", "step": index + 1, "total": steps})

    verts, faces, info = engine.generate(
        request["lines"],
        ddim_steps=steps,
        eta=float(request.get("eta", 0.0)),
        guidance=float(request.get("guidance", 1.0)),
        seed=int(request.get("seed", 0)),
        iso=float(request.get("iso", 0.005)),
        simplify=float(request.get("simplify", 0.01)),
        up_axis=str(request.get("up_axis", "y")),
        fit_to_sketch=bool(request.get("fit_to_sketch", True)),
        callback=callback,
    )
    output = request["output"]
    mcubes.export_obj(verts, faces, output)
    send({
        "event": "done",
        "output": output,
        "info": {
            "n_strokes": int(info["n_strokes"]),
            "n_tokens": int(info["n_tokens"]),
            "truncated": bool(info["truncated"]),
            "n_verts": int(len(verts)),
        },
    })


if __name__ == "__main__":
    main()
