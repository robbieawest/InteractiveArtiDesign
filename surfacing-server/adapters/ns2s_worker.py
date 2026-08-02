"""Resident NeuralSketch2Surf inference process.

Run in the ns2s venv by adapters/ns2s.py and kept alive between jobs. The
point is the ~15s of setup a fresh interpreter pays before it can do ~1s of
work — almost all of it ROCm context creation on the first `.cuda()`. Loading
the model once and holding it turns a benchmark over N sketches from N*16s
into 15s + N*1s.

Protocol: one JSON request per line on stdin, one JSON event per line on
stdout. Requests are

    {"input": <dir>, "output": <dir>, "threshold": f, "margin": f,
     "img_size": i}

and the reply is one {"event": "file", "name": ..., "ok": bool} per .obj
found, then {"event": "done", "count": n} — or {"event": "error", ...} if the
whole request failed. Anything the model prints goes to stderr so it cannot
corrupt the stream.
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
