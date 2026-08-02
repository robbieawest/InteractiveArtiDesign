import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from .base import EmitFn, LogFn, ProgressFn, SurfacingAdapter
from .common import (
    JOBS_DIR,
    METHODS_DIR,
    SERVER_DIR,
    WORKER_STOP_TIMEOUT,
    combine_meshes,
    group_strokes_by_part,
    method_env,
    register_resident_worker,
    spawn,
)

METHOD_NAME = "vrs2s"
VRS2S_DIR = METHODS_DIR / "VRSketch2Shape"
# override with the VRS2S_PYTHON env var if the env lives elsewhere
VRS2S_PYTHON = Path(
    os.environ.get("VRS2S_PYTHON", SERVER_DIR / ".venv-vrs2s" / "bin" / "python")
)
# not in the repo — downloaded from https://huggingface.co/YiziChen/sketch2model
VRS2S_CHECKPOINT = Path(
    os.environ.get(
        "VRS2S_CHECKPOINT",
        VRS2S_DIR / "weights" / "all_class" / "df_epoch_best_multicls.pth",
    )
)

WORKER_SCRIPT = Path(__file__).resolve().parent / "vrs2s_worker.py"
# set VRS2S_WORKER=0 to force the one-shot path (each job its own interpreter)
USE_WORKER = os.environ.get("VRS2S_WORKER", "1") != "0"


class Worker:
    """A resident generation process, spawned on first use and reused.

    Same bargain as the ns2s worker: loading transformers, the diffusion UNet
    and the VQVAE — then building the ROCm context on the first `.cuda()` —
    dwarfs the actual sampling, so one process holds the model for the whole
    sweep. Serialized under a lock, which costs nothing for a GPU-bound method
    the benchmark runs one at a time anyway."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _spawn(self, log: LogFn) -> subprocess.Popen:
        log("starting VRSketch2Shape worker (loads the model once)")
        proc = subprocess.Popen(
            [str(VRS2S_PYTHON), str(WORKER_SCRIPT), str(VRS2S_DIR),
             str(VRS2S_CHECKPOINT)],
            cwd=VRS2S_DIR,
            env=method_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # the model layer's chatter goes to the server's own stderr rather
            # than into the protocol stream
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        ready = proc.stdout.readline()
        if not ready:
            raise RuntimeError("VRSketch2Shape worker died during startup")
        event = json.loads(ready)
        if event.get("event") != "ready":
            raise RuntimeError(f"unexpected worker greeting: {event}")
        log(f"worker ready on {event.get('device')}")
        return proc

    def run(
        self,
        request: dict[str, Any],
        log: LogFn,
        on_step: Callable[[int, int], None],
    ) -> dict[str, Any]:
        """Generate one shape, calling `on_step(step, total)` as sampling
        proceeds. Returns the worker's `info` dict."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._spawn(log)
            proc = self._proc
            assert proc.stdin is not None and proc.stdout is not None

            try:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
            except BrokenPipeError:
                # died between jobs; one retry with a fresh process
                self._proc = proc = self._spawn(log)
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()

            while True:
                line = proc.stdout.readline()
                if not line:
                    self._proc = None
                    raise RuntimeError(
                        "VRSketch2Shape worker exited mid-job "
                        "(set VRS2S_WORKER=0 to fall back to one process per job)"
                    )
                event = json.loads(line)
                kind = event.get("event")
                if kind == "progress":
                    on_step(int(event["step"]), int(event["total"]))
                elif kind == "log":
                    log(str(event.get("message", "")))
                elif kind == "done":
                    return dict(event.get("info", {}))
                elif kind == "error":
                    raise RuntimeError(str(event.get("message")))

    def stop(self) -> bool:
        """Terminate the worker and wait for it to go, since its VRAM is only
        returned when the process is actually reaped. True if one was up."""
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None or proc.poll() is not None:
                return False
            proc.terminate()
            try:
                proc.wait(timeout=WORKER_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return True


    def kill(self) -> bool:
        """Terminate without taking the lock.

        `stop()` cannot interrupt a job: `run()` holds the lock for the whole
        exchange, so a stop requested mid-inference would queue behind the
        very thing it means to end. This reaches past that and terminates the
        process; `run()` then reads EOF and fails, which is how the cancelled
        job learns it is over. `_proc` is left for `run()`/`_spawn` to reset —
        both check `poll()` before reusing it."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=WORKER_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return True


WORKER = Worker()
register_resident_worker(METHOD_NAME, WORKER)


class Vrs2sAdapter(SurfacingAdapter):
    """VRSketch2Shape / Sketch2Shape: Order Matters — 3D Shape Generation from
    Sequential VR Sketches (methods/VRSketch2Shape submodule).

    Unlike the other adapters this one does not *fit* a surface to the strokes.
    An order-aware BERT encoder turns the sketch into a conditioning sequence,
    a latent diffusion model samples a shape from it, and a VQVAE decodes that
    to a 64**3 truncated SDF which marching cubes turns into a mesh. The result
    is a plausible whole object, not a surface through the curves — expect it
    to close gaps, invent unsketched geometry and ignore strokes it can't read.

    Two consequences worth knowing before benchmarking against vns/ns2s:

    - Its prior covers the four ShapeNet categories it was trained on
      (airplane, cabinet, chair, table). Sketches outside those — machinery,
      articulated assemblies — get mapped onto the nearest thing it knows.
    - It is stochastic. `seed` selects which sample you get; two runs of the
      same parameters with different seeds are legitimately different shapes.

    Stroke *order* is part of the input, so the document's stroke order is
    preserved end to end."""

    name = METHOD_NAME

    params = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Generate a shape per part and merge, instead of one for "
            "the whole sketch. Rarely what you want here: the model generates "
            "complete objects from a category prior, so a single part usually "
            "comes back as a whole chair or table. Unassigned strokes are "
            "ignored in this mode.",
        },
        {
            "name": "ddim_steps",
            "label": "Sampling steps",
            "type": "int",
            "default": 100,
            "min": 10,
            "max": 250,
            "step": 10,
            "help": "DDIM denoising steps. Runtime is linear in this; 100 is "
            "what the paper evaluates with, and fewer gives a coarser sample.",
        },
        {
            "name": "seed",
            "label": "Seed",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 9999,
            "step": 1,
            "help": "The model samples rather than fits, so the seed picks "
            "which shape you get out of the conditional distribution. Vary it "
            "across runs to see the spread for one sketch.",
        },
        {
            "name": "guidance",
            "label": "Guidance scale",
            "type": "float",
            "default": 1.0,
            "min": 1.0,
            "max": 5.0,
            "step": 0.5,
            "help": "Classifier-free guidance: >1 pushes the sample harder "
            "toward the sketch conditioning. Experimental — the released model "
            "was trained without condition dropout, so its unconditional "
            "branch (an all-zero sketch) is off-distribution and high values "
            "degrade quickly. 1.0 disables it, which is what the paper's own "
            "evaluation does.",
        },
        {
            "name": "simplify",
            "label": "Stroke simplification",
            "type": "float",
            "default": 0.01,
            "min": 0.0,
            "max": 0.1,
            "step": 0.005,
            "help": "Polyline decimation tolerance before tokenizing, as a "
            "fraction of the sketch radius. The model reads at most 1200 "
            "tokens (one per point, plus a separator per stroke), and it was "
            "trained on heavily decimated strokes — 0 keeps every point, which "
            "is unlike anything it saw in training.",
        },
        {
            "name": "iso",
            "label": "Iso level",
            "type": "float",
            "default": 0.005,
            "min": -0.05,
            "max": 0.05,
            "step": 0.005,
            "help": "Level the surface is extracted at from the decoded "
            "truncated SDF (0 is the nominal surface). Slightly positive is "
            "what the paper's evaluation uses; raising it inflates the shape "
            "and closes thin gaps, lowering it thins it out.",
        },
        {
            "name": "up_axis",
            "label": "Sketch up axis",
            "type": "choice",
            "default": "y",
            "choices": ["y", "z"],
            "help": "Which axis is up in the sketch. The model was trained on "
            "ShapeNet-aligned, y-up shapes and has no way to recover from a "
            "rotated input — a z-up sketch comes back as an unrelated shape.",
        },
        {
            "name": "fit_to_sketch",
            "label": "Fit to sketch bounds",
            "type": "bool",
            "default": True,
            "help": "Scale and center the generated mesh so its bounding box "
            "matches the sketch's. Off, the analytic mapping out of the "
            "model's normalized box is used instead, which is truer to the raw "
            "output but often lands beside the strokes rather than on them.",
        },
    ]

    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        import trimesh  # server env

        if not VRS2S_PYTHON.exists():
            raise RuntimeError(
                f"VRSketch2Shape environment not found at {VRS2S_PYTHON} — "
                "set it up per requirements-vrs2s.txt (or point VRS2S_PYTHON at it)"
            )
        if not VRS2S_CHECKPOINT.exists():
            raise RuntimeError(
                f"Sketch2Shape checkpoint not found at {VRS2S_CHECKPOINT} — "
                "download df_epoch_best_multicls.pth from "
                "https://huggingface.co/YiziChen/sketch2model (or point "
                "VRS2S_CHECKPOINT at it)"
            )

        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        request_base = {
            "ddim_steps": int(options.pop("ddim_steps", 100)),
            "seed": int(options.pop("seed", 0)),
            "guidance": float(options.pop("guidance", 1.0)),
            "simplify": float(options.pop("simplify", 0.01)),
            "iso": float(options.pop("iso", 0.005)),
            "up_axis": str(options.pop("up_axis", "y")),
            "fit_to_sketch": bool(options.pop("fit_to_sketch", True)),
        }
        for leftover in options:
            log(f"ignoring unknown option {leftover!r}")

        job_dir = JOBS_DIR / f"vrs2s-{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True)

        # one (label, strokes) unit of work per generation
        units: list[tuple[str, list[dict[str, Any]]]] = []
        if part_based:
            groups, part_names = group_strokes_by_part(sketch, log)
            for part_id, strokes in groups.items():
                units.append((str(part_names.get(part_id, part_id)), strokes))
        else:
            units.append(("sketch", sketch.get("strokes", [])))

        meshes = []
        for index, (label, strokes) in enumerate(units):
            lines = [s["points"] for s in strokes if len(s.get("points", [])) >= 2]
            if not lines:
                log(f"skipping '{label}': no stroke with at least two points")
                continue

            out_path = job_dir / f"{index:02d}.obj"
            span = 0.95 / len(units)
            base = 0.02 + span * index

            def on_step(step: int, total: int, base=base, span=span, label=label) -> None:
                report(base + span * (step / max(total, 1)),
                       f"{label}: sampling {step}/{total}")

            log(f"'{label}': {len(lines)} strokes")
            info = self._generate(
                {**request_base, "lines": lines, "output": str(out_path)},
                log,
                on_step,
            )
            if info.get("truncated"):
                log(f"'{label}': {info['n_tokens']} tokens exceeds the model's "
                    "1200 — the tail of the sketch was cut off (raise "
                    "simplification)")
            log(f"'{label}': {info.get('n_verts', '?')} vertices")

            mesh = trimesh.load(out_path, force="mesh")
            meshes.append(mesh)
            emit(label, mesh.export(file_type="glb"))

        if not meshes:
            raise RuntimeError("VRSketch2Shape produced no shapes")

        report(0.97, "converting result to glb")
        combined = combine_meshes(meshes, log) if len(meshes) > 1 else meshes[0]
        data = combined.export(file_type="glb")
        report(1.0, f"done ({len(meshes)} shape(s))")
        return data

    def _generate(
        self, request: dict[str, Any], log: LogFn, on_step: Callable[[int, int], None]
    ) -> dict[str, Any]:
        if USE_WORKER:
            return WORKER.run(request, log, on_step)
        return self._generate_once(request, log)

    def _generate_once(self, request: dict[str, Any], log: LogFn) -> dict[str, Any]:
        """One interpreter per generation — the pre-worker path, kept for
        VRS2S_WORKER=0 and for debugging the method in isolation. No progress:
        headless.py's CLI runs to completion before it says anything."""
        job_dir = Path(request["output"]).parent
        lines_path = job_dir / (Path(request["output"]).stem + "_lines.json")
        lines_path.write_text(json.dumps(request["lines"]))

        cmd = [
            str(VRS2S_PYTHON), "headless.py", str(lines_path), request["output"],
            "--model_path", str(VRS2S_CHECKPOINT),
            "--ddim_steps", str(request["ddim_steps"]),
            "--guidance", str(request["guidance"]),
            "--seed", str(request["seed"]),
            "--iso", str(request["iso"]),
            "--simplify", str(request["simplify"]),
            "--up_axis", str(request["up_axis"]),
        ]
        if not request.get("fit_to_sketch", True):
            cmd.append("--no_fit")

        log_path = job_dir / (Path(request["output"]).stem + ".log")
        tail: list[str] = []
        with open(log_path, "w") as log_file:
            proc = spawn(
                cmd,
                # cwd matters: headless.py imports models/ and dataloader/
                # relative to the repo root
                cwd=VRS2S_DIR,
                env=method_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                clean = line.rstrip("\n").split("\r")[-1].rstrip()
                if clean:
                    log(clean)
                tail = (tail + [line])[-30:]
            code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"VRSketch2Shape exited with code {code}; last output "
                f"(full log: {log_path}):\n" + "".join(tail)
            )
        return {}
