import hashlib
import json
import os
import re
import subprocess
import threading
import uuid
from collections import OrderedDict
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
    write_curve_obj,
)

METHOD_NAME = "ns2s"
NS2S_DIR = METHODS_DIR / "NeuralSketch2Surf"
# override with the NS2S_PYTHON env var if the env lives elsewhere
NS2S_PYTHON = Path(
    os.environ.get("NS2S_PYTHON", SERVER_DIR / ".venv-ns2s" / "bin" / "python")
)
# not in the repo — downloaded from https://huggingface.co/HongshengY/S2V_Net
NS2S_CHECKPOINT = Path(
    os.environ.get("NS2S_CHECKPOINT", NS2S_DIR / "checkpoints" / "best_model_jit.pt")
)

# the network is trained at 112**3 with feature_size 24; both are fixed by the
# checkpoint, so they are constants here rather than user-facing parameters
IMG_SIZE = 112
FEATURE_SIZE = 24

WORKER_SCRIPT = Path(__file__).resolve().parent / "ns2s_worker.py"
# set NS2S_WORKER=0 to force the one-shot path (each job its own interpreter)
USE_WORKER = os.environ.get("NS2S_WORKER", "1") != "0"

# Container for a probability grid on its way to the client: magic, u32
# version, u32 header length, the JSON header, then the u8 payload. Same shape
# as the TRELLIS flow bundle for the same reason — a fixed-stride block of
# voxels wants a length-prefixed header in front of it, not a JSON array.
BUNDLE_MAGIC = b"NSVX"
BUNDLE_VERSION = 1
BUNDLE_KIND = "ns2s-volume"

# tqdm's bar over the input files, e.g. " 50%|#####     | 1/2 [00:01<00:00, ...]"
PROGRESS_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[")


class Worker:
    """A resident inference process, spawned on first use and reused.

    Setup costs ~15s (almost all of it the first `.cuda()` building the ROCm
    context) against ~1s of actual inference, so a fresh interpreter per job
    means a benchmark spends most of its time starting up. One process holds
    the model instead.

    Jobs run on server worker threads, so the exchange is serialized under a
    lock — which costs nothing here, since the methods are GPU-bound and the
    benchmark runs them one at a time anyway."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _spawn(self, log: LogFn) -> subprocess.Popen:
        log("starting NeuralSketch2Surf worker (loads the model once)")
        proc = subprocess.Popen(
            [str(NS2S_PYTHON), str(WORKER_SCRIPT), str(NS2S_DIR),
             str(NS2S_CHECKPOINT), str(IMG_SIZE), str(FEATURE_SIZE)],
            cwd=NS2S_DIR,
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
            raise RuntimeError("NeuralSketch2Surf worker died during startup")
        event = json.loads(ready)
        if event.get("event") != "ready":
            raise RuntimeError(f"unexpected worker greeting: {event}")
        log(f"worker ready on {event.get('device')}")
        return proc

    def run(
        self,
        request: dict[str, Any],
        log: LogFn,
        on_file: Callable[[str, bool, int], None],
    ) -> int:
        """Surface every .obj in the request's input dir, calling
        `on_file(name, ok, index)` as each one lands. Returns how many were
        processed."""
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

            index = 0
            while True:
                line = proc.stdout.readline()
                if not line:
                    self._proc = None
                    raise RuntimeError(
                        "NeuralSketch2Surf worker exited mid-job "
                        "(set NS2S_WORKER=0 to fall back to one process per job)"
                    )
                event = json.loads(line)
                kind = event.get("event")
                if kind == "file":
                    on_file(event["name"], bool(event.get("ok")), index)
                    index += 1
                elif kind == "log":
                    log(str(event.get("message", "")))
                elif kind == "done":
                    return int(event.get("count", index))
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


def require_environment() -> None:
    """Fail early and legibly when the method is not installed here."""
    if not NS2S_PYTHON.exists():
        raise RuntimeError(
            f"NeuralSketch2Surf environment not found at {NS2S_PYTHON} — "
            "set it up per requirements-ns2s.txt (or point NS2S_PYTHON at it)"
        )
    if not NS2S_CHECKPOINT.exists():
        raise RuntimeError(
            f"S2V-Net checkpoint not found at {NS2S_CHECKPOINT} — download "
            "it from https://huggingface.co/HongshengY/S2V_Net (or point "
            "NS2S_CHECKPOINT at it)"
        )


# Fields kept from the runs that just happened, so a second consumer of the
# same prediction does not pay for it twice. The case this exists for: TRELLIS
# conditions on renders of an ns2s surface, which the *client* has to make
# before it can submit its job — so it asks for the mesh in its own job, and
# the TRELLIS job that follows wants the same field to inpaint with. One
# forward pass, two jobs, and the alternative is a second 13GB model load and
# another minute of GPU for an answer already computed.
#
# Keyed by the bytes of the .obj that was predicted from, plus everything that
# changes the prediction. That is the whole sketch, exactly: both callers write
# their .obj with `write_curve_obj`, so identical strokes give identical bytes
# and different strokes cannot collide. A miss is only ever a re-solve, never a
# wrong answer, which is what lets the key be this blunt.
FIELD_CACHE: "OrderedDict[tuple[str, int, float, float], tuple[Any, dict[str, Any]]]" = (
    OrderedDict()
)
# a 112^3 u8 field is 1.4MB; this bounds the cache at a few tens of MB
FIELD_CACHE_LIMIT = 16


def _field_key(obj_bytes: bytes, margin: float, blur: float) -> tuple:
    return (
        hashlib.sha256(obj_bytes).hexdigest(), IMG_SIZE, float(margin),
        float(blur),
    )


def cache_field(key: tuple, field: Any, header: dict[str, Any]) -> None:
    FIELD_CACHE[key] = (field, header)
    FIELD_CACHE.move_to_end(key)
    while len(FIELD_CACHE) > FIELD_CACHE_LIMIT:
        FIELD_CACHE.popitem(last=False)


def read_probability_files(payload: Path, meta: Path) -> tuple[Any, dict[str, Any]]:
    """The two files the worker writes, as `(field, header)` — the field a
    `[x, y, z]` u8 array in the method's own normalized cube."""
    import numpy as np  # server env

    header = json.loads(meta.read_text())
    grid = int(header.get("grid", IMG_SIZE))
    # the file is in 3D-texture order (x fastest); undo that to get the
    # [x, y, z] array the prediction was made in
    field = np.frombuffer(payload.read_bytes(), dtype=np.uint8).reshape(
        grid, grid, grid
    ).transpose(2, 1, 0)
    return field, header


def probability_grids(
    sketches: dict[str, dict[str, Any]],
    work_dir: Path,
    margin: float,
    blur: float,
    log: LogFn,
    before_predict: Optional[Callable[[], None]] = None,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Predict one occupancy field per named sketch, for another adapter.

    This is the surfacing method used as a *component* rather than as a
    result: TRELLIS calls it to build an inpainting constraint out of a
    predicted surface instead of only the strokes. Everything about the
    prediction is the same as a probability-volume run — same worker, same
    request — so the field a caller inpaints with is byte-identical to the one
    the viewer would have drawn.

    Returns `{name: (field, header)}` with the field as a `[x, y, z]` u8 array
    in the method's own normalized cube, which `header["align"]` maps back to
    world. Names whose sketch could not be written or predicted are absent
    rather than raising — a caller that still has the strokes has something to
    fall back on.

    Fields already in `FIELD_CACHE` are taken from there. If every sketch hits,
    nothing is loaded and nothing runs — which is the difference between a
    surfaced-condition TRELLIS run costing one ns2s solve and costing two.

    The caller owns the GPU question: this loads the ns2s model (~13GB
    resident), so anything that needs the card afterwards should
    `release_other_workers` once the grids are in hand. `before_predict` runs
    immediately before the model would load and not at all when the cache
    covers everything, which is where that eviction belongs — paying it for a
    prediction that never happens would tear down a resident worker for
    nothing.
    """
    in_dir = work_dir / "ns2s-in"
    out_dir = work_dir / "ns2s-out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    grids: dict[str, tuple[Any, dict[str, Any]]] = {}
    stems: dict[str, str] = {}
    keys: dict[str, tuple] = {}
    for index, (name, sketch) in enumerate(sketches.items()):
        stem = f"unit_{index}"
        path = in_dir / f"{stem}.obj"
        try:
            write_curve_obj(sketch, path)
        except ValueError as exc:
            log(f"ns2s: nothing to surface for '{name}': {exc}")
            continue
        key = _field_key(path.read_bytes(), margin, blur)
        cached = FIELD_CACHE.get(key)
        if cached is not None:
            FIELD_CACHE.move_to_end(key)
            log(f"ns2s: reusing the occupancy field already predicted for "
                f"'{name}'")
            grids[name] = cached
            # the worker predicts every .obj in the directory, so a hit has to
            # take its input back out again or it is solved anyway
            path.unlink()
            continue
        keys[stem] = key
        stems[stem] = name
    if not stems:
        return grids

    require_environment()
    if before_predict is not None:
        before_predict()
    log(f"ns2s: predicting {len(stems)} occupancy field(s) "
        f"(blur {blur:g} voxel(s))")
    WORKER.run(
        {
            "input": str(in_dir),
            "output": str(out_dir),
            # unused in volume mode: nothing is thresholded on this side
            "threshold": 0.5,
            "margin": margin,
            "img_size": IMG_SIZE,
            "volume": True,
            "blur": blur,
        },
        log,
        lambda name, ok, index: None,
    )

    for stem, name in stems.items():
        payload = out_dir / f"{stem}_prob.u8"
        meta = out_dir / f"{stem}_prob.json"
        if not (payload.exists() and meta.exists()):
            log(f"ns2s: no occupancy field for '{name}'")
            continue
        grids[name] = read_probability_files(payload, meta)
        cache_field(keys[stem], *grids[name])
    return grids


class Ns2sAdapter(SurfacingAdapter):
    """NeuralSketch2Surf: Fast Neural Surfacing of Unoriented 3D Sketches
    (methods/NeuralSketch2Surf submodule).

    Voxelizes the sketch to a 112**3 grid, predicts an occupancy field with
    S2V-Net (SwinUNETR-V2 backbone + residual refinement, both baked into the
    TorchScript checkpoint), and extracts the surface with marching cubes.
    Seconds per sketch rather than VNS's minutes, but it targets *closed*
    surfaces — open sheets and stray decorative strokes are outside what it
    was trained for.

    Whole-object (default) surfaces all strokes at once; part-based fits each
    part separately and merges. The method normalizes each input internally
    and maps the mesh back, so output is already in sketch world coordinates
    — and because every part gets its own full 112**3 grid, part-based mode
    also raises the effective resolution.

    Optional smoothing runs the paper's fidelity-vs-smoothness post-process
    (smooth.py) headlessly on each result.

    In probability-volume mode the run stops one step earlier and publishes
    the occupancy field itself instead of a mesh — the client raymarches it
    over the sketch, and the threshold marching cubes would have baked in
    becomes a slider. The same field is what an inpainting signal is built
    from, which is the reason it can leave here at all."""

    name = METHOD_NAME

    params = [
        {
            "name": "part_based",
            "label": "Part-based",
            "type": "bool",
            "default": False,
            "help": "Surface each part separately, then merge the meshes, "
            "instead of fitting one surface to the whole sketch. Each part "
            "gets its own 112³ grid, so small parts come out sharper. "
            "Unassigned strokes are ignored in this mode.",
        },
        {
            "name": "probability_volume",
            "label": "Probability volume",
            "type": "bool",
            "default": False,
            "help": "Stop at the network's output and return the per-voxel "
            "occupancy probabilities instead of a mesh. No marching cubes "
            "runs, so no threshold is chosen for you: the client raymarches "
            "the field over the sketch and the threshold becomes something "
            "you move while looking at it. This run produces no surface.",
        },
        {
            "name": "blur",
            "label": "Volume blur",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 4.0,
            "step": 0.1,
            "enabledWhen": {"param": "probability_volume", "equals": True},
            "help": "Gaussian smoothing of the probability field, in voxels, "
            "applied before it is quantized and sent. Blurring here rather "
            "than in the viewer is what makes the picture and the numbers "
            "agree: this is the field, so anything reading it downstream — "
            "an inpainting constraint, a threshold — reads the smoothed one. "
            "The viewer's own blur slider still works, and stacks on top.",
        },
        {
            "name": "threshold",
            "label": "Surface threshold",
            "type": "float",
            "default": 0.6,
            "min": 0.05,
            "max": 0.95,
            "step": 0.05,
            "help": "Occupancy probability the surface is extracted at. Lower "
            "gives a fatter surface that bridges gaps more readily; higher "
            "hugs only confident interior and may drop thin features. If no "
            "voxel reaches it the prediction counts as empty and the run fails.",
            "enabledWhen": {"param": "probability_volume", "equals": False},
        },
        {
            "name": "margin",
            "label": "Grid margin",
            "type": "float",
            "default": 1.2,
            "min": 1.0,
            "max": 2.0,
            "step": 0.05,
            "help": "Extent of the voxel grid relative to the sketch bounding "
            "box (which is normalized to ±1). 1.0 makes the grid hug the "
            "strokes exactly, so a surface bulging outside them gets clipped; "
            "larger is safer but spends fewer voxels on the shape itself.",
        },
        {
            "name": "smooth",
            "label": "Smooth result",
            "type": "bool",
            "default": False,
            "help": "Run the paper's post-process: Laplacian smoothing "
            "balanced against attraction to the sketch curves, then hole "
            "filling, normal repair and Taubin smoothing.",
            "enabledWhen": {"param": "probability_volume", "equals": False},
        },
        {
            "name": "smoothness",
            "label": "Fidelity vs smooth",
            "type": "float",
            "default": 0.5,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "1.0 adheres closely to the sketch strokes, 0.0 gives the "
            "smoother surface; values in between blend the two.",
            "enabledWhen": {"param": "smooth", "equals": True},
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

        require_environment()

        options = dict(options)
        part_based = bool(options.pop("part_based", False))
        volume = bool(options.pop("probability_volume", False))
        # Internal, not a user parameter: mesh as usual but keep the field the
        # mesh was extracted from, in FIELD_CACHE, for whoever asked. TRELLIS's
        # surfaced image condition sets it — the client needs the mesh to
        # render before it can submit its own job, and the TRELLIS job that
        # follows wants the same prediction to inpaint with.
        with_volume = bool(options.pop("with_volume", False)) and not volume
        blur = float(options.pop("blur", 0.0))
        smooth = bool(options.pop("smooth", False))
        smoothness = float(options.pop("smoothness", 0.5))
        threshold = float(options.pop("threshold", 0.6))
        margin = float(options.pop("margin", 1.2))
        for leftover in options:
            log(f"ignoring unknown option {leftover!r}")

        job_dir = JOBS_DIR / f"ns2s-{uuid.uuid4().hex[:8]}"
        in_dir = job_dir / "in"
        out_dir = job_dir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        # one .obj per unit of work; inference.py takes a whole directory, so
        # the part-based case is still a single subprocess (and a single model
        # load) no matter how many parts there are
        names: list[str] = []
        labels: dict[str, str] = {}
        if part_based:
            groups, part_names = group_strokes_by_part(sketch, log)
            for i, (part_id, strokes) in enumerate(groups.items()):
                label = part_names.get(part_id, part_id)
                stem = f"part_{i}"
                try:
                    write_curve_obj({"strokes": strokes}, in_dir / f"{stem}.obj")
                except ValueError as exc:
                    log(f"skipping part '{label}': {exc}")
                    continue
                log(f"part '{label}': {len(strokes)} strokes -> {stem}.obj")
                names.append(stem)
                labels[stem] = str(label)
            if not names:
                raise ValueError("no part had enough stroke points to surface")
        else:
            write_curve_obj(sketch, in_dir / "sketch.obj")
            log(f"input: {len(sketch.get('strokes', []))} strokes -> sketch.obj")
            names.append("sketch")
            labels["sketch"] = "sketch"

        # one subprocess handles every input, so partials come from watching
        # its output directory rather than from the call returning
        emitted: set[str] = set()

        def publish_finished() -> None:
            for stem in names:
                if stem in emitted:
                    continue
                if volume:
                    # the .json is written after the payload, so its presence
                    # means the grid is complete rather than half-flushed
                    payload = out_dir / f"{stem}_prob.u8"
                    meta = out_dir / f"{stem}_prob.json"
                    if payload.exists() and meta.exists():
                        emitted.add(stem)
                        emit(labels[stem], self._bundle(payload, meta, log),
                             BUNDLE_KIND)
                    continue
                recon = out_dir / f"{stem}_recon.obj"
                # the .npz is written after the .obj, so its presence means
                # the mesh file is complete rather than half-flushed
                if recon.exists() and (out_dir / f"{stem}_data.npz").exists():
                    emitted.add(stem)
                    # unsmoothed: this is a progress preview, the returned
                    # result is the smoothed one when smoothing is on
                    emit(labels[stem], trimesh.load(recon, force="mesh")
                         .export(file_type="glb"))

        def on_file(frac: float, msg: str) -> None:
            report(0.03 + 0.82 * frac, msg)
            publish_finished()

        report(0.03, f"running S2V-Net on {len(names)} sketch(es)")
        self._run_inference(
            in_dir, out_dir, threshold, margin, volume, blur, log, on_file,
            len(names), with_volume
        )
        publish_finished()

        if with_volume:
            self._cache_fields(in_dir, out_dir, names, margin, blur, log)

        if volume:
            # Nothing was meshed, so there is nothing to return as geometry:
            # the grids went out as artifacts above and the client renders
            # them itself. An empty mesh keeps the protocol's promise of a glb
            # without pretending a surface exists — it exports as nodes with no
            # accessors, which every loader reads as "nothing here".
            # (trimesh.Scene() cannot stand in: it refuses to export empty.)
            if not emitted:
                raise RuntimeError(
                    "NeuralSketch2Surf produced no probability grid — every "
                    "input failed to voxelize (too few stroke points?)"
                )
            report(1.0, f"done ({len(emitted)} probability grid(s), no mesh)")
            return trimesh.Trimesh().export(file_type="glb")

        # inference.py reports per-file failures on stdout and still exits 0,
        # so the missing outputs are the only reliable signal
        results: list[Path] = []
        for stem in names:
            recon = out_dir / f"{stem}_recon.obj"
            if recon.exists():
                results.append(recon)
            else:
                log(f"no surface produced for {stem} (empty prediction?)")
        if not results:
            raise RuntimeError(
                "NeuralSketch2Surf produced no surfaces — the prediction was "
                "empty for every input (try a lower surface threshold)"
            )

        if smooth:
            report(0.85, f"smoothing {len(results)} surface(s)")
            results = [
                self._smooth(in_dir / f"{p.name[: -len('_recon.obj')]}.obj",
                             p, smoothness, log)
                for p in results
            ]

        report(0.95, "converting result to glb")
        meshes = [trimesh.load(p, force="mesh") for p in results]
        combined = combine_meshes(meshes, log) if part_based else meshes[0]
        data = combined.export(file_type="glb")
        report(1.0, f"done ({len(meshes)} surface(s))")
        return data

    def _cache_fields(
        self,
        in_dir: Path,
        out_dir: Path,
        names: list[str],
        margin: float,
        blur: float,
        log: LogFn,
    ) -> None:
        """Put this run's fields where `probability_grids` will find them.

        Keyed on the .obj that produced each one, which is the same file the
        other caller writes for the same strokes — so the key matches when the
        sketch does and misses harmlessly when it does not.
        """
        kept = 0
        for stem in names:
            payload = out_dir / f"{stem}_prob.u8"
            meta = out_dir / f"{stem}_prob.json"
            source = in_dir / f"{stem}.obj"
            if not (payload.exists() and meta.exists() and source.exists()):
                continue
            cache_field(
                _field_key(source.read_bytes(), margin, blur),
                *read_probability_files(payload, meta),
            )
            kept += 1
        if kept:
            log(f"kept {kept} probability field(s) for the run that asked "
                "for them")

    def _run_inference(
        self,
        in_dir: Path,
        out_dir: Path,
        threshold: float,
        margin: float,
        volume: bool,
        blur: float,
        log: LogFn,
        on_file: Callable[[float, str], None],
        total: int = 1,
        with_volume: bool = False,
    ) -> None:
        if USE_WORKER or volume or with_volume:
            # The probability path exists only in our worker script —
            # upstream's inference.py has no flag that stops before marching
            # cubes — so it overrides NS2S_WORKER=0 rather than silently
            # handing back meshes instead of grids.
            if (volume or with_volume) and not USE_WORKER:
                log("probability volume needs the worker; ignoring NS2S_WORKER=0")
            self._run_inference_worker(
                in_dir, out_dir, threshold, margin, volume, blur, log, on_file,
                total, with_volume
            )
            return
        self._run_inference_once(in_dir, out_dir, threshold, margin, log, on_file)

    def _run_inference_worker(
        self,
        in_dir: Path,
        out_dir: Path,
        threshold: float,
        margin: float,
        volume: bool,
        blur: float,
        log: LogFn,
        on_file: Callable[[float, str], None],
        total: int,
        with_volume: bool = False,
    ) -> None:
        """Hand the request to the resident worker. Progress comes from its
        per-file events rather than from scraping tqdm."""
        def handle(name: str, ok: bool, index: int) -> None:
            if not ok:
                log(f"nothing produced for {name} " + (
                    "(the sketch would not voxelize?)" if volume
                    else "(empty prediction?)"))
            done = index + 1
            on_file(done / max(total, 1), f"sketch {done}/{total}")

        WORKER.run(
            {
                "input": str(in_dir),
                "output": str(out_dir),
                "threshold": threshold,
                "margin": margin,
                "img_size": IMG_SIZE,
                # `with_volume` means "volume as well as the mesh": the worker
                # marches the same field it quantizes, so the surface a caller
                # renders and the grid it inpaints with are the same prediction
                # at the same level
                "volume": volume or with_volume,
                "mesh": with_volume,
                "blur": blur,
            },
            log,
            handle,
        )

    def _run_inference_once(
        self,
        in_dir: Path,
        out_dir: Path,
        threshold: float,
        margin: float,
        log: LogFn,
        on_file: Callable[[float, str], None],
    ) -> None:
        """One interpreter per job — the pre-worker path, kept for
        NS2S_WORKER=0 and for debugging the method in isolation."""
        cmd = [
            str(NS2S_PYTHON), "inference.py",
            "--model_path", str(NS2S_CHECKPOINT),
            "--input_dir", str(in_dir),
            "--output_dir", str(out_dir),
            "--threshold", str(threshold),
            "--img_size", str(IMG_SIZE),
            "--margin", str(margin),
            # --eager is our fork's flag: rebuild the network from source and
            # load the released .pt's weights into it. The published trace
            # bakes a CPU constant into the SwinV2 logit_scale clamp, which
            # recent torch refuses to mix with cuda tensors — without this the
            # GPU path dies. Output is identical to the trace on CPU.
            "--eager",
        ]
        self._run(cmd, out_dir / "inference.log", "inference", log, on_file)

    @staticmethod
    def _bundle(payload: Path, meta: Path, log: LogFn) -> bytes:
        """Pack one probability grid for the client: magic, JSON header,
        voxels.

        The header carries the alignment because only the method knows it —
        the grid is written in the normalization `voxelize_strict_aligned`
        chose, and without the way back the cloud would float somewhere near
        the sketch at the wrong size. See `save_probability_grid` in
        ns2s_worker.py for where those numbers come from."""
        header = json.loads(meta.read_text())
        voxels = payload.read_bytes()
        grid = int(header.get("grid", IMG_SIZE))
        expected = grid ** 3
        if len(voxels) != expected:
            raise RuntimeError(
                f"probability grid is {len(voxels)} bytes, expected "
                f"{expected} for a {grid}^3 field"
            )
        log(f"probability grid: {grid}^3 ({len(voxels) / 1e6:.1f} MB), "
            f"max {header.get('max', 0):.3f}, mean {header.get('mean', 0):.4f}")
        encoded = json.dumps(header).encode()
        return b"".join([
            BUNDLE_MAGIC,
            BUNDLE_VERSION.to_bytes(4, "little"),
            len(encoded).to_bytes(4, "little"),
            encoded,
            voxels,
        ])

    def _smooth(
        self, sketch_obj: Path, mesh_obj: Path, ratio: float, log: LogFn
    ) -> Path:
        """Post-process one reconstruction against its own sketch curves.
        Returns the smoothed mesh, or the original if smoothing failed —
        a rough surface beats no surface."""
        out_path = mesh_obj.with_name(mesh_obj.stem + "_smooth.obj")
        cmd = [
            str(NS2S_PYTHON), "smooth.py", str(sketch_obj), str(mesh_obj),
            "--headless", "--ratio", str(ratio), "--output", str(out_path),
        ]
        try:
            self._run(cmd, out_path.with_suffix(".log"), "smoothing", log)
        except RuntimeError as exc:
            log(f"{exc}\nkeeping the unsmoothed surface for {mesh_obj.name}")
            return mesh_obj
        if not out_path.exists():
            log(f"smoothing produced no output for {mesh_obj.name}; keeping it")
            return mesh_obj
        return out_path

    @staticmethod
    def _run(
        cmd: list[str],
        log_path: Path,
        what: str,
        log: LogFn,
        on_file: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Run one method subprocess, streaming its output to the client's log
        window and scraping tqdm for progress."""
        tail: list[str] = []
        with open(log_path, "w") as log_file:
            proc = spawn(
                cmd,
                # cwd matters: inference.py imports train112TVloss and network/
                # relative to the repo root
                cwd=NS2S_DIR,
                env=method_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                # keep only the final state of \r-refreshing progress bars
                clean = line.rstrip("\n").split("\r")[-1].rstrip()
                if clean:
                    log(clean)
                tail = (tail + [line])[-30:]
                match = on_file and PROGRESS_RE.search(clean)
                if match:
                    done, total = match.groups()
                    on_file(
                        int(done) / max(int(total), 1), f"sketch {done}/{total}"
                    )
            code = proc.wait()
        if code != 0:
            raise RuntimeError(
                f"NeuralSketch2Surf {what} exited with code {code}; last output "
                f"(full log: {log_path}):\n" + "".join(tail)
            )
