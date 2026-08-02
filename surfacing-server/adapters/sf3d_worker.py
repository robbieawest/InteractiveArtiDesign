"""One piecewise-smooth surface fitting run, inside the sf3d venv.

Run by adapters/sf3d.py as `python sf3d_worker.py <config.json>`. The method's
own drivers (`scripts/run/run_segment_and_fit.py`, `scripts/run/run_projection.py`)
are batch scripts: they walk `metadata.csv`, resolve every path relative to the
repo root, and call `polyscope.init()` unconditionally. None of that survives a
server. So this worker skips them and calls the library underneath directly —
`main.build_proxy.build`, `main.segment_and_fit.SegmentFit`, and
`run.run_projection.run_projection`, all of which take arrays and objects
rather than file paths.

Protocol: one JSON event per line on stdout, everything the method prints goes
to stderr so it cannot corrupt the stream. Events are

    {"event": "log",      "message": str}
    {"event": "progress", "stage": str, "frac": float, "message": str}
    {"event": "mesh",     "path": str, "kind": "proxy"|"snapshot"|"final"}
    {"event": "done"} | {"event": "error", "message": str}

`frac` is this stage's own 0..1 progress; the adapter maps stages onto the
job's progress bar.
"""

import json
import os
import sys
import traceback
from pathlib import Path

# stdout is protocol — hand the method a stderr-backed stdout before it can
# print anything (importing polyscope alone is chatty)
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def send(payload: dict) -> None:
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


def log(message: str) -> None:
    send({"event": "log", "message": str(message)})


def progress(stage: str, frac: float, message: str = "") -> None:
    send({
        "event": "progress",
        "stage": stage,
        "frac": max(0.0, min(1.0, float(frac))),
        "message": message,
    })


def publish(path: Path, kind: str) -> None:
    send({"event": "mesh", "path": str(path), "kind": kind})


# ---------------------------------------------------------------- proxy prep

# The method needs a proxy that is a *manifold* mesh of roughly uniform edge
# length: it stores it in a halfedge structure (PyGEL) and inserts patch
# boundaries into it, and `metadata.csv`'s `proxy_resolution` column exists
# because too coarse a proxy cannot represent the details the strokes ask for.
# The paper remeshes VIPSS output with Instant Meshes; that is a GUI binary
# with a Makefile, so we use MeshLab's isotropic explicit remeshing instead,
# which is the same operation and pip-installable.
def prepare_proxy(raw_path: Path, out_path: Path, edge_length: float) -> Path:
    import pymeshlab

    # pymeshlab renamed AbsoluteValue -> PureValue in 2023.12
    absolute = getattr(pymeshlab, "PureValue", None) or pymeshlab.AbsoluteValue

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(raw_path))
    log(f"proxy in: {ms.current_mesh().vertex_number()} verts, "
        f"{ms.current_mesh().face_number()} faces")

    # filter names have moved between pymeshlab majors; skip what is missing
    # rather than pinning a version the user may not be able to install
    def apply(name: str, **kwargs) -> None:
        fn = getattr(ms, name, None)
        if fn is None:
            log(f"pymeshlab has no filter {name!r}, skipping")
            return
        try:
            fn(**kwargs)
        except Exception as exc:
            log(f"{name} failed ({exc}), continuing")

    apply("meshing_remove_duplicate_vertices")
    apply("meshing_remove_duplicate_faces")
    apply("meshing_remove_null_faces")
    apply("meshing_remove_unreferenced_vertices")
    # marching-cubes output routinely carries small detached blobs; they become
    # extra connected components the segmentation would waste patches on
    apply("meshing_remove_connected_component_by_diameter",
          mincomponentdiag=pymeshlab.PercentageValue(20.0))
    apply("meshing_repair_non_manifold_edges")
    apply("meshing_repair_non_manifold_vertices")
    apply("meshing_isotropic_explicit_remeshing",
          targetlen=absolute(edge_length),
          iterations=6,
          adaptive=False)
    apply("meshing_repair_non_manifold_edges")
    apply("meshing_repair_non_manifold_vertices")

    ms.save_current_mesh(str(out_path), save_vertex_color=False)
    log(f"proxy out: {ms.current_mesh().vertex_number()} verts, "
        f"{ms.current_mesh().face_number()} faces, "
        f"target edge length {edge_length:.5g}")
    return out_path


# --------------------------------------------------------------------- main

def numpy_compat() -> None:
    """Restore the `np.float` / `np.int` aliases numpy 2 removed.

    The method and its pygco submodule are 2021 code and use them in 12
    places. The alternatives are worse: pinning numpy < 1.24 fights every
    other wheel in this env, and editing the vendored submodules means a
    patch that disappears the next time anyone re-clones them. These aliases
    were plain builtins, so restoring them is exactly what the code expects
    and changes no behaviour.
    """
    import numpy as np

    # only the two the code actually uses — numpy reserves `object` and `str`
    # for future scalar types and warns if you so much as look them up
    for name, builtin in (("float", float), ("int", int)):
        if not hasattr(np, name):
            setattr(np, name, builtin)


def run(config: dict) -> None:
    numpy_compat()

    import numpy as np
    import trimesh
    from pygel3d import hmesh

    repo = Path(config["repo"])
    scripts = repo / "scripts"
    # segment_and_fit.py does `sys.path.append('../external/pygco')`, relative
    # to the working directory — so cwd has to be scripts/, not the repo root
    os.chdir(scripts)
    sys.path.insert(0, str(scripts))

    from main.build_proxy import build
    from main.mesh_refinement import remove_islands_from_labelling
    from main.segment_and_fit import SegmentFit
    from utils.loader import try_load_obj_data
    from utils.pygel import get_faces
    import main.mesh_optimization as mesh_optimization
    import run.run_projection as run_projection

    work = Path(config["work_dir"])
    work.mkdir(parents=True, exist_ok=True)

    # --- 1. proxy -----------------------------------------------------------
    progress("proxy", 0.0, "preparing proxy mesh")
    proxy_path = prepare_proxy(
        Path(config["proxy_raw"]), work / "proxy.obj", float(config["edge_length"])
    )
    proxy_mesh = hmesh.load(str(proxy_path))
    if not hmesh.valid(proxy_mesh):
        raise RuntimeError(
            "the proxy mesh is not a valid manifold after remeshing — this "
            "method cannot run on it. Try a different proxy method, or a "
            "coarser proxy resolution."
        )
    trimesh.load(proxy_path).export(work / "proxy.ply")
    publish(work / "proxy.ply", "proxy")
    progress("proxy", 1.0, "proxy ready")

    # --- 2. strokes → proxy association (the paper's "initialization") ------
    progress("init", 0.0, "associating strokes with the proxy")
    data_points, segments = try_load_obj_data(str(config["sketch_obj"]))
    log(f"sketch: {len(data_points)} points, {len(segments)} segments")

    proxy, data_points, segments = build(
        data_points, segments,
        proxy_mesh,
        config["is_symmetric"],
        False,   # open_boundary: needs hand-marked border stroke points
        [],      # on_border_stroke_points
        [],      # ignored_stroke_points
        config["default_edge_weight"],
        config["stroke_edge_weight"],
        config["edge_length_factor_power"],
        True,    # adaptive_graph_simplification
        False,   # display
    )
    progress("init", 1.0, f"{proxy.nodes_count} graph nodes")

    # --- 3. alternate segmentation / model fitting --------------------------
    max_iterations = int(config["max_iterations"])
    sf = SegmentFit(
        proxy,
        "first_order",
        config["w_all"],
        config["w_unary"],
        config["w_smooth"],
        config["w_labels"],
        float(config["sketch_error_dist"]),
        config["lambda_regularization"],
        w_proxy_vertices_fit=0,
        w_normals_fit=config["w_normals_fit"],
    )
    progress("segment", 0.0, f"initializing {config['L0']} candidate models")
    sf.initialize_models(
        int(config["L0"]),
        sample_ratio_bounds=(0.05, 0.1),
        initial_proposal_strategy=3,
        init_lambda_factor=10.0,
        init_normals_factor=10.0,
        random_seed=int(config["random_seed"]),
        degree=int(config["max_model_degree"]),
    )

    iteration = 0
    while iteration < max_iterations:
        sf.optimize_labelling()
        energy = sf.energy_progression[-1] if sf.energy_progression else float("nan")
        patches = len(np.unique(sf.labels_progression[-1]))
        progress(
            "segment",
            (iteration + 1) / max_iterations,
            f"iteration {iteration + 1}/{max_iterations}, "
            f"{patches} patches, energy {energy:.4g}",
        )
        if sf.converged:
            log(f"segmentation converged after {iteration + 1} iteration(s)")
            break
        sf.propose_models(
            max_model_degree=int(config["max_model_degree"]),
            propose_lower_degree=True,
            propose_new=False,
            propose_new_by_split=False,
            propose_new_by_merge=True,
        )
        iteration += 1
    else:
        log(f"segmentation hit the {max_iterations}-iteration cap without "
            "converging; using the last state")
    progress("segment", 1.0, "segmentation done")

    # --- 4. project the proxy onto the fitted models ------------------------
    # a fresh copy: build() works on a graph over the proxy and the projection
    # inserts patch boundaries into the mesh, so it needs the untouched one
    proxy_mesh_initial = hmesh.load(str(proxy_path))

    labels = proxy.property_by_vertex(sf.labels_progression[-1])
    labels = remove_islands_from_labelling(proxy_mesh_initial, labels)
    assigned, reindexed = np.unique(labels, return_inverse=True)
    models = [sf.models_progression[-1][i] for i in assigned]
    in_out = np.asarray(proxy.in_out_labelling, dtype=bool)
    log(f"projecting onto {len(models)} surface model(s)")

    projection_iterations = int(config["projection_iterations"])
    snapshot_every = int(config["snapshot_every"])
    faces: dict[str, object] = {}
    snapshots = {"count": 0}

    # optimize_mesh receives the mesh *after* boundary insertion, which is the
    # topology the optimizer's vertex vector belongs to — capture its faces so
    # a snapshot taken mid-solve is a renderable mesh and not a point soup
    original_optimize_mesh = run_projection.optimize_mesh

    def optimize_mesh(mesh, V, *args, **kwargs):
        faces["F"] = get_faces(mesh)
        return original_optimize_mesh(mesh, V, *args, **kwargs)

    # `minimize` is a module global inside mesh_optimization, so replacing it
    # there is enough to get at L-BFGS's per-iteration callback without
    # touching the method's source
    original_minimize = mesh_optimization.minimize

    def minimize(fun, x0, **kwargs):
        # the method hands L-BFGS the (N, 3) vertex array and relies on scipy
        # ravelling it — scipy raises on a 2-D x0 since 1.11. Its energy and
        # gradient already take the flat vector (they call vec3() on it), so
        # flattening here is what the code means, not a workaround.
        x0 = np.asarray(x0, dtype=float).ravel()
        inner_callback = kwargs.pop("callback", None)
        state = {"i": 0}

        def callback(xk):
            if inner_callback is not None:
                inner_callback(xk)
            state["i"] += 1
            i = state["i"]
            progress(
                "project",
                min(i / max(projection_iterations, 1), 0.99),
                f"optimization step {i}/{projection_iterations}",
            )
            if snapshot_every > 0 and i % snapshot_every == 0 and "F" in faces:
                V = np.asarray(xk, dtype=float).reshape((-1, 3))
                path = work / f"snapshot_{snapshots['count']:04d}.ply"
                try:
                    trimesh.Trimesh(V, faces["F"], process=False).export(path)
                except Exception as exc:  # a snapshot is never worth failing on
                    log(f"snapshot {i} failed ({exc})")
                    return
                snapshots["count"] += 1
                publish(path, "snapshot")

        return original_minimize(fun, x0, callback=callback, **kwargs)

    run_projection.optimize_mesh = optimize_mesh
    mesh_optimization.minimize = minimize
    try:
        progress("project", 0.0, "inserting patch boundaries")
        V_opt, F, all_labels, is_seam, is_sharp, weights, projection_log = (
            run_projection.run_projection(
                proxy_mesh_initial,
                reindexed,
                models,
                proxy.get_stroke_point_by_edge(),
                in_out,
                config["is_symmetric"],
                bool(config["snap_to_strokes"]),
                projection_iterations,
                False,  # exact gradient, as the paper's driver defaults to
                False,  # iprint
            )
        )
    finally:
        run_projection.optimize_mesh = original_optimize_mesh
        mesh_optimization.minimize = original_minimize

    log("insertion {:.1f}s, projection {:.1f}s".format(
        projection_log["insertion_time"], projection_log["projection_time"]))

    final = work / "result.ply"
    trimesh.Trimesh(V_opt, F, process=False).export(final)
    progress("project", 1.0, f"{len(V_opt)} vertices, {len(F)} faces")
    publish(final, "final")


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    try:
        run(config)
        send({"event": "done"})
    except Exception as exc:
        traceback.print_exc()
        send({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        sys.exit(1)


if __name__ == "__main__":
    main()
