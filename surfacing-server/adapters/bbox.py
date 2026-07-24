import time
from typing import Any, Optional

import numpy as np
import trimesh

from .base import LogFn, ProgressFn, SurfacingAdapter


class BBoxAdapter(SurfacingAdapter):
    """Test method: a padded bounding box per part (plus one for unassigned
    strokes). No ML — it exists to exercise the whole submit/poll/result
    pipeline, including progress reporting and part-aware output (one named
    glb node per part)."""

    name = "bbox"

    params = [
        {
            "name": "pad",
            "label": "Padding",
            "type": "float",
            "default": 0.02,
            "min": 0.0,
            "max": 0.5,
            "step": 0.01,
            "help": "Box padding as a fraction of the part's diagonal",
        },
    ]

    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
    ) -> bytes:
        part_names = {p["id"]: p["name"] for p in sketch.get("parts", [])}
        groups: dict[Optional[str], list[np.ndarray]] = {}
        for stroke in sketch.get("strokes", []):
            points = stroke.get("points", [])
            if points:
                part_id = stroke.get("partId")
                groups.setdefault(part_id, []).append(
                    np.asarray(points, dtype=float)
                )
        if not groups:
            raise ValueError("sketch has no stroke points")

        scene = trimesh.Scene()
        for i, (part_id, arrays) in enumerate(groups.items()):
            name = part_names.get(part_id, "unassigned")
            report(i / len(groups), f"boxing {name}")
            time.sleep(0.5)  # fake optimization time so progress is visible
            points = np.vstack(arrays)
            lo, hi = points.min(axis=0), points.max(axis=0)
            log(f"{name}: {len(arrays)} strokes, {len(points)} points, "
                f"bounds {lo.round(3).tolist()} .. {hi.round(3).tolist()}")
            pad_fraction = float(options.get("pad", 0.02))
            pad = pad_fraction * (float(np.linalg.norm(hi - lo)) or 1.0)
            box = trimesh.creation.box(bounds=np.array([lo - pad, hi + pad]))
            box.visual.face_colors = [255, 170, 60, 140]
            scene.add_geometry(box, node_name=name, geom_name=name)

        report(1.0, "exporting")
        return scene.export(file_type="glb")
