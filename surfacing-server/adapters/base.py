from abc import ABC, abstractmethod
from typing import Any, Callable

# report(progress 0..1, message="")
ProgressFn = Callable[..., None]


class SurfacingAdapter(ABC):
    """One surfacing method. Real methods live in their own repos and python
    environments; their adapter invokes them as a subprocess (e.g.
    `conda run -n vns python ...`) and translates sketch → method input and
    method output → glb. This server's own environment stays torch-free."""

    name: str

    @abstractmethod
    def run(
        self, sketch: dict[str, Any], options: dict[str, Any], report: ProgressFn
    ) -> bytes:
        """Run the method to completion and return a binary glTF (.glb).
        Called on a worker thread; call `report` freely along the way."""
