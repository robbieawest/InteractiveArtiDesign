from abc import ABC, abstractmethod
from typing import Any, Callable

# report(progress 0..1, message="")
ProgressFn = Callable[..., None]
# log(line) — free-form text streamed to the client's log window
LogFn = Callable[[str], None]


class SurfacingAdapter(ABC):
    """One surfacing method. Real methods live in their own repos and python
    environments; their adapter invokes them as a subprocess (e.g.
    `conda run -n vns python ...`) and translates sketch → method input and
    method output → glb. This server's own environment stays torch-free."""

    name: str

    # User-editable parameters, rendered generically by the Surfacer panel
    # and passed back (by `name`) inside the job's `options` dict. Each spec:
    #   { name, label, type: "int"|"float"|"bool"|"choice", default,
    #     min?, max?, step?, choices?, help?,
    #     enabledWhen?: { param, equals } }
    # enabledWhen greys the input out until another param holds `equals`
    # (e.g. a per-part control that unlocks when a "part-based" toggle is on).
    params: list[dict[str, Any]] = []

    @abstractmethod
    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
    ) -> bytes:
        """Run the method to completion and return a binary glTF (.glb).
        Called on a worker thread. `report(progress, message)` drives the
        progress bar; `log(line)` streams any text to the UI log window."""
