from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

# report(progress 0..1, message="")
ProgressFn = Callable[..., None]
# log(line) — free-form text streamed to the client's log window
LogFn = Callable[[str], None]
# emit(name, glb) — a piece of the result that is already final (typically one
# part), published while the job keeps running so the client can show geometry
# before the whole thing is done. Optional: a method with nothing meaningful to
# show until the end simply never calls it.
EmitFn = Callable[[str, bytes], None]


class SurfacingAdapter(ABC):
    """One surfacing method. Real methods live in their own repos and python
    environments; their adapter invokes them as a subprocess (e.g.
    `conda run -n vns python ...`) and translates sketch → method input and
    method output → glb. This server's own environment stays torch-free."""

    name: str

    # Whether running this method needs the GPU. A GPU method evicts every
    # other method's resident worker before it starts (see
    # `common.release_other_workers`); a CPU-only one leaves them alone, so a
    # trivial run never costs a benchmark its warm worker.
    uses_gpu: bool = True

    # User-editable parameters, rendered generically by the Surfacer panel
    # and passed back (by `name`) inside the job's `options` dict. Each spec:
    #   { name, label, type: "int"|"float"|"bool"|"choice", default,
    #     min?, max?, step?, choices?, help?,
    #     enabledWhen?: { param, equals } }
    # enabledWhen greys the input out until another param holds `equals`
    # (e.g. a per-part control that unlocks when a "part-based" toggle is on).
    params: list[dict[str, Any]] = []

    # Rendered views of the sketch, for methods that condition on images
    # rather than consuming the strokes as geometry. None (the default) means
    # this method wants none and the client sends nothing extra.
    #
    #   { "selector": <param name> | None,
    #     "specs": { <param value> | "*": { size, count, pitch,
    #                                       strokeColor, strokeThickness,
    #                                       margin } } }
    #
    # The client renders to the spec its current options select and sends the
    # PNGs as `options["views"]` — a flat list, or {part name: [...]} when the
    # run is part-based. A selector exists because a method can offer several
    # conditioning strategies wanting different renders; without one, "*" is
    # used. A strategy that needs no images simply has no entry.
    #
    # The numbers live here, not in the client's renderer, because they are
    # the *method's* requirements: TRELLIS wants light thick strokes at 518px
    # because of what its preprocessing and DINOv2 do to them, which is a fact
    # about TRELLIS and not about drawing sketches. The renderer stays generic
    # and the client needs no per-method knowledge.
    view_spec: Optional[dict[str, Any]] = None

    @abstractmethod
    def run(
        self,
        sketch: dict[str, Any],
        options: dict[str, Any],
        report: ProgressFn,
        log: LogFn,
        emit: EmitFn,
    ) -> bytes:
        """Run the method to completion and return a binary glTF (.glb).
        Called on a worker thread. `report(progress, message)` drives the
        progress bar; `log(line)` streams any text to the UI log window;
        `emit(name, glb)` publishes a finished piece of the result (a part)
        while the rest is still running."""
