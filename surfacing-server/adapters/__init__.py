from .base import LogFn, ProgressFn, SurfacingAdapter
from .bbox import BBoxAdapter
from .vns import VnsAdapter

ADAPTERS: dict[str, SurfacingAdapter] = {
    adapter.name: adapter for adapter in [BBoxAdapter(), VnsAdapter()]
}

__all__ = ["ADAPTERS", "LogFn", "ProgressFn", "SurfacingAdapter"]
