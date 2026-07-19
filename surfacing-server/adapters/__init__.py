from .base import ProgressFn, SurfacingAdapter
from .bbox import BBoxAdapter

ADAPTERS: dict[str, SurfacingAdapter] = {
    adapter.name: adapter for adapter in [BBoxAdapter()]
}

__all__ = ["ADAPTERS", "ProgressFn", "SurfacingAdapter"]
