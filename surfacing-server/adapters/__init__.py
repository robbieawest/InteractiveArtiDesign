from .base import LogFn, ProgressFn, SurfacingAdapter
from .bbox import BBoxAdapter
from .neuvas import NeuvasAdapter
from .ns2s import Ns2sAdapter
from .sf3d import Sf3dAdapter
from .vns import VnsAdapter
from .vrs2s import Vrs2sAdapter

ADAPTERS: dict[str, SurfacingAdapter] = {
    adapter.name: adapter
    for adapter in [
        BBoxAdapter(), NeuvasAdapter(), Ns2sAdapter(), Sf3dAdapter(),
        VnsAdapter(), Vrs2sAdapter()
    ]
}

__all__ = ["ADAPTERS", "LogFn", "ProgressFn", "SurfacingAdapter"]
