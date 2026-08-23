from .base import LogFn, ProgressFn, SurfacingAdapter
from .bbox import BBoxAdapter
from .neuvas import NeuvasAdapter
from .ns2s import Ns2sAdapter
from .sf3d import Sf3dAdapter
from .trellis import TrellisAdapter
from .trellis2 import Trellis2Adapter
from .vns import VnsAdapter
from .vrs2s import Vrs2sAdapter

# Every adapter, whether or not this machine can run it. Filtering happens
# where methods are *listed* (see `server.health`), not here: a job that names
# an unavailable method should still reach its adapter and get that adapter's
# explanation rather than a bare "unknown method".
ADAPTERS: dict[str, SurfacingAdapter] = {
    adapter.name: adapter
    for adapter in [
        BBoxAdapter(), NeuvasAdapter(), Ns2sAdapter(), Sf3dAdapter(),
        TrellisAdapter(), Trellis2Adapter(), VnsAdapter(), Vrs2sAdapter()
    ]
}

__all__ = ["ADAPTERS", "LogFn", "ProgressFn", "SurfacingAdapter"]
