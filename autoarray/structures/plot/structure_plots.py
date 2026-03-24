"""
Thin convenience aliases that forward directly to the core plot functions.

``plot_array``, ``plot_grid``, and ``plot_yx`` now accept autoarray objects
natively, so these wrappers exist only for name-discoverability.
"""
from autoarray.plot.plots.array import plot_array as plot_array_2d
from autoarray.plot.plots.grid import plot_grid as plot_grid_2d
from autoarray.plot.plots.yx import plot_yx as plot_yx_1d

__all__ = ["plot_array_2d", "plot_grid_2d", "plot_yx_1d"]
