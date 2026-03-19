from autoarray.plot.wrap.two_d.grid_plot import GridPlot


class ParallelOverscanPlot(GridPlot):
    @property
    def defaults(self):
        return {"c": "k", "linestyle": "-", "linewidth": 1}

    """
    Plots the lines of a parallel overscan `Region2D` object.

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """
