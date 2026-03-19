from autoarray.plot.wrap.two_d.grid_plot import GridPlot


class IndexPlot(GridPlot):
    @property
    def defaults(self):
        return {"c": "r,g,b,m,y,k", "linewidth": 3}

    """
    Plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass
