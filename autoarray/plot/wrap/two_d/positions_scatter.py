from autoarray.plot.wrap.two_d.grid_scatter import GridScatter


class PositionsScatter(GridScatter):
    @property
    def defaults(self):
        return {"c": "k,m,y,b,r,g", "marker": ".", "s": 32}

    """
    Plots the (y,x) coordinates that are input in a plotter via the `positions` input.

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass
