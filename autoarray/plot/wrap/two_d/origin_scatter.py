from autoarray.plot.wrap.two_d.grid_scatter import GridScatter


class OriginScatter(GridScatter):
    @property
    def defaults(self):
        return {"c": "k", "marker": "x", "s": 80}

    """
    Plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """
