from autoarray.plot.wrap.two_d.grid_scatter import GridScatter


class IndexScatter(GridScatter):
    @property
    def defaults(self):
        return {"c": "r,g,b,m,y,k", "marker": ".", "s": 20}

    """
    Plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass
