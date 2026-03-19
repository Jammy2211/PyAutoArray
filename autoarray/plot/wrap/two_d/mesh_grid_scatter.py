from autoarray.plot.wrap.two_d.grid_scatter import GridScatter


class MeshGridScatter(GridScatter):
    @property
    def defaults(self):
        return {"c": "r", "marker": ".", "s": 2}

    """
    Plots the grid of a `Mesh` object (see `autoarray.inversion`).

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """
