from autoarray.plot.wrap.two_d.grid_scatter import GridScatter


class BorderScatter(GridScatter):
    @property
    def defaults(self):
        return {"c": "r", "marker": ".", "s": 30}

    """
    Plots a border over an image, using the `Mask2d` object's (y,x) `border` property.

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass
