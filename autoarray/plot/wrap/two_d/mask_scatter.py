from autoarray.plot.wrap.two_d.grid_scatter import GridScatter


class MaskScatter(GridScatter):
    @property
    def defaults(self):
        return {"c": "k", "marker": "x", "s": 10}

    """
    Plots a mask over an image, using the `Mask2d` object's (y,x) `edge` property.

    See `wrap.base.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """
