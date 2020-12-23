from autoarray.plot.wrap_mat import wrap_structure


class OriginScatter(wrap_structure.GridScatter):
    """
    Plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    @classmethod
    def sub(cls, colors=None):
        return OriginScatter(colors=colors, from_subplot_config=True)


class MaskScatter(wrap_structure.GridScatter):
    """
    Plots a mask over an image, using the `Mask2d` object's (y,x) `edge_grid_sub_1` property.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    @classmethod
    def sub(cls, colors=None):
        return MaskScatter(colors=colors, from_subplot_config=True)


class BorderScatter(wrap_structure.GridScatter):
    """
    Plots a border over an image, using the `Mask2d` object's (y,x) `border_grid_sub_1` property.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    @classmethod
    def sub(cls, colors=None):
        return BorderScatter(colors=colors, from_subplot_config=True)


class PositionsScatter(wrap_structure.GridScatter):
    """
    Plots the (y,x) coordinates that are input in a plotter via the `positions` input.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    @classmethod
    def sub(cls, colors=None):
        return PositionsScatter(colors=colors, from_subplot_config=True)


class IndexScatter(wrap_structure.GridScatter):
    """
    Plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return IndexScatter(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class PixelizationGridScatter(wrap_structure.GridScatter):
    """
    Plots the grid of a `Pixelization` object (see `autoarray.inversion`).

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    @classmethod
    def sub(cls, colors=None):
        return PixelizationGridScatter(colors=colors, from_subplot_config=True)


class ParallelOverscanLine(wrap_structure.LinePlot):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(ParallelOverscanLine, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return ParallelOverscanLine(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialPrescanLine(wrap_structure.LinePlot):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialPrescanLine, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialPrescanLine(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class SerialOverscanLine(wrap_structure.LinePlot):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(SerialOverscanLine, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return SerialOverscanLine(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )
