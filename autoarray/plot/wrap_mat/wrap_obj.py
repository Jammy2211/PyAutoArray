from autoarray.plot.wrap_mat import wrap_structure


class AbstractWrapObj:
    @property
    def config_folder(self):
        return "wrap_obj"


class OriginScatter(AbstractWrapObj, wrap_structure.GridScatter):
    """
    Plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class MaskScatter(AbstractWrapObj, wrap_structure.GridScatter):
    """
    Plots a mask over an image, using the `Mask2d` object's (y,x) `edge_grid_sub_1` property.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class BorderScatter(AbstractWrapObj, wrap_structure.GridScatter):
    """
    Plots a border over an image, using the `Mask2d` object's (y,x) `border_grid_sub_1` property.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class PositionsScatter(AbstractWrapObj, wrap_structure.GridScatter):
    """
    Plots the (y,x) coordinates that are input in a plotter via the `positions` input.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class IndexScatter(AbstractWrapObj, wrap_structure.GridScatter):
    """
    Plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class PixelizationGridScatter(AbstractWrapObj, wrap_structure.GridScatter):
    """
    Plots the grid of a `Pixelization` object (see `autoarray.inversion`).

    See `wrap_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class ParallelOverscanLine(AbstractWrapObj, wrap_structure.LinePlot):
    pass


class SerialPrescanLine(AbstractWrapObj, wrap_structure.LinePlot):
    pass


class SerialOverscanLine(AbstractWrapObj, wrap_structure.LinePlot):
    pass
