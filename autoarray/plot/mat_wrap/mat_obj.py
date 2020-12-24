from autoarray.plot.mat_wrap import mat_structure


class AbstractMatObj:
    @property
    def config_folder(self):
        return "mat_obj"


class OriginScatter(AbstractMatObj, mat_structure.GridScatter):
    """
    Plots the (y,x) coordinates of the origin of a data structure (e.g. as a black cross).

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class MaskScatter(AbstractMatObj, mat_structure.GridScatter):
    """
    Plots a mask over an image, using the `Mask2d` object's (y,x) `edge_grid_sub_1` property.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class BorderScatter(AbstractMatObj, mat_structure.GridScatter):
    """
    Plots a border over an image, using the `Mask2d` object's (y,x) `border_grid_sub_1` property.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class PositionsScatter(AbstractMatObj, mat_structure.GridScatter):
    """
    Plots the (y,x) coordinates that are input in a plotter via the `positions` input.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class IndexScatter(AbstractMatObj, mat_structure.GridScatter):
    """
    Plots specific (y,x) coordinates of a grid (or grids) via their 1d or 2d indexes.

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class PixelizationGridScatter(AbstractMatObj, mat_structure.GridScatter):
    """
    Plots the grid of a `Pixelization` object (see `autoarray.inversion`).

    See `mat_structure.Scatter` for a description of how matplotlib is wrapped to make this plot.
    """

    pass


class ParallelOverscanPlot(AbstractMatObj, mat_structure.LinePlot):
    pass


class SerialPrescanPlot(AbstractMatObj, mat_structure.LinePlot):
    pass


class SerialOverscanPlot(AbstractMatObj, mat_structure.LinePlot):
    pass
