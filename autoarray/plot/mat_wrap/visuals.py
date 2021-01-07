from autoarray.mask import mask_1d
from autoarray.structures import arrays, grids, lines as l, vector_fields
from autoarray.plot.mat_wrap import mat_plot, include as inc
from autoarray.mask import mask_2d
from matplotlib import patches as ptch
import typing


class AbstractVisuals:
    def __add__(self, other):
        """
        Adds two `Visuals` classes together.

        When we perform plotting, the `Include` class is used to create additional `Visuals` class from the data
        structures that are plotted, for example:

        mask = Mask2D.circular(shape_2d=(100, 100), pixel_scales=0.1, radius=3.0)
        array = Array.ones(shape_2d=(100, 100), pixel_scales=0.1)
        masked_array = al.Array.manual_mask(array=array, mask=mask)
        include_2d = Include2D(mask=True)
        array_plotter = aplt.ArrayPlotter(array=masked_array, include_2d=include_2d)
        array_plotter.figure_array()

        Because `mask=True` in `Include2D` the function `figure_array` extracts the `Mask2D` from the `masked_array`
        and plots it. It does this by creating a new `Visuals2D` object.

        If the user did not manually input a `Visuals2d` object, the one created in `function_array` is the one used to
        plot the image

        However, if the user specifies their own `Visuals2D` object and passed it to the plotter, e.g.:

        visuals_2d = Visuals2D(origin=(0.0, 0.0))
        include_2d = Include2D(mask=True)
        array_plotter = aplt.ArrayPlotter(array=masked_array, include_2d=include_2d)

        We now wish for the `Plotter` to plot the `origin` in the user's input `Visuals2D` object and the `Mask2d`
        extracted via the `Include2D`. To achieve this, two `Visuals2D` objects are created: (i) the user's input
        instance (with an origin) and; (ii) the one created by the `Include2D` object (with a mask).

        This `__add__` override means we can add the two together to make the final `Visuals2D` object that is
        plotted on the figure containing both the `origin` and `Mask2D`.:

        visuals_2d = visuals_2d_via_user + visuals_2d_via_include

        The ordering of the addition has been specifically chosen to ensure that the `visuals_2d_via_user` does not
        retain the attributes that are added to it by the `visuals_2d_via_include`. This ensures that if multiple plots
        are made, the same `visuals_2d_via_user` is used for every plot. If this were not the case, it would
        permenantly inherit attributes from the `Visuals` from the `Include` method and plot them on all figures.

        Parameters
        ----------
        other

        Returns
        -------

        """

        for attr, value in self.__dict__.items():
            try:
                if other.__dict__[attr] is None and self.__dict__[attr] is not None:
                    other.__dict__[attr] = self.__dict__[attr]
            except KeyError:
                pass

        return other

    @property
    def plotter(self):
        raise NotImplementedError()

    @property
    def include(self):
        raise NotImplementedError()

    def plot_via_plotter(self, plotter):
        raise NotImplementedError()


class Visuals1D(AbstractVisuals):
    def __init__(
        self,
        mask: mask_1d.Mask1D = None,
        lines: typing.List[l.Line] = None,
        origin: grids.Grid = None,
    ):

        self.mask = mask
        self.lines = lines
        self.origin = origin

    @property
    def plotter(self):
        return mat_plot.MatPlot1D()

    @property
    def include(self):
        return inc.Include1D()

    def plot_via_plotter(self, plotter):

        if self.lines is not None:
            plotter.grid_plot.plot_grid_grouped(grid_grouped=self.lines)


class Visuals2D(AbstractVisuals):
    def __init__(
        self,
        origin: grids.Grid = None,
        mask: mask_2d.Mask2D = None,
        border: grids.Grid = None,
        lines: typing.List[l.Line] = None,
        positions: grids.GridIrregular = None,
        grid: grids.Grid = None,
        pixelization_grid: grids.Grid = None,
        vector_field: vector_fields.VectorFieldIrregular = None,
        patches: typing.List[ptch.Patch] = None,
        array_overlay: arrays.Array = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        self.origin = origin
        self.mask = mask
        self.border = border
        self.lines = lines
        self.positions = positions
        self.grid = grid
        self.pixelization_grid = pixelization_grid
        self.vector_field = vector_field
        self.patches = patches
        self.array_overlay = array_overlay
        self.parallel_overscan = parallel_overscan
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan

    def plot_via_plotter(self, plotter):

        if self.origin is not None:
            plotter.origin_scatter.scatter_grid(grid=self.origin)

        if self.mask is not None:
            plotter.mask_scatter.scatter_grid(
                grid=self.mask.geometry.edge_grid_sub_1.in_1d_binned
            )

        if self.border is not None:
            plotter.border_scatter.scatter_grid(grid=self.border)

        if self.grid is not None:
            plotter.grid_scatter.scatter_grid(grid=self.grid)

        if self.pixelization_grid is not None:
            plotter.pixelization_grid_scatter.scatter_grid(grid=self.pixelization_grid)

        if self.positions is not None:
            plotter.positions_scatter.scatter_grid_grouped(grid_grouped=self.positions)

        if self.vector_field is not None:
            plotter.vector_field_quiver.quiver_vector_field(
                vector_field=self.vector_field
            )

        if self.patches is not None:
            plotter.patch_overlay.overlay_patches(patches=self.patches)

        if self.lines is not None:
            plotter.grid_plot.plot_grid_grouped(grid_grouped=self.lines)
