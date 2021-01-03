from autoarray.mask import mask_1d
from autoarray.structures import arrays, grids, lines as l, vector_fields
from autoarray.plot.mat_wrap import mat_plot, include as inc
from autoarray.mask import mask_2d
from matplotlib import patches as ptch
import typing


class AbstractVisuals:
    def __add__(self, other):

        for attr, value in other.__dict__.items():
            try:
                if self.__dict__[attr] is None and other.__dict__[attr] is not None:
                    self.__dict__[attr] = other.__dict__[attr]
            except KeyError:
                pass

        return self

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

        # if self.origin is not None:
        #     plotter.origin_scatter.scatter_grid(grid=self.origin)
        #
        # if self.mask is not None:
        #     plotter.mask_scatter.scatter_grid(
        #         grid=self.mask.geometry.edge_grid_sub_1.in_1d_binned
        #     )

        if self.lines is not None:
            plotter.grid_plot.plot_grid_grouped(grid_grouped=self.lines)


class Visuals2D(AbstractVisuals):
    def __init__(
        self,
        mask: mask_2d.Mask2D = None,
        lines: typing.List[l.Line] = None,
        positions: grids.GridIrregular = None,
        grid: grids.Grid = None,
        pixelization_grid: grids.Grid = None,
        vector_field: vector_fields.VectorFieldIrregular = None,
        patches: typing.List[ptch.Patch] = None,
        array_overlay: arrays.Array = None,
        origin: grids.Grid = None,
        border: grids.Grid = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
    ):

        self.mask = mask
        self.lines = lines
        self.positions = positions
        self.grid = grid
        self.pixelization_grid = pixelization_grid
        self.vector_field = vector_field
        self.patches = patches
        self.array_overlay = array_overlay
        self.origin = origin
        self.border = border
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
