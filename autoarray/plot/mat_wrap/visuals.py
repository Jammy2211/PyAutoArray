from abc import ABC
from matplotlib import patches as ptch
import numpy as np
from typing import List, Optional, Union

from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.grids.one_d.grid_1d import Grid1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d import Grid2D
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.vector_fields.vector_field_irregular import (
    VectorField2DIrregular,
)
from autoarray.plot.mat_wrap.include import Include1D


class AbstractVisuals(ABC):
    def __add__(self, other):
        """
        Adds two `Visuals` classes together.

        When we perform plotting, the `Include` class is used to create additional `Visuals` class from the data
        structures that are plotted, for example:

        mask = Mask2D.circular(shape_native=(100, 100), pixel_scales=0.1, radius=3.0)
        array = Array2D.ones(shape_native=(100, 100), pixel_scales=0.1)
        masked_array = al.Array2D.manual_mask(array=array, mask=mask)
        include_2d = Include2D(mask=True)
        array_plotter = aplt.Array2DPlotter(array=masked_array, include_2d=include_2d)
        array_plotter.figure()

        Because `mask=True` in `Include2D` the function `figure` extracts the `Mask2D` from the `masked_array`
        and plots it. It does this by creating a new `Visuals2D` object.

        If the user did not manually input a `Visuals2d` object, the one created in `function_array` is the one used to
        plot the image

        However, if the user specifies their own `Visuals2D` object and passed it to the plotter, e.g.:

        visuals_2d = Visuals2D(origin=(0.0, 0.0))
        include_2d = Include2D(mask=True)
        array_plotter = aplt.Array2DPlotter(array=masked_array, include_2d=include_2d)

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
        """

        for attr, value in self.__dict__.items():
            try:
                if other.__dict__[attr] is None and self.__dict__[attr] is not None:
                    other.__dict__[attr] = self.__dict__[attr]
            except KeyError:
                pass

        return other


class Visuals1D(AbstractVisuals):
    def __init__(
        self,
        mask: Optional[Mask1D] = None,
        origin: Optional[Grid1D] = None,
        points: Optional[Grid1D] = None,
        vertical_line: Optional[float] = None,
        shaded_region: Optional[
            Union[List[List], List[Array1D], List[np.ndarray]]
        ] = None,
    ):

        self.mask = mask
        self.origin = origin
        self.points = points
        self.vertical_line = vertical_line
        self.shaded_region = shaded_region

    @property
    def include(self):
        return Include1D()

    def plot_via_plotter(self, plotter):

        if self.points is not None:

            plotter.yx_scatter.scatter_yx(y=self.points, x=np.arange(len(self.points)))

        if self.vertical_line is not None:

            plotter.vertical_line_axvline.axvline_vertical_line(
                vertical_line=self.vertical_line
            )


class Visuals2D(AbstractVisuals):
    def __init__(
        self,
        origin: Grid2D = None,
        mask: Mask2D = None,
        border: Grid2D = None,
        lines: List[Array1D] = None,
        positions: Union[Grid2DIrregular, List[Grid2DIrregular]] = None,
        grid: Grid2D = None,
        pixelization_grid: Grid2D = None,
        vector_field: VectorField2DIrregular = None,
        patches: List[ptch.Patch] = None,
        array_overlay: Array2D = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        indexes=None,
        pixelization_indexes=None,
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
        self.indexes = indexes
        self.pixelization_indexes = pixelization_indexes

    def plot_via_plotter(self, plotter, grid_indexes=None, mapper=None):

        if self.origin is not None:
            plotter.origin_scatter.scatter_grid(grid=Grid2DIrregular(grid=self.origin))

        if self.mask is not None:
            plotter.mask_scatter.scatter_grid(grid=self.mask.edge_grid_sub_1.binned)

        if self.border is not None:
            plotter.border_scatter.scatter_grid(grid=self.border)

        if self.grid is not None:
            plotter.grid_scatter.scatter_grid(grid=self.grid)

        if self.pixelization_grid is not None:
            plotter.pixelization_grid_scatter.scatter_grid(grid=self.pixelization_grid)

        if self.positions is not None:
            plotter.positions_scatter.scatter_grid(grid=self.positions)

        if self.vector_field is not None:
            plotter.vector_field_quiver.quiver_vector_field(
                vector_field=self.vector_field
            )

        if self.patches is not None:
            plotter.patch_overlay.overlay_patches(patches=self.patches)

        if self.lines is not None:
            plotter.grid_plot.plot_grid(grid=self.lines)

        if self.indexes is not None:
            plotter.index_scatter.scatter_grid_indexes(
                grid=grid_indexes, indexes=self.indexes
            )

        if self.pixelization_indexes is not None and mapper is not None:
            indexes = mapper.slim_indexes_from_pixelization_indexes(
                pixelization_indexes=self.pixelization_indexes
            )

            plotter.index_scatter.scatter_grid_indexes(
                grid=mapper.source_grid_slim, indexes=indexes
            )
