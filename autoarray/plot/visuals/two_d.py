from matplotlib import patches as ptch
import numpy as np
from typing import List, Optional, Union

from autoarray.mask.mask_2d import Mask2D
from autoarray.plot.visuals.abstract import AbstractVisuals
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.vectors.irregular import VectorYX2DIrregular


class Visuals2D(AbstractVisuals):
    def __init__(
        self,
        origin: Optional[Grid2D] = None,
        mask: Optional[Mask2D] = None,
        border: Optional[Grid2D] = None,
        lines: Optional[Union[List[Array1D], Grid2DIrregular]] = None,
        positions: Optional[Union[Grid2DIrregular, List[Grid2DIrregular]]] = None,
        grid: Optional[Grid2D] = None,
        mesh_grid: Optional[Grid2D] = None,
        vectors: Optional[VectorYX2DIrregular] = None,
        patches: Optional[List[ptch.Patch]] = None,
        fill_region: Optional[List] = None,
        array_overlay: Optional[Array2D] = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        indexes=None,
    ):
        self.origin = origin
        self.mask = mask
        self.border = border
        self.lines = lines
        self.positions = positions
        self.grid = grid
        self.mesh_grid = mesh_grid
        self.vectors = vectors
        self.patches = patches
        self.fill_region = fill_region
        self.array_overlay = array_overlay
        self.parallel_overscan = parallel_overscan
        self.serial_prescan = serial_prescan
        self.serial_overscan = serial_overscan
        self.indexes = indexes

    def plot_via_plotter(self, plotter, grid_indexes=None):

        if self.mask is not None:
            plotter.mask_scatter.scatter_grid(grid=self.mask.derive_grid.edge.array)

        if self.origin is not None:

            origin = self.origin

            if isinstance(origin, tuple):

                origin = Grid2DIrregular(values=[origin])

            plotter.origin_scatter.scatter_grid(
                grid=Grid2DIrregular(values=origin).array
            )

        if self.border is not None:
            try:
                plotter.border_scatter.scatter_grid(grid=self.border.array)
            except AttributeError:
                plotter.border_scatter.scatter_grid(grid=self.border)

        if self.grid is not None:
            try:
                plotter.grid_scatter.scatter_grid(grid=self.grid.array)
            except AttributeError:
                plotter.grid_scatter.scatter_grid(grid=self.grid)

        if self.mesh_grid is not None:
            plotter.mesh_grid_scatter.scatter_grid(grid=self.mesh_grid.array)

        if self.positions is not None:
            try:
                plotter.positions_scatter.scatter_grid(grid=self.positions.array)
            except (AttributeError, ValueError):
                plotter.positions_scatter.scatter_grid(grid=self.positions)

        if self.vectors is not None:
            plotter.vector_yx_quiver.quiver_vectors(vectors=self.vectors)

        if self.patches is not None:
            plotter.patch_overlay.overlay_patches(patches=self.patches)

        if self.fill_region is not None:
            plotter.fill.plot_fill(fill_region=self.fill_region)

        if self.lines is not None:
            plotter.grid_plot.plot_grid(grid=self.lines)

        if self.indexes is not None and grid_indexes is not None:

            plotter.index_scatter.scatter_grid_indexes(
                grid=np.array(grid_indexes),
                indexes=self.indexes,
            )
