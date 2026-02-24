import numpy as np
from typing import Optional

from autoarray.settings import Settings
from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class AbstractMesh:
    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    def relocated_grid_from(
        self, border_relocator: BorderRelocator, source_plane_data_grid: Grid2D, xp=np
    ) -> Grid2D:
        """
        Relocates all coordinates of the input `source_plane_data_grid` that are outside of a
        border (which is defined by a grid of (y,x) coordinates) to the edge of this border.

        The border is determined from the mask of the 2D data in the `data` frame before any transformations of the
        data's grid are performed. The border is all pixels in this mask that are pixels at its extreme edge. These
        pixel indexes are used to then determine a grid of (y,x) coordinates from the transformed `source_grid_grid` in
        the `source` reference frame, whereby points located outside of it are relocated to the border's edge.

        A full description of relocation is given in the method grid_2d.relocated_grid_from()`.

        This is used in the project PyAutoLens to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D (y,x) grid of coordinates, whose coordinates outside the border are relocated to its edge.
        """
        if border_relocator is not None:
            return border_relocator.relocated_grid_from(
                grid=source_plane_data_grid, xp=xp
            )

        return Grid2D(
            values=source_plane_data_grid.array,
            mask=source_plane_data_grid.mask,
            over_sample_size=source_plane_data_grid.over_sampler.sub_size,
            over_sampled=source_plane_data_grid.over_sampled,
            over_sampler=source_plane_data_grid.over_sampler,
            xp=xp,
        )

    @property
    def zeroed_pixels_to_keep(self):
        """
        Return the positive indices of pixels that should be kept (solved for),
        accounting for zeroed pixels specified using Python-style negative indexing.

        This property assumes that `self.zeroed_pixels` contains **negative indices**
        referring to entries counted from the right-hand side of the parameter array
        (e.g. -1 is the last entry, -2 the second-to-last, etc.).

        These negative indices are converted to their corresponding positive indices
        before constructing a boolean mask over the full set of mapper indices.

        Returns
        -------
        np.ndarray
            A 1D array of positive indices corresponding to pixels that are *not*
            zeroed and should therefore be included in the solve.
        """
        # Negative indices from the right (e.g. [-1, -2, ...])
        ids_zeros_neg = np.array(self.zeroed_pixels, dtype=int)

        # Total number of values being solved for
        n_values = self.pixels

        # Convert negative indices to positive
        ids_zeros_pos = n_values + ids_zeros_neg

        values_to_solve = np.ones(n_values, dtype=bool)
        values_to_solve[ids_zeros_pos] = False

        return np.where(values_to_solve)[0]

    def relocated_mesh_grid_from(
        self,
        border_relocator: Optional[BorderRelocator],
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        xp=np,
    ):
        """
        Relocates all coordinates of the input `source_plane_mesh_grid` that are outside of a border (which
        is defined by a grid of (y,x) coordinates) to the edge of this border.

        The border is determined from the mask of the 2D data in the `data` frame before any transformations of the
        data's grid are performed. The border is all pixels in this mask that are pixels at its extreme edge. These
        pixel indexes are used to then determine a grid of (y,x) coordinates from the transformed `source_grid_grid` in
        the `source` reference frame, whereby points located outside of it are relocated to the border's edge.

        A full description of relocation is given in the method grid_2d.relocated_grid_from()`.

        This is used in the project `PyAutoLens` to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The centres of every pixel in the `source` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the `data` frame and applying a transformation
            to this.
        """
        if border_relocator is not None:
            return border_relocator.relocated_mesh_grid_from(
                grid=source_plane_data_grid, mesh_grid=source_plane_mesh_grid, xp=xp
            )
        return source_plane_mesh_grid

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        raise NotImplementedError

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))
