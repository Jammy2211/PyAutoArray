import copy
import numpy as np
from typing import Dict, Optional

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.sparse_2d import Grid2DSparse
from autoarray.preloads import Preloads
from autoarray.inversion.pixelization.mappers.mapper_grids import MapperGrids
from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh
from autoarray.inversion.pixelization.settings import SettingsPixelization

from autoarray.numba_util import profile_func


class Triangulation(AbstractMesh):
    def mapper_grids_from(
        self,
        source_grid_slim: Grid2D,
        source_mesh_grid: Grid2DSparse = None,
        data_mesh_grid: Grid2DSparse = None,
        hyper_data: np.ndarray = None,
        settings=SettingsPixelization(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ) -> MapperGrids:
        """
        Mapper objects describe the mappings between pixels in the masked 2D data and the pixels in a mesh,
        in both the `data` and `source` frames.

        This function returns a `MapperVoronoiNoInterp` as follows:

        1) Before this routine is called, a sparse grid of (y,x) coordinates are computed from the 2D masked data,
        the `data_mesh_grid`, which acts as the Voronoi pixel centres of the mesh and mapper.

        2) Before this routine is called, operations are performed on this `data_mesh_grid` that transform it
        from a 2D grid which overlaps with the 2D mask of the data in the `data` frame to an irregular grid in
        the `source` frame, the `source_mesh_grid`.

        3) If `settings.use_border=True`, the border of the input `source_grid_slim` is used to relocate all of the
        grid's (y,x) coordinates beyond the border to the edge of the border.

        4) If `settings.use_border=True`, the border of the input `source_grid_slim` is used to relocate all of the
        transformed `source_mesh_grid`'s (y,x) coordinates beyond the border to the edge of the border.

        5) Use the transformed `source_mesh_grid`'s (y,x) coordinates as the centres of the Voronoi
        mesh.

        6) Return the `MapperVoronoiNoInterp`.

        Parameters
        ----------
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_mesh_grid
            The centres of every Voronoi pixel in the `source` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the `data` frame and applying a transformation
            to this.
        data_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_mesh_grid`.
        hyper_data
            Not used for a rectangular mesh.
        settings
            Settings controlling the mesh for example if a border is used to relocate its exterior coordinates.
        preloads
            Object which may contain preloaded arrays of quantities computed in the mesh, which are passed via
            this object speed up the calculation.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.profiling_dict = profiling_dict

        relocated_source_grid_slim = self.relocated_grid_from(
            source_grid_slim=source_grid_slim, settings=settings, preloads=preloads
        )

        relocated_source_mesh_grid = self.relocated_mesh_grid_from(
            source_grid_slim=source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            settings=settings,
        )

        try:

            source_mesh_grid = self.mesh_grid_from(
                source_grid_slim=relocated_source_grid_slim,
                source_mesh_grid=relocated_source_mesh_grid,
                sparse_index_for_slim_index=source_mesh_grid.sparse_index_for_slim_index,
            )
        except ValueError as e:
            raise e

        return MapperGrids(
            source_grid_slim=relocated_source_grid_slim,
            source_mesh_grid=source_mesh_grid,
            data_mesh_grid=data_mesh_grid,
            hyper_data=hyper_data,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def relocated_mesh_grid_from(
        self,
        source_grid_slim: Grid2D,
        source_mesh_grid: Grid2DSparse,
        settings: SettingsPixelization = SettingsPixelization(),
    ):
        """
        Relocates all coordinates of the input `source_mesh_grid` that are outside of a border (which
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
        source_grid_slim
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_mesh_grid
            The centres of every Voronoi pixel in the `source` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the `data` frame and applying a transformation
            to this.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """
        if settings.use_border:
            return source_grid_slim.relocated_mesh_grid_from(mesh_grid=source_mesh_grid)
        return source_mesh_grid
