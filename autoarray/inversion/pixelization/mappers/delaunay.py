from typing import Optional, Tuple
import numpy as np

from autoconf import cached_property

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.settings import Settings
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.interpolator.delaunay import InterpolatorDelaunay
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoarray.inversion.pixelization.mesh_geometry.delaunay import MeshGeometryDelaunay


class MapperDelaunay(AbstractMapper):

    def __init__(
        self,
        mask,
        mesh,
        source_plane_data_grid: Grid2DIrregular,
        source_plane_mesh_grid: Grid2DIrregular,
        regularization: Optional[AbstractRegularization],
        border_relocator: BorderRelocator,
        adapt_data: Optional[np.ndarray] = None,
        settings: Settings = None,
        preloads=None,
        image_plane_mesh_grid: Optional[Grid2DIrregular] = None,
        xp=np,
    ):
        """
        To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` and `pixelization` packages, where
        the four grids are explained (`image_plane_data_grid`, `source_plane_data_grid`,
        `image_plane_mesh_grid`,`source_plane_mesh_grid`)

        If you are unfamliar withe above objects, read through the docstrings of the `pixelization`, `mesh` and
        `image_mesh` packages.

        A `Mapper` determines the mappings between the masked data grid's pixels (`image_plane_data_grid` and
        `source_plane_data_grid`) and the pxelization's pixels (`image_plane_mesh_grid` and `source_plane_mesh_grid`).

        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_plane_data_grid[0]` corresponds to the transformed value
        of `image_plane_data_grid[0]` and so on).

        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
        noting that associations are made by pairing `source_plane_mesh_grid` with `source_plane_data_grid`.

        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `mesh_grid` maps to the index of a pixel on the `grid_slim` as follows:

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the pixelization's 1st pixel.
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel maps to the pixelization's 4th pixel.
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel maps to the pixelization's 2nd pixel.

        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, using a
        `Delaunay` pixelization, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles):

        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the pixelization's 1st pixel.
        - pix_indexes_for_sub_slim_index[0, 1] = 3: the data's 1st sub-pixel also maps to the pixelization's 4th pixel.
        - pix_indexes_for_sub_slim_index[0, 2] = 5: the data's 1st sub-pixel also maps to the pixelization's 6th pixel.

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a pixelization. This matrix is the basis of performing an `Inversion`,
        which reconstructs the data using the `source_plane_mesh_grid`.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The 2D grid of (y,x) centres of every pixelization pixel in the `source` frame.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        adapt_data
            An image which is used to determine the `image_plane_mesh_grid` and therefore adapt the distribution of
            pixels of the Delaunay grid to the data it discretizes.
        mesh_weight_map
            The weight map used to weight the creation of the rectangular mesh grid, which is used for the
            `RectangularBrightness` mesh which adapts the size of its pixels to where the source is reconstructed.
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution,
            which for a mapper smooths neighboring pixels on the mesh.
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        preloads
            The JAX preloads, storing shape information so that JAX knows in advance the shapes of arrays used
            in the mapping matrix and indexes of certain array entries, for example to zero source pixels in the
            linear inversion.
        image_plane_mesh_grid
            The sparse set of (y,x) coordinates computed from the unmasked data in the `data` frame. This has a
            transformation applied to it to create the `source_plane_mesh_grid`.
        """
        super().__init__(
            mask=mask,
            mesh=mesh,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            regularization=regularization,
            border_relocator=border_relocator,
            adapt_data=adapt_data,
            settings=settings,
            preloads=preloads,
            image_plane_mesh_grid=image_plane_mesh_grid,
            xp=xp,
        )

    @property
    def delaunay(self):
        return self.interpolator.delaunay

    @cached_property
    def interpolator(self):
        """
        Return the Delaunay ``source_plane_mesh_grid`` as a ``InterpolatorDelaunay`` object, which provides additional
        functionality for performing operations that exploit the geometry of a Delaunay mesh.

        Parameters
        ----------
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            ``source`` reference frame.
        source_plane_mesh_grid
            The centres of every Delaunay pixel in the ``source`` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the image-plane and applying a transformation
            to this.
        settings
            Settings controlling the pixelization for example if a border is used to relocate its exterior coordinates.
        """
        return InterpolatorDelaunay(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid=self.source_plane_data_grid,
            preloads=self.preloads,
            _xp=self._xp,
        )

    @cached_property
    def mesh_geometry(self):
        return MeshGeometryDelaunay(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid=self.source_plane_data_grid,
            xp=self._xp,
        )
