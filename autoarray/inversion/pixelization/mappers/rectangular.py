from typing import Optional, Tuple
import numpy as np

from autoconf import cached_property

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
from autoarray.inversion.pixelization.mesh_grid.rectangular_2d import Mesh2DRectangular
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoarray.inversion.pixelization.mappers import mapper_util


class MapperRectangular(AbstractMapper):
    
    def __init__(
        self,
        mask,
        mesh,
        source_plane_data_grid: Grid2DIrregular,
        source_plane_mesh_grid: Grid2DIrregular,
        regularization: Optional[AbstractRegularization],
        border_relocator: BorderRelocator,
        adapt_data: Optional[np.ndarray] = None,
        settings: SettingsInversion = SettingsInversion(),
        preloads=None,
        mesh_weight_map : Optional[np.ndarray] = None,
        xp=np,
    ):
        """
        To understand a `Mapper` one must be familiar `Mesh` objects and the `mesh` and `pixelization` packages, where
        the four grids are explained (`image_plane_data_grid`, `source_plane_data_grid`,
        `image_plane_mesh_grid`,`source_plane_mesh_grid`)
    
        If you are unfamliar withe above objects, read through the docstrings of the `pixelization`, `mesh` and
        `image_mesh` packages.
    
        A `Mapper` determines the mappings between the masked data grid's pixels (`image_plane_data_grid` and
        `source_plane_data_grid`) and the mesh's pixels (`image_plane_mesh_grid` and `source_plane_mesh_grid`).
    
        The 1D Indexing of each grid is identical in the `data` and `source` frames (e.g. the transformation does not
        change the indexing, such that `source_plane_data_grid[0]` corresponds to the transformed value
        of `image_plane_data_grid[0]` and so on).
    
        A mapper therefore only needs to determine the index mappings between the `grid_slim` and `mesh_grid`,
        noting that associations are made by pairing `source_plane_mesh_grid` with `source_plane_data_grid`.
    
        Mappings are represented in the 2D ndarray `pix_indexes_for_sub_slim_index`, whereby the index of
        a pixel on the `mesh_grid` maps to the index of a pixel on the `grid_slim` as follows:
    
        - pix_indexes_for_sub_slim_index[0, 0] = 0: the data's 1st sub-pixel maps to the mesh's 1st pixel.
        - pix_indexes_for_sub_slim_index[1, 0] = 3: the data's 2nd sub-pixel maps to the mesh's 4th pixel.
        - pix_indexes_for_sub_slim_index[2, 0] = 1: the data's 3rd sub-pixel maps to the mesh's 2nd pixel.
    
        The second dimension of this array (where all three examples above are 0) is used for cases where a
        single pixel on the `grid_slim` maps to multiple pixels on the `mesh_grid`. For example, a
        `Delaunay` triangulation, where every `grid_slim` pixel maps to three Delaunay pixels (the corners of the
        triangles) with varying interpolation weights .
    
        For a `RectangularAdaptDensity` mesh every pixel in the masked data maps to only one pixel, thus the second
        dimension of `pix_indexes_for_sub_slim_index` is always of size 1.
    
        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked data pixel annd the pixels of a mesh. This matrix is the basis of performing an `Inversion`,
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
        mesh_weight_map
            The weight map used to weight the creation of the rectangular mesh grid, which adapts the size of
            the rectangular pixels to be smaller in the brighter regions of the source.
         xp
            The array module (e.g. `numpy` or `jax.numpy`) used to perform calculations and store arrays in the mapper.
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
            xp=xp,
        )
        self.mesh_weight_map = mesh_weight_map

    @property
    def mesh_geometry(self):
        """
        Return the rectangular `source_plane_mesh_grid` as a `Mesh2DRectangular` object, which provides additional
        functionality for perform operatons that exploit the geometry of a rectangular pixelization.

        Parameters
        ----------
        source_plane_data_grid
            The (y,x) grid of coordinates over which the rectangular pixelization is overlaid, where this grid may have
            had exterior pixels relocated to its edge via the border.
        source_plane_mesh_grid
            Not used for a rectangular pixelization, because the pixelization grid in the `source` frame is computed
            by overlaying the `source_plane_data_grid` with the rectangular pixelization.
        """
        return Mesh2DRectangular(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid_over_sampled=self.source_plane_data_grid.over_sampled,
            preloads=self.preloads,
            _xp=self._xp,
        )

    @property
    def shape_native(self) -> Tuple[int, ...]:
        return self.mesh.shape

    @cached_property
    def pix_sub_weights(self) -> PixSubWeights:
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `RectangularAdaptDensity` mesh.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to mesh pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to mesh pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's mesh
        pixel mapping

        These are packaged into the class `PixSubWeights` with attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the mesh pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the RectangularAdaptDensity
        mesh's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the RectangularAdaptDensity
        mesh's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one mesh pixel (for example a `Delaunay` triangulation
        where each data pixel maps to 3 Delaunay triangles with interpolation weights). The weights of multiple mappings
        are stored in the array `pix_weights_for_sub_slim_index`.

        For a RectangularAdaptDensity pixelization each data sub-pixel maps to a single mesh pixel, thus the second
        dimension of the array `pix_indexes_for_sub_slim_index` 1 and all entries in `pix_weights_for_sub_slim_index`
        are equal to 1.0.
        """
        mappings, weights = (
            mapper_util.adaptive_rectangular_mappings_weights_via_interpolation_from(
                source_grid_size=self.shape_native[0],
                source_plane_data_grid=self.source_plane_data_grid.array,
                source_plane_data_grid_over_sampled=self.source_plane_data_grid.over_sampled,
                mesh_weight_map=self.mesh_weight_map,
                xp=self._xp,
            )
        )

        return PixSubWeights(
            mappings=mappings,
            sizes=4 * self._xp.ones(len(mappings), dtype="int"),
            weights=weights,
        )

    @property
    def areas_transformed(self):
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `mesh_util.rectangular_neighbors_from`.
        """
        return mapper_util.adaptive_rectangular_areas_from(
            source_grid_shape=self.shape_native,
            source_plane_data_grid=self.source_plane_data_grid.array,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )

    @property
    def areas_for_magnification(self):
        """
        The area of every pixel in the rectangular pixelization.

        Returns
        -------
        ndarray
            The area of every pixel in the rectangular pixelization.
        """
        return self.areas_transformed

    @property
    def edges_transformed(self):
        """
        A class packing the ndarrays describing the neighbors of every pixel in the rectangular pixelization (see
        `Neighbors` for a complete description of the neighboring scheme).

        The neighbors of a rectangular pixelization are computed by exploiting the uniform and symmetric nature of the
        rectangular grid, as described in the method `mesh_util.rectangular_neighbors_from`.
        """

        # edges defined in 0 -> 1 space, there is one more edge than pixel centers on each side
        edges_y = self._xp.linspace(1, 0, self.shape_native[0] + 1)
        edges_x = self._xp.linspace(0, 1, self.shape_native[1] + 1)

        edges_reshaped = self._xp.stack([edges_y, edges_x]).T

        return mapper_util.adaptive_rectangular_transformed_grid_from(
            source_plane_data_grid=self.source_plane_data_grid.array,
            grid=edges_reshaped,
            mesh_weight_map=self.mesh_weight_map,
            xp=self._xp,
        )
