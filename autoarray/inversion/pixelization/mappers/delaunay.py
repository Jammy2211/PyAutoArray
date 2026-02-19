from typing import Optional, Tuple
import numpy as np

from autoconf import cached_property

from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.inversion.pixelization.mappers.abstract import PixSubWeights
from autoarray.inversion.pixelization.mesh_grid.delaunay_2d import Mesh2DDelaunay
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization.border_relocator import BorderRelocator
from autoarray.inversion.pixelization.mappers import mapper_util

def triangle_area_xp(c0, c1, c2, xp):
    """
    Twice triangle area using vector cross product magnitude.
    Calling via xp ensures NumPy or JAX backend operation.
    """
    v0 = c1 - c0  # (..., 2)
    v1 = c2 - c0
    cross = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
    return xp.abs(cross)


def pixel_weights_delaunay_from(
    source_plane_data_grid,  # (N_sub, 2)
    source_plane_mesh_grid,  # (N_pix, 2)
    pix_indexes_for_sub_slim_index,  # (N_sub, 3), padded with -1
    xp=np,  # backend: np (default) or jnp
):
    """
    XP-compatible (NumPy/JAX) version of pixel_weights_delaunay_from.

    Computes barycentric weights for Delaunay triangle interpolation.
    """

    N_sub = pix_indexes_for_sub_slim_index.shape[0]

    # -----------------------------
    # CASE MASKS
    # -----------------------------
    # If pix_indexes_for_sub_slim_index[sub][1] == -1 → NOT in simplex
    has_simplex = pix_indexes_for_sub_slim_index[:, 1] != -1  # (N_sub,)

    # -----------------------------
    # GATHER TRIANGLE VERTICES
    # -----------------------------
    # Clip negatives (for padded entries) so that indexing doesn't crash
    safe_indices = pix_indexes_for_sub_slim_index.clip(min=0)

    # (N_sub, 3, 2)
    vertices = source_plane_mesh_grid[safe_indices]

    p0 = vertices[:, 0]  # (N_sub, 2)
    p1 = vertices[:, 1]
    p2 = vertices[:, 2]

    # Query points
    q = source_plane_data_grid  # (N_sub, 2)

    # -----------------------------
    # TRIANGLE AREAS (barycentric numerators)
    # -----------------------------
    a0 = triangle_area_xp(p1, p2, q, xp)
    a1 = triangle_area_xp(p0, p2, q, xp)
    a2 = triangle_area_xp(p0, p1, q, xp)

    area_sum = a0 + a1 + a2

    # (N_sub, 3)
    weights_bary = xp.stack([a0, a1, a2], axis=1) / area_sum[:, None]

    # -----------------------------
    # NEAREST-NEIGHBOUR CASE
    # -----------------------------
    # For no-simplex: weight = [1,0,0]
    weights_nn = xp.stack(
        [
            xp.ones(N_sub),
            xp.zeros(N_sub),
            xp.zeros(N_sub),
        ],
        axis=1,
    )

    # -----------------------------
    # SELECT BETWEEN CASES
    # -----------------------------
    pixel_weights = xp.where(has_simplex[:, None], weights_bary, weights_nn)

    return pixel_weights


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
        settings: SettingsInversion = SettingsInversion(),
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
            xp=xp,
        )
        self.image_plane_mesh_grid = image_plane_mesh_grid

    @property
    def delaunay(self):
        return self.mesh_geometry.delaunay

    @property
    def mesh_geometry(self):
        """
        Return the Delaunay ``source_plane_mesh_grid`` as a ``Mesh2DDelaunay`` object, which provides additional
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
        return Mesh2DDelaunay(
            mesh=self.mesh,
            mesh_grid=self.source_plane_mesh_grid,
            data_grid_over_sampled=self.source_plane_data_grid.over_sampled,
            preloads=self.preloads,
            _xp=self._xp,
        )

    @cached_property
    def pix_sub_weights(self) -> PixSubWeights:
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Delaunay` pixelization.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to pixelization pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to pixelization pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's pixelization
        pixel mapping

        These are packaged into the class `PixSubWeights` with attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the RectangularAdaptDensity
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the RectangularAdaptDensity
        pixelization's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one pixelization pixel.

        For a `Delaunay` pixelization each data pixel maps to 3 Delaunay triangles with interpolation, for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Delaunay
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[0, 1] = 5`: The data's first (index 0) sub-pixel also maps to the Delaunay
        pixelization's sixth (index 5) pixel.

        - `pix_indexes_for_sub_slim_index[0, 2] = 8`: The data's first (index 0) sub-pixel also maps to the Delaunay
        pixelization's ninth (index 8) pixel.

        The interpolation weights of these multiple mappings are stored in the array `pix_weights_for_sub_slim_index`.

        For the Delaunay pixelization these mappings are calculated using the Scipy spatial library
        (see `mapper_numba_util.pix_indexes_for_sub_slim_index_delaunay_from`).
        """
        delaunay = self.mesh_geometry.delaunay

        mappings = delaunay.mappings.astype("int")
        sizes = delaunay.sizes.astype("int")

        weights = pixel_weights_delaunay_from(
            source_plane_data_grid=self.source_plane_data_grid.over_sampled,
            source_plane_mesh_grid=self.source_plane_mesh_grid.array,
            pix_indexes_for_sub_slim_index=mappings,
            xp=self._xp,
        )

        return PixSubWeights(mappings=mappings, sizes=sizes, weights=weights)

    @property
    def pix_sub_weights_split_points(self) -> PixSubWeights:
        """
        The property `pix_sub_weights` property describes the calculation of the `PixSubWeights` object, which contains
        numpy arrays describing how data-points and mapper pixels map to one another and the weights of these mappings.

        For certain regularization schemes (e.g. `ConstantSplit`, `AdaptSplit`) regularization uses
        mappings which are split in a cross configuration in order to factor in the derivative of the mapper
        reconstruction.

        This property returns a unique set of `PixSubWeights` used for these regularization schemes which compute
        mappings and weights at each point on the split cross.
        """
        delaunay = self.mesh_geometry.delaunay

        splitted_weights = pixel_weights_delaunay_from(
            source_plane_data_grid=delaunay.split_points,
            source_plane_mesh_grid=self.source_plane_mesh_grid.array,
            pix_indexes_for_sub_slim_index=delaunay.splitted_mappings.astype("int"),
            xp=self._xp,
        )

        append_line_int = np.zeros((len(splitted_weights), 1), dtype="int") - 1
        append_line_float = np.zeros((len(splitted_weights), 1), dtype="float")

        return PixSubWeights(
            mappings=self._xp.hstack(
                (delaunay.splitted_mappings.astype(self._xp.int32), append_line_int)
            ),
            sizes=delaunay.splitted_sizes.astype(self._xp.int32),
            weights=self._xp.hstack((splitted_weights, append_line_float)),
        )
