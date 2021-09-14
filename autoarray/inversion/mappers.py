import itertools
import numpy as np
from typing import Dict, Optional

from autoconf import cached_property

from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DRectangular
from autoarray.structures.grids.two_d.grid_2d_pixelization import Grid2DVoronoi

from autoarray.numba_util import profile_func
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.structures.grids.two_d import grid_2d_util
from autoarray.inversion import mapper_util


class UniqueMappings:
    def __init__(self, data_to_pix_unique, data_weights, pix_lengths):

        self.data_to_pix_unique = data_to_pix_unique.astype("int")
        self.data_weights = data_weights
        self.pix_lengths = pix_lengths.astype("int")


def mapper(
    source_grid_slim,
    source_pixelization_grid,
    data_pixelization_grid=None,
    hyper_data=None,
):

    if isinstance(source_pixelization_grid, Grid2DRectangular):
        return MapperRectangular(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_data,
        )
    elif isinstance(source_pixelization_grid, Grid2DVoronoi):
        return MapperVoronoi(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_data,
        )


class Mapper:
    def __init__(
        self,
        source_grid_slim,
        source_pixelization_grid,
        data_pixelization_grid=None,
        hyper_image=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Abstract base class representing a mapper, which maps unmasked pixels on a masked 2D array in the form of
        a grid, see the *hyper_galaxies.array.grid* module to discretized pixels in a pixelization.

        1D structures are used to represent these mappings, for example between the different grid in a grid
        (e.g. the / sub grid). This follows the syntax grid_to_grid, whereby the index of a value on one grid
        equals that of another grid, for example:

        - data_to_pix[2] = 1  tells us that the 3rd pixel on a grid maps to the 2nd pixel of a pixelization.
        - sub_to_pix4] = 2  tells us that the 5th sub-pixel of a sub-grid maps to the 3rd pixel of a pixelization.
        - pix_to_data[2] = 5 tells us that the 3rd pixel of a pixelization maps to the 6th (unmasked) pixel of a
                            grid.

        Mapping Matrix:

        The mapper allows us to create a mapping matrix, which is a matrix representing the mapping between every
        unmasked pixel of a grid and the pixels of a pixelization. Non-zero entries signify a mapping, whereas zeros
        signify no mapping.

        For example, if the grid has 5 pixels and the pixelization 3 pixels, with the following mappings:

        pixel 0 -> pixelization pixel 0
        pixel 1 -> pixelization pixel 0
        pixel 2 -> pixelization pixel 1
        pixel 3 -> pixelization pixel 1
        pixel 4 -> pixelization pixel 2

        The mapping matrix (which is of dimensions regular_pixels x pixelization_pixels) would appear as follows:

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        The mapping matrix is in fact built using the sub-grid of the grid, whereby each pixel is
        divided into a grid of sub-pixels which are all paired to pixels in the pixelization. The entries
        in the mapping matrix now become fractional values dependent on the sub-grid size. For example, for a 2x2
        sub-grid in each pixel which means the fraction value is 1.0/(2.0^2) = 0.25, if we have the following mappings:

        pixel 0 -> sub pixel 0 -> pixelization pixel 0
        pixel 0 -> sub pixel 1 -> pixelization pixel 1
        pixel 0 -> sub pixel 2 -> pixelization pixel 1
        pixel 0 -> sub pixel 3 -> pixelization pixel 1
        pixel 1 -> sub pixel 0 -> pixelization pixel 1
        pixel 1 -> sub pixel 1 -> pixelization pixel 1
        pixel 1 -> sub pixel 2 -> pixelization pixel 1
        pixel 1 -> sub pixel 3 -> pixelization pixel 1
        pixel 2 -> sub pixel 0 -> pixelization pixel 2
        pixel 2 -> sub pixel 1 -> pixelization pixel 2
        pixel 2 -> sub pixel 2 -> pixelization pixel 3
        pixel 2 -> sub pixel 3 -> pixelization pixel 3

        The mapping matrix (which is still of dimensions regular_pixels x source_pixels) would appear as follows:

        [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
        [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
        [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]

        Parameters
        ----------
        pixels
            The number of pixels in the mapper's pixelization.
        source_grid_slim: gridStack
            A stack of grid's which are mapped to the pixelization (includes an and sub grid).
        hyper_image
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """

        self.source_grid_slim = source_grid_slim
        self.source_pixelization_grid = source_pixelization_grid
        self.data_pixelization_grid = data_pixelization_grid

        self.hyper_image = hyper_image
        self.profiling_dict = profiling_dict

    @property
    def pixels(self):
        return self.source_pixelization_grid.pixels

    @property
    def slim_index_for_sub_slim_index(self):
        return self.source_grid_slim.mask.slim_index_for_sub_slim_index

    @property
    def pixelization_index_for_sub_slim_index(self):
        raise NotImplementedError(
            "pixelization_index_for_sub_slim_index should be overridden"
        )

    @property
    def all_sub_slim_indexes_for_pixelization_index(self):
        """
        Returns the mappings between a pixelization's pixels and the unmasked sub-grid pixels. These mappings
        are determined after the grid is used to determine the pixelization.

        The pixelization's pixels map to different number of sub-grid pixels, thus a list of lists is used to
        represent these mappings.
        """
        all_sub_slim_indexes_for_pixelization_index = [[] for _ in range(self.pixels)]

        for slim_index, pix_index in enumerate(
            self.pixelization_index_for_sub_slim_index
        ):
            all_sub_slim_indexes_for_pixelization_index[pix_index].append(slim_index)

        return all_sub_slim_indexes_for_pixelization_index

    @cached_property
    @profile_func
    def data_unique_mappings(self):
        """
        The w_tilde formalism requires us to compute an array that gives the unique mappings between the sub-pixels of
        every image pixel to their corresponding pixelization pixels.
        """

        data_to_pix_unique, data_weights, pix_lengths = mapper_util.data_slim_to_pixelization_unique_from(
            data_pixels=self.source_grid_slim.shape_slim,
            pixelization_index_for_sub_slim_index=self.pixelization_index_for_sub_slim_index,
            sub_size=self.source_grid_slim.sub_size,
        )

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )

    @cached_property
    @profile_func
    def mapping_matrix(self):
        """
        The `mapping_matrix` is a matrix that represents the image-pixel to pixelization-pixel mappings above in a
        2D matrix. It in the following paper as matrix `f` https://arxiv.org/pdf/astro-ph/0302587.pdf.

        A full description is given in `mapping_matrix_from`.
        """
        return mapper_util.mapping_matrix_from(
            pixelization_index_for_sub_slim_index=self.pixelization_index_for_sub_slim_index,
            pixels=self.pixels,
            total_mask_pixels=self.source_grid_slim.mask.pixels_in_mask,
            slim_index_for_sub_slim_index=self.slim_index_for_sub_slim_index,
            sub_fraction=self.source_grid_slim.mask.sub_fraction,
        )

    def pixel_signals_from(self, signal_scale):

        return mapper_util.adaptive_pixel_signals_from(
            pixels=self.pixels,
            signal_scale=signal_scale,
            pixelization_index_for_sub_slim_index=self.pixelization_index_for_sub_slim_index,
            slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
            hyper_image=self.hyper_image,
        )

    def slim_indexes_from_pixelization_indexes(self, pixelization_indexes):

        image_for_source = self.all_sub_slim_indexes_for_pixelization_index

        if not any(isinstance(i, list) for i in pixelization_indexes):
            return list(
                itertools.chain.from_iterable(
                    [image_for_source[index] for index in pixelization_indexes]
                )
            )
        else:
            indexes = []
            for source_pixel_index_list in pixelization_indexes:
                indexes.append(
                    list(
                        itertools.chain.from_iterable(
                            [
                                image_for_source[index]
                                for index in source_pixel_index_list
                            ]
                        )
                    )
                )
            return indexes

    def reconstruction_from(self, solution_vector):
        """
        Given the solution vector of an inversion (see *inversions.Inversion*), determine the reconstructed
        pixelization of the rectangular pixelization by using the mapper.
        """
        raise NotImplementedError()


class MapperRectangular(Mapper):
    def __init__(
        self,
        source_grid_slim,
        source_pixelization_grid,
        data_pixelization_grid=None,
        hyper_image=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Class representing a rectangular mapper, which maps unmasked pixels on a masked 2D array in the form of
        a grid, see the *hyper_galaxies.array.grid* module to pixels discretized on a rectangular grid.

        The and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels
            The number of pixels in the rectangular pixelization (y_pixels*x_pixels).
        source_grid_slim : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        shape_native
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the rectangular pixelization.
        """
        super().__init__(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @property
    def shape_native(self):
        return self.source_pixelization_grid.shape_native

    @cached_property
    @profile_func
    def pixelization_index_for_sub_slim_index(self):
        """
        The `Mapper` contains:

         1) The traced grid of (y,x) source pixel coordinate centres.
         2) The traced grid of (y,x) image pixel coordinates.

        The function below pairs every image-pixel coordinate to every source-pixel centre.

        In the API, the `pixelization_index` refers to the source pixel index (e.g. source pixel 0, 1, 2 etc.) whereas
        the sub_slim index refers to the index of a sub-gridded image pixel (e.g. sub pixel 0, 1, 2 etc.).
        """
        return grid_2d_util.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=self.source_grid_slim,
            shape_native=self.source_pixelization_grid.shape_native,
            pixel_scales=self.source_pixelization_grid.pixel_scales,
            origin=self.source_pixelization_grid.origin,
        ).astype("int")

    def reconstruction_from(self, solution_vector):
        """
        Given the solution vector of an inversion (see *inversions.Inversion*), determine the reconstructed
        pixelization of the rectangular pixelization by using the mapper.
        """
        recon = array_2d_util.array_2d_native_from(
            array_2d_slim=solution_vector,
            mask_2d=np.full(
                fill_value=False, shape=self.source_pixelization_grid.shape_native
            ),
            sub_size=1,
        )
        return Array2D.manual(
            array=recon,
            sub_size=1,
            pixel_scales=self.source_pixelization_grid.pixel_scales,
            origin=self.source_pixelization_grid.origin,
        )


class MapperVoronoi(Mapper):
    def __init__(
        self,
        source_grid_slim,
        source_pixelization_grid,
        data_pixelization_grid=None,
        hyper_image=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Class representing a Voronoi mapper, which maps unmasked pixels on a masked 2D array in the form of
        a grid, see the *hyper_galaxies.array.grid* module to pixels discretized on a Voronoi grid.

        The irand non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge
        of how different grid map to one another.

        Parameters
        ----------
        pixels
            The number of pixels in the Voronoi pixelization.
        source_grid_slim : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        voronoi : scipy.spatial.Voronoi
            Class storing the Voronoi grid's 
        geometry : pixelization.Voronoi.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the Vornoi pixelization.
        hyper_image
            A pre-computed hyper-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        super().__init__(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_image,
            profiling_dict=profiling_dict,
        )

    @cached_property
    @profile_func
    def pixelization_index_for_sub_slim_index(self):
        """
        The 1D index mappings between the sub pixels and Voronoi pixelization pixels.
        """

        return mapper_util.pixelization_index_for_voronoi_sub_slim_index_from(
            grid=self.source_grid_slim,
            nearest_pixelization_index_for_slim_index=self.source_pixelization_grid.nearest_pixelization_index_for_slim_index,
            slim_index_for_sub_slim_index=self.source_grid_slim.mask.slim_index_for_sub_slim_index,
            pixelization_grid=self.source_pixelization_grid,
            pixel_neighbors=self.source_pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=self.source_pixelization_grid.pixel_neighbors.sizes,
        ).astype("int")

    @property
    def voronoi(self):
        return self.source_pixelization_grid.voronoi

    def reconstruction_from(self, solution_vector):
        return solution_vector
