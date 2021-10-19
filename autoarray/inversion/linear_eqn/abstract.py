import numpy as np
from typing import Dict, List, Optional, Union

from autoconf import cached_property

from autoarray.numba_util import profile_func

from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi


class AbstractLinearEqn:
    def __init__(
        self,
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        mapper_list: List[Union[MapperRectangular, MapperVoronoi]],
        profiling_dict: Optional[Dict] = None,
    ):

        self.noise_map = noise_map
        self.mapper_list = mapper_list

        self.profiling_dict = profiling_dict

    @property
    def has_one_mapper(self):
        if len(self.mapper_list) == 1:
            return True
        return False

    @property
    @profile_func
    def mapping_matrix(self) -> np.ndarray:
        """
        For a given pixelization pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the
        image  plane. This therefore creates a 'image' of the source pixel (which corresponds to a set of values that
        mostly zeros, but with 1's where mappings occur).

        Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function
        of our  dataset via 2D convolution. This uses the methods
        in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:
        """

        return np.hstack([mapper.mapping_matrix for mapper in self.mapper_list])

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @profile_func
    def data_vector_from(self, data, preloads):
        raise NotImplementedError

    @property
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def source_quantity_of_mappers_from(
        self, source_quantity: np.ndarray
    ) -> List[np.ndarray]:
        """
        Certain results in an `Inversion` are stored as a ndarray which contains the values of that quantity for
        every mapper. For example, the `reconstruction` of an inversion is a ndarray which has the source flux
        values of every mapper within the inversion.

        This function converts such an ndarray of `source_quantity` to a list of ndarrays, where each list index
        corresponds to each mapper in the inversion.

        Parameters
        ----------
        source_quantity
            The quantity whose values are mapped to a list of values for each individual mapper.

        Returns
        -------
        The list of ndarrays of values for each individual mapper.

        """
        source_quantity_of_mappers = []

        index = 0

        for mapper in self.mapper_list:
            source_quantity_of_mappers.append(
                source_quantity[index : index + mapper.pixels]
            )

            index += mapper.pixels

        return source_quantity_of_mappers

    @profile_func
    def mapped_reconstructed_data_of_mappers_from(
        self, reconstruction
    ) -> List[Array2D]:
        raise NotImplementedError

    @profile_func
    def mapped_reconstructed_image_of_mappers_from(self, reconstruction) -> Array2D:
        raise NotImplementedError

    @property
    def total_mappers(self):
        return len(self.mapper_list)
