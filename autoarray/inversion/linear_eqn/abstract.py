import numpy as np
from typing import Dict, List, Optional, Union

from autoarray.numba_util import profile_func

from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.inversion.linear_obj import LinearObj
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.structures.arrays.two_d.array_2d import Array2D

from autoarray.exc import InversionException


class AbstractLEq:
    def __init__(
        self,
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):

        self.noise_map = noise_map
        self.linear_obj_list = linear_obj_list

        self.profiling_dict = profiling_dict

    @property
    def mapper_list(self):

        mapper_list = [
            linear_obj if isinstance(linear_obj, AbstractMapper) else None
            for linear_obj in self.linear_obj_list
        ]

        return list(filter(None, mapper_list))

    @property
    def has_mapper(self):
        return len(self.mapper_list) > 0

    @property
    def has_one_mapper(self):
        return len(self.mapper_list) == 1

    @property
    def no_mapper_list(self):

        mapper_list = [
            linear_obj if not isinstance(linear_obj, AbstractMapper) else None
            for linear_obj in self.linear_obj_list
        ]

        return list(filter(None, mapper_list))

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

        return np.hstack([mapper.mapping_matrix for mapper in self.linear_obj_list])

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        raise NotImplementedError

    @profile_func
    def data_vector_from(self, data, preloads):
        raise NotImplementedError

    @property
    def curvature_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def source_quantity_dict_from(
        self, source_quantity: np.ndarray
    ) -> Dict[LinearObj, np.ndarray]:
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
        source_quantity_dict = {}

        index = 0

        for linear_obj in self.linear_obj_list:

            source_quantity_dict[linear_obj] = source_quantity[
                index : index + linear_obj.pixels
            ]

            index += linear_obj.pixels

        return source_quantity_dict

    @profile_func
    def mapped_reconstructed_data_dict_from(
        self, reconstruction
    ) -> Dict[LinearObj, Union[Array2D, Visibilities]]:
        raise NotImplementedError

    @profile_func
    def mapped_reconstructed_image_dict_from(
        self, reconstruction
    ) -> Dict[LinearObj, Union[Array2D, Visibilities]]:
        raise NotImplementedError

    @property
    def total_mappers(self):
        return len(self.mapper_list)
