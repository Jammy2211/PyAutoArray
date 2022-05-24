import numpy as np
from typing import Dict, List, Optional, Union

from autoarray.numba_util import profile_func

from autoarray.inversion.mappers.abstract import AbstractMapper
from autoarray.inversion.linear_obj import LinearObj
from autoarray.inversion.linear_obj import LinearObjFunc
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.structures.arrays.uniform_2d import Array2D


class AbstractLEq:
    def __init__(
        self,
        noise_map: Union[Array2D, VisibilitiesNoiseMap],
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved.

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. For example:

        - `Mapper` objects describe the mappings between the data's values and pixels another pixelizaiton
        (e.g. a rectangular grid, Voronoi mesh, etc.).

        - `LinearObjFunc` objects describe the mappings between the data's values and a functional form.

        From the `mapping_matrix` a system of linear equations can be constructed, which can then be solved for using
        the `Inversion` object. This class provides functions for setting up the system of linear equations.

        Parameters
        ----------
        noise_map
            The noise-map of the observed data which values are solved for.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """
        self.noise_map = noise_map
        self.linear_obj_list = linear_obj_list
        self.settings = settings
        self.profiling_dict = profiling_dict

    @property
    def linear_obj_func_list(self) -> List[LinearObjFunc]:
        """
        Returns a list of all linear objects based on analytic functions used to construct the simultaneous linear
        equations.

        This property removes other linear objects (E.g. `Mapper` objects).
        """
        linear_obj_func_list = [
            linear_obj if isinstance(linear_obj, LinearObjFunc) else None
            for linear_obj in self.linear_obj_list
        ]

        return list(filter(None, linear_obj_func_list))

    @property
    def linear_obj_func_index_list(self) -> List[int]:

        linear_obj_func_index_list = []

        pixel_count = 0

        for linear_obj in self.linear_obj_list:

            if isinstance(linear_obj, LinearObjFunc):

                linear_obj_func_index_list.append(pixel_count)

            pixel_count += linear_obj.pixels

        return linear_obj_func_index_list

    @property
    def has_linear_obj_func(self):
        return len(self.linear_obj_func_list) > 0

    @property
    def mapper_list(self) -> List[AbstractMapper]:
        """
        Returns a list of all mappers used to construct the simultaneous linear equations.

        This property removes other linear objects (E.g. `LinearObjFunc` objects).
        """
        mapper_list = [
            linear_obj if isinstance(linear_obj, AbstractMapper) else None
            for linear_obj in self.linear_obj_list
        ]

        return list(filter(None, mapper_list))

    @property
    def has_mapper(self) -> bool:
        return len(self.mapper_list) > 0

    @property
    def has_one_mapper(self) -> bool:
        return len(self.mapper_list) == 1

    @property
    def no_mapper_list(self) -> List[LinearObj]:
        """
        Returns a list of all linear objects that are not mappers which used to construct the simultaneous linear
        equations.

        This property retains linear objects which are not mappers (E.g. `LinearObjFunc` objects).
        """
        mapper_list = [
            linear_obj if not isinstance(linear_obj, AbstractMapper) else None
            for linear_obj in self.linear_obj_list
        ]

        return list(filter(None, mapper_list))

    @property
    def add_to_curvature_diag(self) -> bool:

        # TODO : Use this criteria once we do full inversion refactor for linear objs.

        # if len(self.linear_obj_func_list) == len(self.linear_obj_list):

        if self.has_linear_obj_func:
            return self.settings.linear_funcs_add_to_curvature_diag
        return False

    @property
    @profile_func
    def mapping_matrix(self) -> np.ndarray:
        """
        The `mapping_matrix` of a linear object describes the mappings between the observed data's values and the
        linear objects model. These are used to construct the simultaneous linear equations which reconstruct the data.

        If there are multiple linear objects, the mapping matrices are stacked such that their simultaneous linear
        equations are solved simultaneously. This property returns the stacked mapping matrix.
        """
        return np.hstack(
            [linear_obj.mapping_matrix for linear_obj in self.linear_obj_list]
        )

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        """
        The linear objects whose mapping matrices are used to construct the simultaneous linear equations can have
        operations applied to them which include this operation in the solution.

        This property returns the final operated-on mapping matrix of every linear object. These are stacked such that
        their simultaneous linear equations are solved simultaneously
        """
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
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays via stacking. This does not track
        which quantities belong to which linear objects, therefore the linear equation's solutions (which are returned
        as ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `source_quantity` (like a `reconstruction`) to a dictionary of ndarrays,
        where the keys are the instances of each mapper in the inversion.

        Parameters
        ----------
        source_quantity
            The quantity whose values are mapped to a dictionary of values for each individual mapper.

        Returns
        -------
        The dictionary of ndarrays of values for each individual mapper.
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
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays. This does not track which
        quantities belong to which linear objects, therefore the linear equation's solutions (which are returned as
        ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed data values, where the keys are the instances of each mapper in the inversion.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """
        raise NotImplementedError

    @profile_func
    def mapped_reconstructed_image_dict_from(
        self, reconstruction
    ) -> Dict[LinearObj, Union[Array2D, Visibilities]]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays. This does not track which
        quantities belong to which linear objects, therefore the linear equation's solutions (which are returned as
        ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed images, where the keys are the instances of each mapper in the inversion.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """
        raise NotImplementedError

    @property
    def total_mappers(self) -> int:
        return len(self.mapper_list)
