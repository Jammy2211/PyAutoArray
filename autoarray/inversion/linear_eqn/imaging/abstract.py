import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.inversion.linear_obj.func_list import LinearObj
from autoarray.inversion.linear_eqn.abstract import AbstractLEq
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver


class AbstractLEqImaging(AbstractLEq):
    def __init__(
        self,
        noise_map: Array2D,
        convolver: Convolver,
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Imaging` objects, where the data is an image
        and the mappings may include a convolution operation described by the imaging data's PSF.

        Parameters
        -----------
        noise_map
            The noise-map of the observed imaging data which values are solved for.
        convolver
            The convolver which performs a 2D convolution on the mapping matrix with the imaging data's PSF.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.convolver = convolver

        super().__init__(
            noise_map=noise_map,
            linear_obj_list=linear_obj_list,
            settings=settings,
            profiling_dict=profiling_dict,
        )

    @property
    def mask(self) -> Array2D:
        return self.noise_map.mask

    @cached_property
    @profile_func
    def operated_mapping_matrix(self) -> np.ndarray:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        If there are multiple linear objects, the blurred mapping matrices are stacked such that their simultaneous
        linear equations are solved simultaneously.
        """
        return np.hstack(self.operated_mapping_matrix_list)

    @property
    def operated_mapping_matrix_list(self) -> List[np.ndarray]:
        """
        The `operated_mapping_matrix` of a linear object describes the mappings between the observed data's values and
        the linear objects model, including a 2D convolution operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        This property returns the a list of each linear object's blurred mapping matrix, which is computed by
        blurring each linear object's `mapping_matrix` property with the `Convolver` operator.

        A linear object may have a `operated_mapping_matrix_override` property, which bypasses  the `mapping_matrix`
        computation and convolution operator and is directly placed in the `operated_mapping_matrix_list`.
        """

        return [
            self.convolver.convolve_mapping_matrix(
                mapping_matrix=linear_obj.mapping_matrix
            )
            if linear_obj.operated_mapping_matrix_override is None
            else linear_obj.operated_mapping_matrix_override
            for linear_obj in self.linear_obj_list
        ]

    def mapped_reconstructed_image_dict_from(
        self, reconstruction: np.ndarray
    ) -> Dict[LinearObj, Array2D]:
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

        For the linear equations which fit imaging datasets, the reconstructed data is the reconstructed image, thus
        the function `mapped_reconstructed_data_dict_from` can be reused.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """
        return self.mapped_reconstructed_data_dict_from(reconstruction=reconstruction)
