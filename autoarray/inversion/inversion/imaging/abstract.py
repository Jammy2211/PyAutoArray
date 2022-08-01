import numpy as np
from typing import Dict, List, Optional

from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver


class AbstractInversionImaging(AbstractInversion):
    def __init__(
        self,
        data: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        preloads=None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.inversion.abstract.AbstractInversion` for a full description).

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

        from autoarray.preloads import Preloads

        preloads = preloads or Preloads()

        self.convolver = convolver

        super().__init__(
            data=data,
            noise_map=noise_map,
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

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
