import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

from autoarray.inversion.linear_eqn.abstract import AbstractLEq
from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.linear_obj.func_list import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.linear_eqn import leq_util

from autoarray.numba_util import profile_func


class AbstractLEqInterferometer(AbstractLEq):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
        an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
        the interferometer dataset's transformer.

        Parameters
        -----------
        noise_map
            The noise-map of the observed interferometer data which values are solved for.
        transformer
            The transformer which performs a non-uniform fast Fourier transform operations on the mapping matrix
            with the interferometer data's transformer.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        super().__init__(
            noise_map=noise_map,
            linear_obj_list=linear_obj_list,
            settings=settings,
            profiling_dict=profiling_dict,
        )

        self.transformer = transformer

    @property
    def mask(self) -> Mask2D:
        return self.transformer.real_space_mask

    @cached_property
    @profile_func
    def transformed_mapping_matrix(self) -> np.ndarray:
        """
        The `transformed_mapping_matrix` of a linear object describes the mappings between the observed data's values
        and the linear objects model, including a non-uniform fast Fourier transform operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        If there are multiple linear objects, the transformed mapping matrices are stacked such that their simultaneous
        linear equations are solved simultaneously.
        """
        return np.hstack(self.transformed_mapping_matrix_list)

    @property
    def transformed_mapping_matrix_list(self) -> List[np.ndarray]:
        """
        The `transformed_mapping_matrix` of a linear object describes the mappings between the observed data's values
        and the linear objects model, including a non-uniform fast Fourier transform operation.

        This is used to construct the simultaneous linear equations which reconstruct the data.

        This property returns the a list of each linear object's transformed mapping matrix.
        """
        return [
            self.transformer.transform_mapping_matrix(
                mapping_matrix=linear_obj.mapping_matrix
            )
            for linear_obj in self.linear_obj_list
        ]

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        """
        The linear objects whose mapping matrices are used to construct the simultaneous linear equations can have
        operations applied to them which include this operation in the solution.

        This property returns the final operated-on mapping matrix of every linear object. These are stacked such that
        their simultaneous linear equations are solved simultaneously

        For the linear equations which solve interferometer data only a non-uniform fast Fourier transform operation
        is performed.
        """
        return self.transformed_mapping_matrix

    @profile_func
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

        For the linear equations which fit interferometer datasets, the reconstructed data is its visibilities. Thus,
        the reconstructed image is computed separately by performing a non-uniform fast Fourier transform which maps
        the `reconstruction`'s values to real space.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """
        mapped_reconstructed_image_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=reconstruction
        )

        for linear_obj in self.linear_obj_list:

            reconstruction = reconstruction_dict[linear_obj]

            mapped_reconstructed_image = leq_util.mapped_reconstructed_data_via_mapping_matrix_from(
                mapping_matrix=linear_obj.mapping_matrix, reconstruction=reconstruction
            )

            mapped_reconstructed_image = Array2D(
                array=mapped_reconstructed_image, mask=self.mask.mask_sub_1
            )

            mapped_reconstructed_image_dict[linear_obj] = mapped_reconstructed_image

        return mapped_reconstructed_image_dict
