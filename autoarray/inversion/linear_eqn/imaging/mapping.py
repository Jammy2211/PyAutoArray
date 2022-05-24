import numpy as np
from typing import Dict, List, Optional

from autoarray.numba_util import profile_func

from autoarray.inversion.linear_eqn.imaging.abstract import AbstractLEqImaging
from autoarray.inversion.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver

from autoarray.inversion.linear_eqn import leq_util


class LEqImagingMapping(AbstractLEqImaging):
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
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description.

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Imaging` objects, where the data is an image
        and the mappings may include a convolution operation described by the imaging data's PSF.

        This class uses the mapping formalism, which constructs the simultaneous linear equations using the
        `mapping_matrix` of every linear object.

        Parameters
        -----------
        noise_map
            The noise-map of the observed imaging data which values are solved for.
        convolver
            The convolver used to include 2D convolution of the mapping matrix with the imaigng data's PSF.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        super().__init__(
            noise_map=noise_map,
            convolver=convolver,
            linear_obj_list=linear_obj_list,
            settings=settings,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def data_vector_from(self, data: Array2D, preloads) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `data_vector` values are solved for simultaneously.

        The calculation is described in more detail in `leq_util.data_vector_via_blurred_mapping_matrix_from`.
        """

        if preloads.operated_mapping_matrix is not None:
            blurred_mapping_matrix = preloads.operated_mapping_matrix
        else:
            blurred_mapping_matrix = self.blurred_mapping_matrix

        return leq_util.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=data,
            noise_map=self.noise_map,
        )

    @property
    @profile_func
    def curvature_matrix(self):
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `curvature_matrix` values are solved for simultaneously. This includes all
        diagonal and off-diagonal terms describing the covariances between linear objects.
        """
        return leq_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.operated_mapping_matrix,
            noise_map=self.noise_map,
            add_to_curvature_diag=self.add_to_curvature_diag,
            linear_obj_func_index_list=self.linear_obj_func_index_list,
        )

    @profile_func
    def mapped_reconstructed_data_dict_from(
        self, reconstruction: np.ndarray
    ) -> Dict[LinearObj, Array2D]:
        """
        When constructing the simultaneous linear equations (via vectors and matrices) the quantities of each individual
        linear object (e.g. their `mapping_matrix`) are combined into single ndarrays via stacking. This does not track
        which quantities belong to which linear objects, therefore the linear equation's solutions (which are returned
        as ndarrays) do not contain information on which linear object(s) they correspond to.

        For example, consider if two `Mapper` objects with 50 and 100 source pixels are used in an `Inversion`.
        The `reconstruction` (which contains the solved for source pixels values) is an ndarray of shape [150], but
        the ndarray itself does not track which values belong to which `Mapper`.

        This function converts an ndarray of a `reconstruction` to a dictionary of ndarrays containing each linear
        object's reconstructed data values, where the keys are the instances of each mapper in the inversion.

        To perform this mapping the `mapping_matrix` is used, which straightforwardly describes how every value of
        the `reconstruction` maps to pixels in the data-frame after the 2D convolution operation has been performed.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """

        mapped_reconstructed_data_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=reconstruction
        )

        blurred_mapping_matrix_list = self.blurred_mapping_matrix_list

        for index, linear_obj in enumerate(self.linear_obj_list):

            reconstruction = reconstruction_dict[linear_obj]

            mapped_reconstructed_image = leq_util.mapped_reconstructed_data_via_mapping_matrix_from(
                mapping_matrix=blurred_mapping_matrix_list[index],
                reconstruction=reconstruction,
            )

            mapped_reconstructed_image = Array2D(
                array=mapped_reconstructed_image, mask=self.mask.mask_sub_1
            )

            mapped_reconstructed_data_dict[linear_obj] = mapped_reconstructed_image

        return mapped_reconstructed_data_dict
