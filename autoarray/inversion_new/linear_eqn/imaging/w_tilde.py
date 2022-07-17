import numpy as np
from scipy.linalg import block_diag
from typing import Dict, List, Optional

from autoarray.numba_util import profile_func

from autoarray.inversion_new.linear_eqn.imaging.abstract import AbstractLEqImaging
from autoarray.inversion_new.linear_obj.func_list import LinearObj
from autoarray.inversion_new.self.unique_mappings import UniqueMappings
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.dataset.imaging import WTildeImaging

from autoarray.inversion.linear_eqn import leq_util


class LEqImagingWTilde(AbstractLEqImaging):
    def __init__(
        self,
        unique_mappings: UniqueMappings,
        data: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        w_tilde: WTildeImaging,
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq`) for a full description.

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Imaging` objects, where the data is an image
        and the mappings may include a convolution operation described by the imaging data's PSF.

        This class uses the w-tilde formalism, which speeds up the construction of the simultaneous linear equations by
        bypassing the construction of a `mapping_matrix`.

        Parameters
        -----------
        noise_map
            The noise-map of the observed imaging data which values are solved for.
        convolver
            The convolver used to include 2D convolution of the mapping matrix with the imaigng data's PSF.
        w_tilde
            An object containing matrices that construct the linear equations via the w-tilde formalism which bypasses
            the mapping matrix.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.unique_mappings = unique_mappings

        super().__init__(
            data=data,
            noise_map=noise_map,
            convolver=convolver,
            settings=settings,
            profiling_dict=profiling_dict,
        )

        if self.settings.use_w_tilde:
            self.w_tilde = w_tilde
            self.w_tilde.check_noise_map(noise_map=noise_map)
        else:
            self.w_tilde = None

    @property
    @profile_func
    def data_vector(self) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
        for simultaneously.

        The calculation is described in more detail in `leq_util.w_tilde_data_imaging_from`.
        """

        w_tilde_data = leq_util.w_tilde_data_imaging_from(
            image_native=self.data.native,
            noise_map_native=self.noise_map.native,
            kernel_native=self.convolver.kernel.native,
            native_index_for_slim_index=self.data.mask.native_index_for_slim_index,
        )

        return leq_util.data_vector_via_w_tilde_data_imaging_from(
            w_tilde_data=w_tilde_data,
            data_to_pix=self.unique_mappings.data_to_pix,
            data_weights=self.unique_mappings.data_weights,
            pix_lengths=self.unique_mappings.pix_lengths,
            pix_pixels=self.unique_mappings.pixels,
        )

    @property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes F using the w_tilde formalism, which is faster as it precomputes the PSF convolution
        of different noise-map pixels (see `curvature_matrix_via_w_tilde_curvature_preload_imaging_from`).

        If there are multiple linear objects the curvature_matrices are combined to ensure their values are solved
        for simultaneously. In the w-tilde formalism this requires us to consider the mappings between data and every
        linear object, meaning that the linear alegbra has both on and off diagonal terms.

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        curvature_matrix = leq_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
            curvature_preload=self.w_tilde.curvature_preload,
            curvature_indexes=self.w_tilde.indexes,
            curvature_lengths=self.w_tilde.lengths,
            data_to_pix=self.unique_mappings.data_to_pix,
            data_weights=self.unique_mappings.data_weights,
            pix_lengths=self.unique_mappings.pix_lengths,
            pix_pixels=self.pixels,
        )

        for i in range(curvature_matrix.shape[0]):
            for j in range(curvature_matrix.shape[1]):
                curvature_matrix[j, i] = curvature_matrix[i, j]

        return curvature_matrix

    @profile_func
    def curvature_matrix_off_diag_from(
        self, unique_mappings_other: UniqueMappings
    ) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes the off-diagonal terms of F using the w_tilde formalism.
        """

        curvature_matrix_off_diag_0 = leq_util.curvature_matrix_off_diags_via_w_tilde_curvature_preload_imaging_from(
            curvature_preload=self.w_tilde.curvature_preload,
            curvature_indexes=self.w_tilde.indexes,
            curvature_lengths=self.w_tilde.lengths,
            data_to_pix_unique_0=self.unique_mappings.data_to_pix,
            data_weights_0=self.unique_mappings.data_weights,
            pix_lengths_0=self.unique_mappings.pix_lengths,
            pix_pixels_0=self.unique_mappings.pixels,
            data_to_pix_unique_1=unique_mappings_other.data_to_pix,
            data_weights_1=unique_mappings_other.data_weights,
            pix_lengths_1=unique_mappings_other.pix_lengths,
            pix_pixels_1=unique_mappings_other.pixels,
        )

        curvature_matrix_off_diag_1 = leq_util.curvature_matrix_off_diags_via_w_tilde_curvature_preload_imaging_from(
            curvature_preload=self.w_tilde.curvature_preload,
            curvature_indexes=self.w_tilde.indexes,
            curvature_lengths=self.w_tilde.lengths,
            data_to_pix_unique_0=unique_mappings_other.data_to_pix,
            data_weights_0=unique_mappings_other.data_weights,
            pix_lengths_0=unique_mappings_other.pix_lengths,
            pix_pixels_0=unique_mappings_other.pixels,
            data_to_pix_unique_1=self.unique_mappings.data_to_pix,
            data_weights_1=self.unique_mappings.data_weights,
            pix_lengths_1=self.unique_mappings.pix_lengths,
            pix_pixels_1=self.unique_mappings.pixels,
        )

        return curvature_matrix_off_diag_0 + curvature_matrix_off_diag_1.T

    @profile_func
    def mapped_reconstructed_data_from(self, reconstruction: np.ndarray) -> Array2D:
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

        The w-tilde formalism bypasses the calculation of the `mapping_matrix` and it therefore cannot be used to map
        the reconstruction's values to the data frame. Instead, the unique data-to-pixelization mappings are used,
        including the 2D convolution operation after mapping is complete.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """

        mapped_reconstructed_image = leq_util.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix=self.unique_mappings.data_to_pix,
            data_weights=self.unique_mappings.data_weights,
            pix_lengths=self.unique_mappings.pix_lengths,
            reconstruction=reconstruction,
        )

        mapped_reconstructed_image = Array2D(
            array=mapped_reconstructed_image, mask=self.mask.mask_sub_1
        )

        return self.convolver.convolve_image_no_blurring(
            image=mapped_reconstructed_image
        )
