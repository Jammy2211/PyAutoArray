import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

from autoarray.inversion.inversion.interferometer.abstract import (
    AbstractInversionInterferometer,
)
from autoarray.dataset.interferometer.w_tilde import WTildeInterferometer
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.inversion.pixelization.mappers.abstract import AbstractMapper
from autoarray.preloads import Preloads
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.inversion import inversion_util

from autoarray.numba_util import profile_func


class InversionInterferometerWTilde(AbstractInversionInterferometer):
    def __init__(
        self,
        data: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        w_tilde: WTildeInterferometer,
        linear_obj_list: List[LinearObj],
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.inversion.abstract.AbstractInversion` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
        an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
        the interferometer dataset's transformer.

        This class uses the w-tilde formalism, which speeds up the construction of the simultaneous linear equations by
        bypassing the construction of a `mapping_matrix`.

        Parameters
        ----------
        noise_map
            The noise-map of the observed interferometer data which values are solved for.
        transformer
            The transformer which performs a non-uniform fast Fourier transform operations on the mapping matrix
            with the interferometer data's transformer.
        w_tilde
            An object containing matrices that construct the linear equations via the w-tilde formalism which bypasses
            the mapping matrix.
        linear_obj_list
            The linear objects used to reconstruct the data's observed values. If multiple linear objects are passed
            the simultaneous linear equations are combined and solved simultaneously.
        profiling_dict
            A dictionary which contains timing of certain functions calls which is used for profiling.
        """

        self.w_tilde = w_tilde
        self.w_tilde.check_noise_map(noise_map=noise_map)

        super().__init__(
            data=data,
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            settings=settings,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

        self.settings = settings

    @cached_property
    @profile_func
    def data_vector(self) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
        for simultaneously.

        The calculation is described in more detail in `inversion_util.w_tilde_data_interferometer_from`.
        """
        return np.dot(
            self.linear_obj_list[0].mapping_matrix.T, self.w_tilde.dirty_image
        )

    @cached_property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `curvature_matrix` values are solved for simultaneously. This includes all
        diagonal and off-diagonal terms describing the covariances between linear objects.
        """
        return self.curvature_matrix_diag

    @property
    @profile_func
    def curvature_matrix_diag(self) -> np.ndarray:
        """
        The `curvature_matrix` is a 2D matrix which uses the mappings between the data and the linear objects to
        construct the simultaneous linear equations.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf, where the
        curvature matrix given by equation (4) and the letter F.

        This function computes the diagonal terms of F using the w_tilde formalism.
        """

        if self.settings.use_w_tilde_numpy:
            return inversion_util.curvature_matrix_via_w_tilde_from(
                w_tilde=self.w_tilde.w_matrix, mapping_matrix=self.mapping_matrix
            )

        from autoarray.inversion.inversion import inversion_util_secret

        mapper = self.cls_list_from(cls=AbstractMapper)[0]

        if not self.settings.use_source_loop:
            return inversion_util_secret.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
                curvature_preload=self.w_tilde.curvature_preload,
                pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
                pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
                pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
                native_index_for_slim_index=self.transformer.real_space_mask.derive_indexes.native_for_slim,
                pix_pixels=self.linear_obj_list[0].params,
            )

        (
            sub_slim_indexes_for_pix_index,
            sub_slim_sizes_for_pix_index,
            sub_slim_weights_for_pix_index,
        ) = mapper.sub_slim_indexes_for_pix_index_arr

        return inversion_util_secret.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from_2(
            curvature_preload=self.w_tilde.curvature_preload,
            native_index_for_slim_index=self.transformer.real_space_mask.derive_indexes.native_for_slim,
            pix_pixels=self.linear_obj_list[0].params,
            sub_slim_indexes_for_pix_index=sub_slim_indexes_for_pix_index.astype("int"),
            sub_slim_sizes_for_pix_index=sub_slim_sizes_for_pix_index.astype("int"),
            sub_slim_weights_for_pix_index=sub_slim_weights_for_pix_index,
        )

    @property
    @profile_func
    def mapped_reconstructed_data_dict(
        self,
    ) -> Dict[LinearObj, Visibilities]:
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

        To perform this mapping the `mapping_matrix` is used, which straightforwardly describes how every value of
        the `reconstruction` maps to pixels in the data-frame after the 2D non-uniform fast Fourier transformer
        operation has been performed.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the image-plane).
        """
        mapped_reconstructed_data_dict = {}

        image_dict = self.mapped_reconstructed_image_dict

        for linear_obj in self.linear_obj_list:
            visibilities = self.transformer.visibilities_from(
                image=image_dict[linear_obj]
            )

            visibilities = Visibilities(visibilities=visibilities)

            mapped_reconstructed_data_dict[linear_obj] = visibilities

        return mapped_reconstructed_data_dict
