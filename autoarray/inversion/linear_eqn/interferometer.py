import numpy as np
from typing import Dict, List, Optional, Union

from autoconf import cached_property

from autoarray.inversion.linear_eqn.abstract import AbstractLEq
from autoarray.dataset.interferometer import WTildeInterferometer
from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.linear_eqn import leq_util

from autoarray.numba_util import profile_func


class AbstractLEqInterferometer(AbstractLEq):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
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


class LEqInterferometerMapping(AbstractLEqInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
        an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
        the interferometer dataset's transformer.

        This class uses the mapping formalism, which constructs the simultaneous linear equations using the
        `mapping_matrix` of every linear object.

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
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def data_vector_from(self, data: Array2D, preloads: Preloads) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects their `operated_mapping_matrix` properties will have already been
        concatenated ensuring their `data_vector` values are solved for simultaneously.

        The calculation is described in more detail in `leq_util.data_vector_via_transformed_mapping_matrix_from`.
        """
        return leq_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            visibilities=data,
            noise_map=self.noise_map,
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

        real_curvature_matrix = leq_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.transformed_mapping_matrix.real,
            noise_map=self.noise_map.real,
        )

        imag_curvature_matrix = leq_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.transformed_mapping_matrix.imag,
            noise_map=self.noise_map.imag,
        )

        return np.add(real_curvature_matrix, imag_curvature_matrix)

    @profile_func
    def mapped_reconstructed_data_dict_from(
        self, reconstruction: np.ndarray
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
            individual mapper (in the data frame).
        """
        mapped_reconstructed_data_dict = {}

        reconstruction_dict = self.source_quantity_dict_from(
            source_quantity=reconstruction
        )

        transformed_mapping_matrix_list = self.transformed_mapping_matrix_list

        for index, linear_obj in enumerate(self.linear_obj_list):

            reconstruction = reconstruction_dict[linear_obj]

            visibilities = leq_util.mapped_reconstructed_visibilities_from(
                transformed_mapping_matrix=transformed_mapping_matrix_list[index],
                reconstruction=reconstruction,
            )

            visibilities = Visibilities(visibilities=visibilities)

            mapped_reconstructed_data_dict[linear_obj] = visibilities

        return mapped_reconstructed_data_dict


class LEqInterferometerWTilde(AbstractLEqInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        w_tilde: WTildeInterferometer,
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

        This class uses the w-tilde formalism, which speeds up the construction of the simultaneous linear equations by
        bypassing the construction of a `mapping_matrix`.

        Parameters
        -----------
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
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

        self.settings = settings

    @profile_func
    def data_vector_from(self, data: Array2D, preloads: Preloads) -> np.ndarray:
        """
        The `data_vector` is a 1D vector whose values are solved for by the simultaneous linear equations constructed
        by this object.

        The linear algebra is described in the paper https://arxiv.org/pdf/astro-ph/0302587.pdf), where the
        data vector is given by equation (4) and the letter D.

        If there are multiple linear objects the `data_vectors` are concatenated ensuring their values are solved
        for simultaneously.

        The calculation is described in more detail in `leq_util.w_tilde_data_interferometer_from`.
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

            return leq_util.curvature_matrix_via_w_tilde_from(
                w_tilde=self.w_tilde.w_matrix, mapping_matrix=self.mapping_matrix
            )

        from autoarray.inversion.inversion import inversion_util_secret

        return inversion_util_secret.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
            curvature_preload=self.w_tilde.curvature_preload,
            pix_indexes_for_sub_slim_index=self.mapper_list[
                0
            ].pix_indexes_for_sub_slim_index,
            pix_size_for_sub_slim_index=self.mapper_list[
                0
            ].pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=self.mapper_list[
                0
            ].pix_weights_for_sub_slim_index,
            native_index_for_slim_index=self.transformer.real_space_mask.native_index_for_slim_index,
            pixelization_pixels=self.linear_obj_list[0].pixels,
        )

    @profile_func
    def mapped_reconstructed_data_dict_from(
        self, reconstruction: np.ndarray
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
            individual mapper (in the data frame).
        """
        mapped_reconstructed_data_dict = {}

        image_dict = self.mapped_reconstructed_image_dict_from(
            reconstruction=reconstruction
        )

        for linear_obj in self.linear_obj_list:

            visibilities = self.transformer.visibilities_from(
                image=image_dict[linear_obj]
            )

            visibilities = Visibilities(visibilities=visibilities)

            mapped_reconstructed_data_dict[linear_obj] = visibilities

        return mapped_reconstructed_data_dict


class LEqInterferometerMappingPyLops(AbstractLEqInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Constructs linear equations (via vectors and matrices) which allow for sets of simultaneous linear equations
        to be solved (see `inversion.linear_eqn.abstract.AbstractLEq` for a full description).

        A linear object describes the mappings between values in observed `data` and the linear object's model via its
        `mapping_matrix`. This class constructs linear equations for `Interferometer` objects, where the data is an
        an array of visibilities and the mappings include a non-uniform fast Fourier transform operation described by
        the interferometer dataset's transformer.

        This class uses the mapping formalism, which constructs the simultaneous linear equations using the
        `mapping_matrix` of every linear object. This is performed using the library PyLops, which uses linear
        operators to avoid these matrices being created explicitly in memory, making the calculation more efficient.

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
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def mapped_reconstructed_data_dict_from(
        self, reconstruction: np.ndarray
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

        The PyLops calculation bypasses the calculation of the `mapping_matrix` and it therefore cannot be used to map
        the reconstruction's values to the data frame. Instead, the unique data-to-pixelization mappings are used,
        including the 2D non-uniform fast Fourier transform operation after mapping is complete.

        Parameters
        ----------
        reconstruction
            The reconstruction (in the source frame) whose values are mapped to a dictionary of values for each
            individual mapper (in the data frame).
        """

        mapped_reconstructed_image_dict = self.mapped_reconstructed_image_dict_from(
            reconstruction=reconstruction
        )

        return {
            linear_obj: self.transformer.visibilities_from(image=image)
            for linear_obj, image in mapped_reconstructed_image_dict.items()
        }
