import numpy as np
from typing import Dict, List, Optional, Union

from autoconf import cached_property

from autoarray.inversion.linear_eqn.abstract import AbstractLEq
from autoarray.dataset.interferometer import WTildeInterferometer
from autoarray.inversion.linear_obj import LinearObj
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.preloads import Preloads
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.linear_eqn import leq_util
from autoarray.inversion.inversion import inversion_interferometer_util

from autoarray.numba_util import profile_func


class AbstractLEqInterferometer(AbstractLEq):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):

        super().__init__(
            noise_map=noise_map,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

        self.transformer = transformer

    @cached_property
    @profile_func
    def transformed_mapping_matrix(self) -> np.ndarray:
        return np.hstack(
            [
                self.transformed_mapping_matrix_of_mapper(mapper_index=mapper_index)
                for mapper_index in range(self.total_mappers)
            ]
        )

    def transformed_mapping_matrix_of_mapper(self, mapper_index: int) -> np.ndarray:
        return self.transformer.transform_mapping_matrix(
            mapping_matrix=self.linear_obj_list[mapper_index].mapping_matrix
        )

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        return self.transformed_mapping_matrix

    def mapped_reconstructed_data_of_mappers_from(self, reconstruction: np.ndarray):
        raise NotImplementedError

    @profile_func
    def mapped_reconstructed_image_of_mappers_from(self, reconstruction: np.ndarray):

        mapped_reconstructed_image_of_mappers = []

        reconstruction_of_mappers = self.source_quantity_of_mappers_from(
            source_quantity=reconstruction
        )

        for mapper_index in range(self.total_mappers):

            mapper = self.linear_obj_list[mapper_index]
            reconstruction = reconstruction_of_mappers[mapper_index]

            mapped_reconstructed_image = leq_util.mapped_reconstructed_data_via_mapping_matrix_from(
                mapping_matrix=mapper.mapping_matrix, reconstruction=reconstruction
            )

            mapped_reconstructed_image = Array2D(
                array=mapped_reconstructed_image,
                mask=mapper.source_grid_slim.mask.mask_sub_1,
            )

            mapped_reconstructed_image_of_mappers.append(mapped_reconstructed_image)

        return mapped_reconstructed_image_of_mappers


class LEqInterferometerMapping(AbstractLEqInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):
        """
        An inversion, which given an input image and noise-map reconstructs the image using a linear inversion, \
        including a convolution that accounts for blurring.

        The inversion uses a 2D pixelization to perform the reconstruction by util each pixelization pixel to a \
        set of image pixels via a mapper. The reconstructed pixelization is smoothed via a regularization scheme to \
        prevent over-fitting noise.

        Parameters
        -----------
        image_1d
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        linear_obj_list : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        blurred_mapping_matrix
            The matrix representing the blurred mappings between the image's sub-grid of pixels and the pixelization \
            pixels.
        regularization_matrix
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix
            The curvature_matrix + regularization matrix.
        solution_vector
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

        super().__init__(
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def data_vector_from(self, data: Visibilities, preloads) -> np.ndarray:

        return leq_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            visibilities=data,
            noise_map=self.noise_map,
        )

    @cached_property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:

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
    def mapped_reconstructed_data_of_mappers_from(
        self, reconstruction: np.ndarray
    ) -> List[Visibilities]:

        mapped_reconstructed_data_of_mappers = []

        reconstruction_of_mappers = self.source_quantity_of_mappers_from(
            source_quantity=reconstruction
        )

        for mapper_index in range(self.total_mappers):

            reconstruction = reconstruction_of_mappers[mapper_index]
            transformed_mapping_matrix = self.transformed_mapping_matrix_of_mapper(
                mapper_index=mapper_index
            )

            visibilities = leq_util.mapped_reconstructed_visibilities_from(
                transformed_mapping_matrix=transformed_mapping_matrix,
                reconstruction=reconstruction,
            )

            visibilities = Visibilities(visibilities=visibilities)

            mapped_reconstructed_data_of_mappers.append(visibilities)

        return mapped_reconstructed_data_of_mappers


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
        An inversion, which given an input image and noise-map reconstructs the image using a linear inversion, \
        including a convolution that accounts for blurring.

        The inversion uses a 2D pixelization to perform the reconstruction by util each pixelization pixel to a \
        set of image pixels via a mapper. The reconstructed pixelization is smoothed via a regularization scheme to \
        prevent over-fitting noise.

        Parameters
        -----------
        image_1d
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        linear_obj_list : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion
        """

        self.w_tilde = w_tilde
        self.w_tilde.check_noise_map(noise_map=noise_map)

        super().__init__(
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def data_vector_from(self, data: Visibilities, preloads: Preloads) -> np.ndarray:
        """
        To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy
        linear algebra libraries to solve. The linear algebra is based on
        the paper https://arxiv.org/pdf/astro-ph/0302587.pdf .

        This requires us to convert `w_tilde_data` into a data vector matrices of dimensions [image_pixels].

        The `data_vector` D is the first such matrix, which is given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        The calculation is performed by the method `w_tilde_data_interferometer_from`.
        """
        return None

    @property
    @profile_func
    def curvature_matrix_diag(self) -> np.ndarray:
        """
        The `curvature_matrix` F is the second matrix, given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        This function computes F using the w_tilde formalism, which is faster as it precomputes the Fourier Transform
        of different visibilities noise-map (see `curvature_matrix_via_w_tilde_curvature_preload_interferometer_from`).

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        return inversion_interferometer_util.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
            curvature_preload=self.w_tilde.curvature_preload,
            pix_index_for_sub_slim_index=self.linear_obj_list[
                0
            ].pix_index_for_sub_slim_index,
            native_index_for_slim_index=self.transformer.real_space_mask.mask.native_index_for_slim_index,
            pixelization_pixels=self.linear_obj_list[0].pixels,
        )

    @profile_func
    def mapped_reconstructed_data_of_mappers_from(
        self, reconstruction: np.ndarray
    ) -> Visibilities:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane to
        reconstruct the image in real-space. We then apply the Fourier Transform to map this to the reconstructed
        visibilities.

        Returns
        -------
        Visibilities
            The reconstructed visibilities which the inversion fits.
        """
        return self.transformer.visibilities_from(
            image=self.mapped_reconstructed_data_of_mappers_from(
                reconstruction=reconstruction
            )
        )


class LEqInterferometerMapperPyLops(AbstractLEqInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        linear_obj_list: List[LinearObj],
        profiling_dict: Optional[Dict] = None,
    ):
        """ An inversion, which given an input image and noise-map reconstructs the image using a linear inversion, \
        including a convolution that accounts for blurring.

        The inversion uses a 2D pixelization to perform the reconstruction by util each pixelization pixel to a \
        set of image pixels via a mapper. The reconstructed pixelization is smoothed via a regularization scheme to \
        prevent over-fitting noise.

        Parameters
        -----------
        image_1d
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        linear_obj_list : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        blurred_mapping_matrix
            The matrix representing the blurred mappings between the image's sub-grid of pixels and the pixelization \
            pixels.
        regularization_matrix
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix
            The curvature_matrix + regularization matrix.
        solution_vector
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

        super().__init__(
            noise_map=noise_map,
            transformer=transformer,
            linear_obj_list=linear_obj_list,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def mapped_reconstructed_data_of_mappers_from(
        self, reconstruction: np.ndarray
    ) -> List[Visibilities]:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane to
        reconstruct the image in real-space. We then apply the Fourier Transform to map this to the reconstructed
        visibilities.

        Returns
        -------
        Visibilities
            The reconstructed visibilities which the inversion fits.
        """

        mapped_reconstructed_image_of_mappers = self.mapped_reconstructed_image_of_mappers_from(
            reconstruction=reconstruction
        )

        return [
            self.transformer.visibilities_from(image=mapped_reconstructed_image)
            for mapped_reconstructed_image in mapped_reconstructed_image_of_mappers
        ]
