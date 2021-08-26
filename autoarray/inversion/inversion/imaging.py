import numpy as np
from typing import Union

from autoconf import cached_property

from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.inversion.regularization import Regularization
from autoarray.inversion.mappers import MapperRectangular
from autoarray.inversion.mappers import MapperVoronoi
from autoarray.preloads import Preloads
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.dataset.imaging import WTildeImaging

from autoarray import exc
from autoarray.inversion.inversion import inversion_util


def inversion_imaging_from(
    dataset,
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization: Regularization,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
):

    return inversion_imaging_unpacked_from(
        image=dataset.image,
        noise_map=dataset.noise_map,
        convolver=dataset.convolver,
        w_tilde=dataset.w_tilde,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
        preloads=preloads,
    )


def inversion_imaging_unpacked_from(
    image: Array2D,
    noise_map: Array2D,
    convolver: Convolver,
    w_tilde,
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization: Regularization,
    settings: SettingsInversion = SettingsInversion(),
    preloads: Preloads = Preloads(),
):

    if preloads.use_w_tilde is not None:
        use_w_tilde = preloads.use_w_tilde
    else:
        use_w_tilde = settings.use_w_tilde

    if use_w_tilde:

        return InversionImagingWTilde(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            preloads=preloads,
        )

    return InversionImagingMapping(
        image=image,
        noise_map=noise_map,
        convolver=convolver,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
        preloads=preloads,
    )


class AbstractInversionImaging(AbstractInversion):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
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
        convolver : convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        mapper : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        regularization_matrix
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix
            The curvature_matrix + regularization matrix.
        solution_vector
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

        self.image = image
        self.convolver = convolver

        super().__init__(
            noise_map=noise_map,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            preloads=preloads,
        )

    @cached_property
    def blurred_mapping_matrix(self) -> np.ndarray:

        if self.preloads.blurred_mapping_matrix is None:

            return self.convolver.convolve_mapping_matrix(
                mapping_matrix=self.mapper.mapping_matrix
            )

        else:

            return self.preloads.blurred_mapping_matrix

    @property
    def residual_map(self):
        return inversion_util.inversion_residual_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            slim_index_for_sub_slim_index=self.mapper.source_grid_slim.mask._slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper.all_sub_slim_indexes_for_pixelization_index,
        )

    @property
    def normalized_residual_map(self):
        return inversion_util.inversion_normalized_residual_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            noise_map_1d=self.noise_map,
            slim_index_for_sub_slim_index=self.mapper.source_grid_slim.mask._slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper.all_sub_slim_indexes_for_pixelization_index,
        )

    @property
    def chi_squared_map(self):
        return inversion_util.inversion_chi_squared_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            noise_map_1d=self.noise_map,
            slim_index_for_sub_slim_index=self.mapper.source_grid_slim.mask._slim_index_for_sub_slim_index,
            all_sub_slim_indexes_for_pixelization_index=self.mapper.all_sub_slim_indexes_for_pixelization_index,
        )

    @property
    def curvature_matrix_sparse_preload(self) -> np.ndarray:
        curvature_matrix_sparse_preload, curvature_matrix_preload_counts = inversion_util.curvature_matrix_sparse_preload_via_mapping_matrix_from(
            mapping_matrix=self.blurred_mapping_matrix
        )

        return curvature_matrix_sparse_preload

    @property
    def curvature_matrix_preload_counts(self) -> np.ndarray:
        curvature_matrix_sparse_preload, curvature_matrix_preload_counts = inversion_util.curvature_matrix_sparse_preload_via_mapping_matrix_from(
            mapping_matrix=self.blurred_mapping_matrix
        )

        return curvature_matrix_preload_counts


class InversionImagingWTilde(AbstractInversionImaging):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        w_tilde: WTildeImaging,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
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
        convolver : convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        mapper : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        regularization_matrix
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix
            The curvature_matrix + regularization matrix.
        solution_vector
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

        if preloads.w_tilde is not None:

            self.w_tilde = preloads.w_tilde

        else:

            self.w_tilde = w_tilde

        self.w_tilde.check_noise_map(noise_map=noise_map)

        super().__init__(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            preloads=preloads,
        )

    @cached_property
    def w_tilde_data(self):
        return inversion_util.w_tilde_data_imaging_from(
            image_native=self.image.native,
            noise_map_native=self.noise_map.native,
            kernel_native=self.convolver.kernel.native,
            native_index_for_slim_index=self.image.mask._native_index_for_slim_index,
        )

    @cached_property
    def data_vector(self) -> np.ndarray:

        return inversion_util.data_vector_via_w_tilde_data_imaging_from(
            w_tilde_data=self.w_tilde_data,
            data_to_pix_unique=self.mapper.data_unique_mappings.data_to_pix_unique,
            data_weights=self.mapper.data_unique_mappings.data_weights,
            pix_lengths=self.mapper.data_unique_mappings.pix_lengths,
            pix_pixels=self.mapper.pixels,
        )

    @cached_property
    def curvature_matrix(self) -> np.ndarray:
        return inversion_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
            w_tilde_curvature_preload=self.w_tilde.curvature_preload,
            w_tilde_curvature_indexes=self.w_tilde.indexes,
            w_tilde_curvature_lengths=self.w_tilde.lengths,
            data_to_pix_unique=self.mapper.data_unique_mappings.data_to_pix_unique,
            data_weights=self.mapper.data_unique_mappings.data_weights,
            pix_lengths=self.mapper.data_unique_mappings.pix_lengths,
            pix_pixels=self.mapper.pixels,
        )

    @cached_property
    def mapped_reconstructed_image(self):

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=self.mapper.data_unique_mappings.data_to_pix_unique,
            data_weights=self.mapper.data_unique_mappings.data_weights,
            pix_lengths=self.mapper.data_unique_mappings.pix_lengths,
            reconstruction=self.reconstruction,
        )

        mapped_reconstructed_image = Array2D(
            array=mapped_reconstructed_image,
            mask=self.mapper.source_grid_slim.mask.mask_sub_1,
        )

        return self.convolver.convolve_image_no_blurring(
            image=mapped_reconstructed_image
        )


class InversionImagingMapping(AbstractInversionImaging):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        settings: SettingsInversion = SettingsInversion(),
        preloads: Preloads = Preloads(),
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
        convolver : convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        mapper : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
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
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            preloads=preloads,
        )

    @cached_property
    def data_vector(self) -> np.ndarray:

        return inversion_util.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=self.blurred_mapping_matrix,
            image=self.image,
            noise_map=self.noise_map,
        )

    @cached_property
    def curvature_matrix(self) -> np.ndarray:

        if self.preloads.curvature_matrix_sparse_preload is None:

            return inversion_util.curvature_matrix_via_mapping_matrix_from(
                mapping_matrix=self.blurred_mapping_matrix, noise_map=self.noise_map
            )

        else:

            return inversion_util.curvature_matrix_via_sparse_preload_from(
                mapping_matrix=self.blurred_mapping_matrix,
                noise_map=self.noise_map,
                curvature_matrix_sparse_preload=self.preloads.curvature_matrix_sparse_preload,
                curvature_matrix_preload_counts=self.preloads.curvature_matrix_preload_counts,
            )

    @cached_property
    def mapped_reconstructed_image(self):

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=self.blurred_mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return Array2D(
            array=mapped_reconstructed_image,
            mask=self.mapper.source_grid_slim.mask.mask_sub_1,
        )
