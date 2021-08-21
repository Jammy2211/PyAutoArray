import numpy as np
from typing import Union

from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.inversion.abstract import AbstractInversionMatrix
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.inversion.regularization import Regularization
from autoarray.inversion.mappers import MapperRectangular
from autoarray.inversion.mappers import MapperVoronoi
from autoarray.preloads import Preloads
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.dataset.imaging import WTildeImaging

from autoarray.inversion.inversion import inversion_util
from autoarray.inversion import mapper_util


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

        return InversionImagingMatrix.from_data_via_w_tilde(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            preloads=preloads,
        )

    return InversionImagingMatrix.from_data_via_pixelization_convolution(
        image=image,
        noise_map=noise_map,
        convolver=convolver,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
    )


class InversionImagingMatrix(AbstractInversion, AbstractInversionMatrix):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        curvature_matrix: np.ndarray,
        regularization_matrix: np.ndarray,
        curvature_reg_matrix: np.ndarray,
        reconstruction: np.ndarray,
        mapped_reconstructed_image: np.ndarray,
        settings: SettingsInversion,
        preload_log_det_regularization_matrix_term: float = None,
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
            noise_map=noise_map,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=reconstruction,
            settings=settings,
            preload_log_det_regularization_matrix_term=preload_log_det_regularization_matrix_term,
        )

        AbstractInversionMatrix.__init__(
            self=self,
            curvature_matrix=curvature_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            regularization_matrix=regularization_matrix,
        )

        self.image = image
        self.convolver = convolver
        self.mapped_reconstructed_image = mapped_reconstructed_image

    @classmethod
    def from_data_via_w_tilde(
        cls,
        image: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        w_tilde: WTildeImaging,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        settings=SettingsInversion(),
        preloads=Preloads(),
    ):

        if preloads.w_tilde is not None:

            w_tilde = preloads.w_tilde

        w_tilde.check_noise_map(noise_map=noise_map)

        data_to_pix_unique, data_weights, pix_lengths = mapper_util.data_slim_to_pixelization_unique_from(
            data_pixels=w_tilde.lengths.shape[0],
            pixelization_index_for_sub_slim_index=mapper.pixelization_index_for_sub_slim_index,
            sub_size=mapper.source_grid_slim.sub_size,
        )

        data_to_pix_unique = data_to_pix_unique.astype("int")
        pix_lengths = pix_lengths.astype("int")

        w_tilde_data = inversion_util.w_tilde_data_imaging_from(
            image_native=image.native,
            noise_map_native=noise_map.native,
            kernel_native=convolver.kernel.native,
            native_index_for_slim_index=image.mask._native_index_for_slim_index,
        )

        data_vector = inversion_util.data_vector_via_w_tilde_data_imaging_from(
            w_tilde_data=w_tilde_data,
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            pix_pixels=mapper.pixels,
        )

        curvature_matrix = inversion_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
            w_tilde_curvature_preload=w_tilde.curvature_preload,
            w_tilde_curvature_indexes=w_tilde.indexes,
            w_tilde_curvature_lengths=w_tilde.lengths,
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            pix_pixels=mapper.pixels,
        )

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

        reconstruction = inversion_util.reconstruction_from(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            settings=settings,
        )

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            reconstruction=reconstruction,
        )

        mapped_reconstructed_image = Array2D(
            array=mapped_reconstructed_image,
            mask=mapper.source_grid_slim.mask.mask_sub_1,
        )

        mapped_reconstructed_image = convolver.convolve_image_no_blurring(
            image=mapped_reconstructed_image
        )

        return InversionImagingMatrix(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            curvature_matrix=curvature_matrix,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            reconstruction=reconstruction,
            mapped_reconstructed_image=mapped_reconstructed_image,
            settings=settings,
            preload_log_det_regularization_matrix_term=preloads.log_det_regularization_matrix_term,
        )

    @classmethod
    def from_data_via_pixelization_convolution(
        cls,
        image: Array2D,
        noise_map: Array2D,
        convolver: Convolver,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: Regularization,
        settings=SettingsInversion(),
        preloads=Preloads(),
    ):

        if preloads.blurred_mapping_matrix is None:

            blurred_mapping_matrix = convolver.convolve_mapping_matrix(
                mapping_matrix=mapper.mapping_matrix
            )

        else:

            blurred_mapping_matrix = preloads.blurred_mapping_matrix

        data_vector = inversion_util.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=image,
            noise_map=noise_map,
        )

        if preloads.curvature_matrix_sparse_preload is None:

            curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
                mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
            )

        else:

            curvature_matrix = inversion_util.curvature_matrix_via_sparse_preload_from(
                mapping_matrix=blurred_mapping_matrix,
                noise_map=noise_map,
                curvature_matrix_sparse_preload=preloads.curvature_matrix_sparse_preload.astype(
                    "int"
                ),
                curvature_matrix_preload_counts=preloads.curvature_matrix_preload_counts.astype(
                    "int"
                ),
            )

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

        reconstruction = inversion_util.reconstruction_from(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            settings=settings,
        )

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
        )

        mapped_reconstructed_image = Array2D(
            array=mapped_reconstructed_image,
            mask=mapper.source_grid_slim.mask.mask_sub_1,
        )

        return InversionImagingMatrix(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            curvature_matrix=curvature_matrix,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            reconstruction=reconstruction,
            mapped_reconstructed_image=mapped_reconstructed_image,
            settings=settings,
            preload_log_det_regularization_matrix_term=preloads.log_det_regularization_matrix_term,
        )

    @property
    def blurred_mapping_matrix(self) -> np.ndarray:

        return self.convolver.convolve_mapping_matrix(
            mapping_matrix=self.mapper.mapping_matrix
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
