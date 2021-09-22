import numpy as np
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.inversion.linear_eqn.abstract import AbstractLinearEqn
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.inversion.regularizations.abstract import AbstractRegularization
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.inversion import inversion_util


class AbstractLinearEqnInterferometer(AbstractLinearEqn):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
        profiling_dict: Optional[Dict] = None,
    ):

        super().__init__(
            noise_map=noise_map,
            mapper=mapper,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )

        self.transformer = transformer

    @cached_property
    def transformed_mapping_matrix(self) -> np.ndarray:

        return self.transformer.transform_mapping_matrix(
            mapping_matrix=self.mapper.mapping_matrix
        )

    def mapped_reconstructed_image_from(self, reconstruction: np.ndarray):

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=self.mapper.mapping_matrix, reconstruction=reconstruction
        )

        return Array2D(
            array=mapped_reconstructed_image,
            mask=self.mapper.source_grid_slim.mask.mask_sub_1,
        )

    def mapped_reconstructed_visibilities_from(self, reconstruction: np.ndarray):
        raise NotImplementedError


class LinearEqnInterferometerMapping(AbstractLinearEqnInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
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
        mapper : inversion.Mapper
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
            mapper=mapper,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )

    def data_vector_from(self, data: Visibilities) -> np.ndarray:

        return inversion_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            visibilities=data,
            noise_map=self.noise_map,
        )

    @property
    def curvature_matrix(self) -> np.ndarray:

        real_curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.transformed_mapping_matrix.real,
            noise_map=self.noise_map.real,
        )

        imag_curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=self.transformed_mapping_matrix.imag,
            noise_map=self.noise_map.imag,
        )

        return np.add(real_curvature_matrix, imag_curvature_matrix)

    def mapped_reconstructed_visibilities_from(
        self, reconstruction: np.ndarray
    ) -> Visibilities:

        visibilities = inversion_util.mapped_reconstructed_visibilities_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            reconstruction=reconstruction,
        )

        return Visibilities(visibilities=visibilities)


class LinearEqnInterferometerLinearOperator(AbstractLinearEqnInterferometer):
    def __init__(
        self,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
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
        mapper : inversion.Mapper
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
            mapper=mapper,
            regularization=regularization,
            profiling_dict=profiling_dict,
        )

    @cached_property
    def preconditioner_matrix(self):

        curvature_matrix_approx = np.multiply(
            np.sum(self.noise_map.weight_list_ordered_1d),
            self.mapper.mapping_matrix.T @ self.mapper.mapping_matrix,
        )

        return np.add(curvature_matrix_approx, self.regularization_matrix)

    @cached_property
    def log_det_curvature_reg_matrix_term(self):
        return 2.0 * np.sum(
            np.log(np.diag(np.linalg.cholesky(self.preconditioner_matrix)))
        )

    def mapped_reconstructed_visibilities_from(
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
        return self.transformer.visibilities_from_image(
            image=self.mapped_reconstructed_image_from(reconstruction=reconstruction)
        )

    @property
    def errors(self):
        return None
