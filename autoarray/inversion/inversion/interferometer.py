import numpy as np
from scipy import sparse
import pylops
from typing import Dict, Optional, Union

from autoconf import cached_property

from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.inversion.regularization import AbstractRegularization
from autoarray.operators.transformer import TransformerDFT
from autoarray.operators.transformer import TransformerNUFFT
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap

from autoarray.inversion.inversion import inversion_util


def inversion_interferometer_from(
    dataset,
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization,
    settings=SettingsInversion(),
    profiling_dict: Optional[Dict] = None,
):

    return inversion_interferometer_unpacked_from(
        visibilities=dataset.visibilities,
        noise_map=dataset.noise_map,
        transformer=dataset.transformer,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
        profiling_dict=profiling_dict,
    )


def inversion_interferometer_unpacked_from(
    visibilities: Visibilities,
    noise_map: VisibilitiesNoiseMap,
    transformer: Union[TransformerDFT, TransformerNUFFT],
    mapper: Union[MapperRectangular, MapperVoronoi],
    regularization: AbstractRegularization,
    settings: SettingsInversion = SettingsInversion(),
    profiling_dict: Optional[Dict] = None,
):
    if not settings.use_linear_operators:

        return InversionInterferometerMapping(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            profiling_dict=profiling_dict,
        )

    return InversionInterferometerLinearOperator(
        visibilities=visibilities,
        noise_map=noise_map,
        transformer=transformer,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
        profiling_dict=profiling_dict,
    )


class AbstractInversionInterferometer(AbstractInversion):
    def __init__(
        self,
        visibilities: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
        settings: SettingsInversion = SettingsInversion(),
        profiling_dict: Optional[Dict] = None,
    ):

        super().__init__(
            noise_map=noise_map,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            profiling_dict=profiling_dict,
        )

        self.visibilities = visibilities
        self.transformer = transformer

    @cached_property
    def transformed_mapping_matrix(self) -> np.ndarray:

        return self.transformer.transform_mapping_matrix(
            mapping_matrix=self.mapper.mapping_matrix
        )

    @property
    def mapped_reconstructed_image(self):

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=self.mapper.mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return Array2D(
            array=mapped_reconstructed_image,
            mask=self.mapper.source_grid_slim.mask.mask_sub_1,
        )

    @property
    def residual_map(self):
        return None

    @property
    def normalized_residual_map(self):
        return None

    @property
    def chi_squared_map(self):
        return None


class InversionInterferometerMapping(AbstractInversionInterferometer):
    def __init__(
        self,
        visibilities: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
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
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
            profiling_dict=profiling_dict,
        )

    @cached_property
    def data_vector(self) -> np.ndarray:

        return inversion_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            visibilities=self.visibilities,
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

    @property
    def mapped_reconstructed_visibilities(self):

        visibilities = inversion_util.mapped_reconstructed_visibilities_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return Visibilities(visibilities=visibilities)


class InversionInterferometerLinearOperator(AbstractInversionInterferometer):
    def __init__(
        self,
        visibilities: Visibilities,
        noise_map: VisibilitiesNoiseMap,
        transformer: TransformerNUFFT,
        mapper: Union[MapperRectangular, MapperVoronoi],
        regularization: AbstractRegularization,
        settings: SettingsInversion = SettingsInversion(),
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
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
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
    def preconditioner_matrix_inverse(self):
        return np.linalg.inv(self.preconditioner_matrix)

    @cached_property
    def reconstruction(self):

        Aop = pylops.MatrixMult(sparse.bsr_matrix(self.mapper.mapping_matrix))

        Fop = self.transformer

        Op = Fop * Aop

        MOp = pylops.MatrixMult(sparse.bsr_matrix(self.preconditioner_matrix_inverse))

        return pylops.NormalEquationsInversion(
            Op=Op,
            Regs=None,
            epsNRs=[1.0],
            data=self.visibilities.ordered_1d,
            Weight=pylops.Diagonal(diag=self.noise_map.weight_list_ordered_1d),
            NRegs=[pylops.MatrixMult(sparse.bsr_matrix(self.regularization_matrix))],
            M=MOp,
            tol=self.settings.tolerance,
            atol=self.settings.tolerance,
            **dict(maxiter=self.settings.maxiter),
        )

    @cached_property
    def log_det_curvature_reg_matrix_term(self):
        return 2.0 * np.sum(
            np.log(np.diag(np.linalg.cholesky(self.preconditioner_matrix)))
        )

    @property
    def mapped_reconstructed_visibilities(self) -> Visibilities:
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
            image=self.mapped_reconstructed_image
        )

    @property
    def errors(self):
        return None
