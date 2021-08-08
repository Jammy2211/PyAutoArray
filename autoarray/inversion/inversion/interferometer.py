import numpy as np

from autoarray import exc
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures import visibilities as vis
from autoarray.operators import transformer as trans
from autoarray.inversion.inversion import inversion_util
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.inversion.inversion.abstract import AbstractInversionMatrix
from autoarray.inversion.inversion.settings import SettingsInversion
from autoarray.inversion import regularization as reg, mappers
from scipy import sparse
import pylops
import typing
from typing import Union


def inversion_interferometer_from(
    dataset,
    mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
    regularization,
    settings=SettingsInversion(),
):

    return inversion_interferometer_unpacked_from(
        visibilities=dataset.visibilities,
        noise_map=dataset.noise_map,
        transformer=dataset.transformer,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
    )


def inversion_interferometer_unpacked_from(
    visibilities: vis.Visibilities,
    noise_map: vis.VisibilitiesNoiseMap,
    transformer: Union[trans.TransformerDFT, trans.TransformerNUFFT],
    mapper: Union[mappers.MapperRectangular, mappers.MapperVoronoi],
    regularization: reg.Regularization,
    settings: SettingsInversion = SettingsInversion(),
):
    return AbstractInversionInterferometer.from_data_mapper_and_regularization(
        visibilities=visibilities,
        noise_map=noise_map,
        transformer=transformer,
        mapper=mapper,
        regularization=regularization,
        settings=settings,
    )


class AbstractInversionInterferometer(AbstractInversion):
    def __init__(
        self,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        regularization: reg.Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        settings: SettingsInversion,
    ):

        super(AbstractInversionInterferometer, self).__init__(
            noise_map=noise_map,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=reconstruction,
            settings=settings,
        )

        self.visibilities = visibilities
        self.transformer = transformer

    @classmethod
    def from_data_mapper_and_regularization(
        cls,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        regularization: reg.Regularization,
        settings=SettingsInversion(use_linear_operators=True),
    ):

        if not settings.use_linear_operators:
            return InversionInterferometerMatrix.from_data_mapper_and_regularization(
                visibilities=visibilities,
                noise_map=noise_map,
                transformer=transformer,
                mapper=mapper,
                regularization=regularization,
                settings=settings,
            )
        else:
            return InversionInterferometerLinearOperator.from_data_mapper_and_regularization(
                visibilities=visibilities,
                noise_map=noise_map,
                transformer=transformer,
                mapper=mapper,
                regularization=regularization,
                settings=settings,
            )

    @property
    def mapped_reconstructed_image(self):

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=self.mapper.mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return array_2d.Array2D(
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


class InversionInterferometerMatrix(
    AbstractInversionInterferometer, AbstractInversionMatrix
):
    def __init__(
        self,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        regularization: reg.Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        transformed_mapping_matrix: np.ndarray,
        curvature_matrix: np.ndarray,
        curvature_reg_matrix: np.ndarray,
        settings: SettingsInversion,
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
        mapper : inversion.mappers.Mapper
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

        super(InversionInterferometerMatrix, self).__init__(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=reconstruction,
            settings=settings,
        )

        AbstractInversionMatrix.__init__(
            self=self,
            curvature_matrix=curvature_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            regularization_matrix=regularization_matrix,
        )

        self.curvature_reg_matrix = curvature_reg_matrix
        self.transformed_mapping_matrix = transformed_mapping_matrix

    @classmethod
    def from_data_mapper_and_regularization(
        cls,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        regularization: reg.Regularization,
        settings=SettingsInversion(),
    ):

        transformed_mapping_matrix = transformer.transform_mapping_matrix(
            mapping_matrix=mapper.mapping_matrix
        )

        data_vector = inversion_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=transformed_mapping_matrix,
            visibilities=visibilities,
            noise_map=noise_map,
        )

        real_curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=transformed_mapping_matrix.real, noise_map=noise_map.real
        )

        imag_curvature_matrix = inversion_util.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=transformed_mapping_matrix.imag, noise_map=noise_map.imag
        )

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        curvature_matrix = np.add(real_curvature_matrix, imag_curvature_matrix)
        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

        try:
            values = np.linalg.solve(curvature_reg_matrix, data_vector)
        except np.linalg.LinAlgError:
            raise exc.InversionException()

        if settings.check_solution:
            if np.isclose(a=values[0], b=values[1], atol=1e-4).all():
                if np.isclose(a=values[0], b=values, atol=1e-4).all():
                    raise exc.InversionException()

        return InversionInterferometerMatrix(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            curvature_matrix=curvature_matrix,
            regularization=regularization,
            transformed_mapping_matrix=transformed_mapping_matrix,
            regularization_matrix=regularization_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            reconstruction=values,
            settings=settings,
        )

    @property
    def mapped_reconstructed_visibilities(self):

        visibilities = inversion_util.mapped_reconstructed_visibilities_from(
            transformed_mapping_matrix=self.transformed_mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return vis.Visibilities(visibilities=visibilities)


class InversionInterferometerLinearOperator(AbstractInversionInterferometer):
    def __init__(
        self,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        regularization: reg.Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        log_det_curvature_reg_matrix_term: float,
        settings: SettingsInversion,
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
        mapper : inversion.mappers.Mapper
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

        self._log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term

        super(InversionInterferometerLinearOperator, self).__init__(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=reconstruction,
            settings=settings,
        )

    @classmethod
    def from_data_mapper_and_regularization(
        cls,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: typing.Union[mappers.MapperRectangular, mappers.MapperVoronoi],
        regularization: reg.Regularization,
        settings=SettingsInversion(),
    ):

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        Aop = pylops.MatrixMult(sparse.bsr_matrix(mapper.mapping_matrix))

        Fop = transformer

        Op = Fop * Aop

        curvature_matrix_approx = np.multiply(
            np.sum(noise_map.weight_list_ordered_1d),
            mapper.mapping_matrix.T @ mapper.mapping_matrix,
        )

        preconditioner_matrix = np.add(curvature_matrix_approx, regularization_matrix)

        preconditioner_inverse_matrix = np.linalg.inv(preconditioner_matrix)

        MOp = pylops.MatrixMult(sparse.bsr_matrix(preconditioner_inverse_matrix))

        log_det_curvature_reg_matrix_term = 2.0 * np.sum(
            np.log(np.diag(np.linalg.cholesky(preconditioner_matrix)))
        )

        reconstruction = pylops.NormalEquationsInversion(
            Op=Op,
            Regs=None,
            epsNRs=[1.0],
            data=visibilities.ordered_1d,
            Weight=pylops.Diagonal(diag=noise_map.weight_list_ordered_1d),
            NRegs=[pylops.MatrixMult(sparse.bsr_matrix(regularization_matrix))],
            M=MOp,
            tol=settings.tolerance,
            atol=settings.tolerance,
            **dict(maxiter=settings.maxiter),
        )

        return InversionInterferometerLinearOperator(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=np.real(reconstruction),
            settings=settings,
            log_det_curvature_reg_matrix_term=log_det_curvature_reg_matrix_term,
        )

    @property
    def log_det_curvature_reg_matrix_term(self):
        return self._log_det_curvature_reg_matrix_term

    @property
    def mapped_reconstructed_visibilities(self):

        return self.transformer.visibilities_from_image(
            image=self.mapped_reconstructed_image
        )

    @property
    def errors(self):
        return None
