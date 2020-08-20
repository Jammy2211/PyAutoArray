import numpy as np

from autoconf import conf
from autoarray import exc
from autoarray.structures import arrays, grids, visibilities as vis
from autoarray.operators import convolver as conv, transformer as trans
from autoarray.inversion import regularization as reg, mappers
from autoarray.dataset import imaging, interferometer
from autoarray.util import inversion_util
from scipy.interpolate import griddata
from scipy import sparse
import pylops


class SettingsInversion:
    def __init__(self, tolerance=1e-5, use_linear_operators=False, check_solution=True):

        self.tolerance = tolerance
        self.use_linear_operators = use_linear_operators
        self.check_solution = check_solution

    @property
    def tag(self):
        return self.use_linear_operators_tag

    @property
    def use_linear_operators_tag(self):
        if not self.use_linear_operators:
            return ""
        else:
            return f"__{conf.instance.tag.get('inversion', 'use_linear_operators')}"


def inversion(masked_dataset, mapper, regularization, settings=SettingsInversion()):

    if isinstance(masked_dataset, imaging.MaskedImaging):

        return InversionImagingMatrix.from_data_mapper_and_regularization(
            image=masked_dataset.image,
            noise_map=masked_dataset.noise_map,
            convolver=masked_dataset.convolver,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
        )

    elif isinstance(masked_dataset, interferometer.MaskedInterferometer):

        return AbstractInversionInterferometer.from_data_mapper_and_regularization(
            visibilities=masked_dataset.visibilities,
            noise_map=masked_dataset.noise_map,
            transformer=masked_dataset.transformer,
            mapper=mapper,
            regularization=regularization,
            settings=settings,
        )


def log_determinant_of_matrix_cholesky(matrix):
    """There are two terms in the inversion's Bayesian log likelihood function which require the log determinant of \
    a matrix. These are (Nightingale & Dye 2015, Nightingale, Dye and Massey 2018):

    ln[det(F + H)] = ln[det(curvature_reg_matrix)]
    ln[det(H)]     = ln[det(regularization_matrix)]

    The curvature_reg_matrix is positive-definite, which means the above log determinants can be computed \
    efficiently (compared to using np.det) by using a Cholesky decomposition first and summing the log of each \
    diagonal term.

    Parameters
    -----------
    matrix : ndarray
        The positive-definite matrix the log determinant is computed for.
    """
    try:
        return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(matrix))))
    except np.linalg.LinAlgError:
        raise exc.InversionException()


class AbstractInversionMatrix:
    def __init__(
        self, curvature_reg_matrix: np.ndarray, regularization_matrix: np.ndarray
    ):

        self.curvature_reg_matrix = curvature_reg_matrix
        self.regularization_matrix = regularization_matrix

    @property
    def log_det_curvature_reg_matrix_term(self):
        return log_determinant_of_matrix_cholesky(self.curvature_reg_matrix)

    @property
    def errors_with_covariance(self):
        return np.linalg.inv(self.curvature_reg_matrix)

    @property
    def errors(self):
        return np.diagonal(self.errors_with_covariance)


class AbstractInversion:
    def __init__(
        self,
        noise_map: np.ndarray,
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        settings: SettingsInversion,
    ):

        self.noise_map = noise_map
        self.mapper = mapper
        self.regularization = regularization
        self.regularization_matrix = regularization_matrix
        self.reconstruction = reconstruction
        self.settings = settings

    def interpolated_reconstruction_from_shape_2d(self, shape_2d=None):
        return self.interpolated_values_from_shape_2d(
            values=self.reconstruction, shape_2d=shape_2d
        )

    def interpolated_errors_from_shape_2d(self, shape_2d=None):
        return self.interpolated_values_from_shape_2d(
            values=self.errors, shape_2d=shape_2d
        )

    def interpolated_values_from_shape_2d(self, values, shape_2d=None):

        if shape_2d is not None:

            grid = grids.Grid.bounding_box(
                bounding_box=self.mapper.pixelization_grid.extent,
                shape_2d=shape_2d,
                buffer_around_corners=False,
            )

        elif (
            conf.instance.general.get("inversion", "interpolated_grid_shape", str)
            in "image_grid"
        ):

            grid = self.mapper.grid

        elif (
            conf.instance.general.get("inversion", "interpolated_grid_shape", str)
            in "source_grid"
        ):

            dimension = int(np.sqrt(self.mapper.pixels))
            shape_2d = (dimension, dimension)

            grid = grids.Grid.bounding_box(
                bounding_box=self.mapper.pixelization_grid.extent,
                shape_2d=shape_2d,
                buffer_around_corners=False,
            )

        else:

            raise exc.InversionException(
                "In the genenal.ini config file a valid option was not found for the"
                "interpolated_grid_shape. Must be {image_grid, source_grid}"
            )

        interpolated_reconstruction = griddata(
            points=self.mapper.pixelization_grid,
            values=values,
            xi=grid.in_2d_binned,
            method="linear",
        )

        interpolated_reconstruction[np.isnan(interpolated_reconstruction)] = 0.0

        return arrays.Array.manual(
            array=interpolated_reconstruction, pixel_scales=grid.pixel_scales
        )

    @property
    def regularization_term(self):
        """ Compute the regularization term of an inversion. This term represents the sum of the difference in flux \
        between every pair of neighboring pixels. This is computed as:

        s_T * H * s = solution_vector.T * regularization_matrix * solution_vector

        The term is referred to as *G_l* in Warren & Dye 2003, Nightingale & Dye 2015.

        The above works include the regularization_matrix coefficient (lambda) in this calculation. In PyAutoLens, \
        this is already in the regularization matrix and thus implicitly included in the matrix multiplication.
        """
        return np.matmul(
            self.reconstruction.T,
            np.matmul(self.regularization_matrix, self.reconstruction),
        )

    @property
    def log_det_regularization_matrix_term(self):
        return log_determinant_of_matrix_cholesky(self.regularization_matrix)

    @property
    def brightest_reconstruction_pixel(self):
        return np.argmax(self.reconstruction)

    @property
    def brightest_reconstruction_pixel_centre(self):
        return grids.GridCoordinates(
            coordinates=[
                self.mapper.pixelization_grid[self.brightest_reconstruction_pixel]
            ]
        )


class InversionImagingMatrix(AbstractInversion, AbstractInversionMatrix):
    def __init__(
        self,
        image: arrays.Array,
        noise_map: arrays.Array,
        convolver: conv.Convolver,
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        blurred_mapping_matrix: np.ndarray,
        regularization_matrix: np.ndarray,
        curvature_reg_matrix: np.ndarray,
        reconstruction: np.ndarray,
        settings: SettingsInversion,
    ):
        """ An inversion, which given an input image and noise-map reconstructs the image using a linear inversion, \
        including a convolution that accounts for blurring.

        The inversion uses a 2D pixelization to perform the reconstruction by util each pixelization pixel to a \
        set of image pixels via a mapper. The reconstructed pixelization is smoothed via a regularization scheme to \
        prevent over-fitting noise.

        Parameters
        -----------
        image_1d : ndarray
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map : ndarray
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the util matrix with the PSF.
        mapper : inversion.mappers.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        blurred_mapping_matrix : ndarray
            The matrix representing the blurred mappings between the image's sub-grid of pixels and the pixelization \
            pixels.
        regularization_matrix : ndarray
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix : ndarray
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix : ndarray
            The curvature_matrix + regularization matrix.
        solution_vector : ndarray
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

        super(InversionImagingMatrix, self).__init__(
            noise_map=noise_map,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=reconstruction,
            settings=settings,
        )

        AbstractInversionMatrix.__init__(
            self=self,
            curvature_reg_matrix=curvature_reg_matrix,
            regularization_matrix=regularization_matrix,
        )

        self.image = image
        self.convolver = convolver
        self.blurred_mapping_matrix = blurred_mapping_matrix

    @classmethod
    def from_data_mapper_and_regularization(
        cls,
        image: arrays.Array,
        noise_map: arrays.Array,
        convolver: conv.Convolver,
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        settings=SettingsInversion(),
    ):

        blurred_mapping_matrix = convolver.convolve_mapping_matrix(
            mapping_matrix=mapper.mapping_matrix
        )

        data_vector = inversion_util.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=image,
            noise_map=noise_map,
        )

        curvature_matrix = inversion_util.curvature_matrix_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
        )

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        curvature_reg_matrix = np.add(curvature_matrix, regularization_matrix)

        try:
            values = np.linalg.solve(curvature_reg_matrix, data_vector)
        except np.linalg.LinAlgError:
            raise exc.InversionException()

        if settings.check_solution:
            if np.isclose(a=values[0], b=values[1], atol=1e-4).all():
                if np.isclose(a=values[0], b=values, atol=1e-4).all():
                    raise exc.InversionException()

        return InversionImagingMatrix(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            regularization=regularization,
            blurred_mapping_matrix=blurred_mapping_matrix,
            regularization_matrix=regularization_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            reconstruction=values,
            settings=settings,
        )

    @property
    def mapped_reconstructed_image(self):
        reconstructed_image = inversion_util.mapped_reconstructed_data_from(
            mapping_matrix=self.blurred_mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return arrays.Array(
            array=reconstructed_image,
            mask=self.mapper.grid.mask.mask_sub_1,
            store_in_1d=True,
        )

    @property
    def residual_map(self):
        return inversion_util.inversion_residual_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            mask_1d_index_for_sub_mask_1d_index=self.mapper.grid.mask.regions._mask_1d_index_for_sub_mask_1d_index,
            all_sub_mask_1d_indexes_for_pixelization_1d_index=self.mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index,
        )

    @property
    def normalized_residual_map(self):
        return inversion_util.inversion_normalized_residual_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            noise_map_1d=self.noise_map,
            mask_1d_index_for_sub_mask_1d_index=self.mapper.grid.mask.regions._mask_1d_index_for_sub_mask_1d_index,
            all_sub_mask_1d_indexes_for_pixelization_1d_index=self.mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index,
        )

    @property
    def chi_squared_map(self):
        return inversion_util.inversion_chi_squared_map_from(
            pixelization_values=self.reconstruction,
            data=self.image,
            noise_map_1d=self.noise_map,
            mask_1d_index_for_sub_mask_1d_index=self.mapper.grid.mask.regions._mask_1d_index_for_sub_mask_1d_index,
            all_sub_mask_1d_indexes_for_pixelization_1d_index=self.mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index,
        )


class AbstractInversionInterferometer(AbstractInversion):
    def __init__(
        self,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: mappers.Mapper,
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
        mapper: mappers.Mapper,
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
        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_from(
            mapping_matrix=self.mapper.mapping_matrix,
            reconstruction=self.reconstruction,
        )

        return arrays.Array(
            array=mapped_reconstructed_image,
            mask=self.mapper.grid.mask.mask_sub_1,
            store_in_1d=True,
        )


class InversionInterferometerMatrix(
    AbstractInversionInterferometer, AbstractInversionMatrix
):
    def __init__(
        self,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        transformed_mapping_matrices: np.ndarray,
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
        image_1d : ndarray
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map : ndarray
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the util matrix with the PSF.
        mapper : inversion.mappers.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        blurred_mapping_matrix : ndarray
            The matrix representing the blurred mappings between the image's sub-grid of pixels and the pixelization \
            pixels.
        regularization_matrix : ndarray
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix : ndarray
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix : ndarray
            The curvature_matrix + regularization matrix.
        solution_vector : ndarray
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
            curvature_reg_matrix=curvature_reg_matrix,
            regularization_matrix=regularization_matrix,
        )

        self.curvature_reg_matrix = curvature_reg_matrix
        self.transformed_mapping_matrices = transformed_mapping_matrices

    @classmethod
    def from_data_mapper_and_regularization(
        cls,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        settings=SettingsInversion(),
    ):

        transformed_mapping_matrices = transformer.transformed_mapping_matrices_from_mapping_matrix(
            mapping_matrix=mapper.mapping_matrix
        )

        real_data_vector = inversion_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=transformed_mapping_matrices[0],
            visibilities=visibilities[:, 0],
            noise_map=noise_map[:, 0],
        )

        imag_data_vector = inversion_util.data_vector_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=transformed_mapping_matrices[1],
            visibilities=visibilities[:, 1],
            noise_map=noise_map[:, 1],
        )

        real_curvature_matrix = inversion_util.curvature_matrix_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=transformed_mapping_matrices[0],
            noise_map=noise_map[:, 0],
        )

        imag_curvature_matrix = inversion_util.curvature_matrix_via_transformed_mapping_matrix_from(
            transformed_mapping_matrix=transformed_mapping_matrices[1],
            noise_map=noise_map[:, 1],
        )

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        data_vector = np.add(real_data_vector, imag_data_vector)

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
            regularization=regularization,
            transformed_mapping_matrices=transformed_mapping_matrices,
            regularization_matrix=regularization_matrix,
            curvature_reg_matrix=curvature_reg_matrix,
            reconstruction=values,
            settings=settings,
        )

    @property
    def mapped_reconstructed_visibilities(self):
        real_visibilities = inversion_util.mapped_reconstructed_data_from(
            mapping_matrix=self.transformed_mapping_matrices[0],
            reconstruction=self.reconstruction,
        )

        imag_visibilities = inversion_util.mapped_reconstructed_data_from(
            mapping_matrix=self.transformed_mapping_matrices[1],
            reconstruction=self.reconstruction,
        )

        return vis.Visibilities(
            visibilities_1d=np.stack((real_visibilities, imag_visibilities), axis=-1)
        )


class InversionInterferometerLinearOperator(AbstractInversionInterferometer):
    def __init__(
        self,
        visibilities: vis.Visibilities,
        noise_map: vis.VisibilitiesNoiseMap,
        transformer: trans.TransformerNUFFT,
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        regularization_matrix: np.ndarray,
        reconstruction: np.ndarray,
        settings: SettingsInversion,
    ):
        """ An inversion, which given an input image and noise-map reconstructs the image using a linear inversion, \
        including a convolution that accounts for blurring.

        The inversion uses a 2D pixelization to perform the reconstruction by util each pixelization pixel to a \
        set of image pixels via a mapper. The reconstructed pixelization is smoothed via a regularization scheme to \
        prevent over-fitting noise.

        Parameters
        -----------
        image_1d : ndarray
            Flattened 1D array of the observed image the inversion is fitting.
        noise_map : ndarray
            Flattened 1D array of the noise-map used by the inversion during the fit.
        convolver : imaging.convolution.Convolver
            The convolver used to blur the util matrix with the PSF.
        mapper : inversion.mappers.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        regularization : inversion.regularization.Regularization
            The regularization scheme applied to smooth the pixelization used to reconstruct the image for the \
            inversion

        Attributes
        -----------
        blurred_mapping_matrix : ndarray
            The matrix representing the blurred mappings between the image's sub-grid of pixels and the pixelization \
            pixels.
        regularization_matrix : ndarray
            The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
        curvature_matrix : ndarray
            The curvature_matrix between each pixelization pixel and all other pixelization pixels (F).
        curvature_reg_matrix : ndarray
            The curvature_matrix + regularization matrix.
        solution_vector : ndarray
            The vector containing the reconstructed fit to the hyper_galaxies.
        """

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
        mapper: mappers.Mapper,
        regularization: reg.Regularization,
        settings=SettingsInversion(),
    ):

        regularization_matrix = regularization.regularization_matrix_from_mapper(
            mapper=mapper
        )

        Aop = pylops.MatrixMult(
            sparse.bsr_matrix(mapper.mapping_matrix), dtype="complex64"
        )
        Fop = transformer

        Op = Fop * Aop

        Rop = reg.RegularizationLop(regularization_matrix=regularization_matrix)

        reconstruction = pylops.NormalEquationsInversion(
            Op=Op,
            Regs=None,
            epsNRs=[1.0],
            NRegs=[Rop],
            data=visibilities.as_complex,
            Weight=noise_map.Wop,
            tol=settings.tolerance,
        )

        # reconstruction = pylops.RegularizedInversion(
        #     Op=Op,
        #     Regs=None,
        #     #      epsNRs=[1.0],
        #     #      NRegs=[Rop],
        #     data=visibilities.as_complex,
        #     Weight=noise_map.Wop,
        #     #       tol=settings.tolerance,
        # )

        return InversionInterferometerLinearOperator(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mapper,
            regularization=regularization,
            regularization_matrix=regularization_matrix,
            reconstruction=np.real(reconstruction),
            settings=settings,
        )

    @property
    def log_det_curvature_reg_matrix_term(self):
        return 0.0

    @property
    def mapped_reconstructed_visibilities(self):

        return self.transformer.visibilities_from_image(
            image=self.mapped_reconstructed_image
        )
