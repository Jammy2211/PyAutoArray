import numpy as np
from typing import Dict, Optional, Union

from autoconf import cached_property
from autoarray.numba_util import profile_func

from autoarray.inversion.linear_eqn.abstract import AbstractLinearEqn
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.inversion.mappers.rectangular import MapperRectangular
from autoarray.inversion.mappers.voronoi import MapperVoronoi
from autoarray.preloads import Preloads
from autoarray.dataset.imaging import WTildeImaging

from autoarray.inversion.inversion import inversion_util


class AbstractLinearEqnImaging(AbstractLinearEqn):
    def __init__(
        self,
        noise_map: Array2D,
        convolver: Convolver,
        mapper: Union[MapperRectangular, MapperVoronoi],
        preloads: Preloads = Preloads(),
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

        self.convolver = convolver

        super().__init__(
            noise_map=noise_map,
            mapper=mapper,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @cached_property
    @profile_func
    def blurred_mapping_matrix(self) -> np.ndarray:
        """
        For a given pixelization pixel on the mapping matrix, we can use it to map it to a set of image-pixels in the
        image  plane. This therefore creates a 'image' of the source pixel (which corresponds to a set of values that
        mostly zeros, but with 1's where mappings occur).

        Before reconstructing the source, we blur every one of these source pixel images with the Point Spread Function
        of our  dataset via 2D convolution. This uses the methods
        in `Convolver.__init__` and `Convolver.convolve_mapping_matrix`:
        """
        if self.preloads.blurred_mapping_matrix is None:

            return self.convolver.convolve_mapping_matrix(
                mapping_matrix=self.mapper.mapping_matrix
            )

        return self.preloads.blurred_mapping_matrix

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        return self.blurred_mapping_matrix

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


class LinearEqnImagingWTilde(AbstractLinearEqnImaging):
    def __init__(
        self,
        noise_map: Array2D,
        convolver: Convolver,
        w_tilde: WTildeImaging,
        mapper: Union[MapperRectangular, MapperVoronoi],
        preloads: Preloads = Preloads(),
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
        mapper : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        """

        if preloads.w_tilde is not None:

            self.w_tilde = preloads.w_tilde

        else:

            self.w_tilde = w_tilde

        self.w_tilde.check_noise_map(noise_map=noise_map)

        super().__init__(
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def data_vector_from(self, data: Array2D) -> np.ndarray:
        """
        To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy
        linear  algebra libraries to solve. The linear algebra is based on
        the paper https://arxiv.org/pdf/astro-ph/0302587.pdf .

        This requires us to convert `w_tilde_data` into a data vector matrices of dimensions [image_pixels].

        The `data_vector` D is the first such matrix, which is given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        The calculation is performed by the method `w_tilde_data_imaging_from`.
        """

        w_tilde_data = inversion_util.w_tilde_data_imaging_from(
            image_native=data.native,
            noise_map_native=self.noise_map.native,
            kernel_native=self.convolver.kernel.native,
            native_index_for_slim_index=data.mask.native_index_for_slim_index,
        )

        return inversion_util.data_vector_via_w_tilde_data_imaging_from(
            w_tilde_data=w_tilde_data,
            data_to_pix_unique=self.mapper.data_unique_mappings.data_to_pix_unique,
            data_weights=self.mapper.data_unique_mappings.data_weights,
            pix_lengths=self.mapper.data_unique_mappings.pix_lengths,
            pix_pixels=self.mapper.pixels,
        )

    @property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` F is the second matrix, given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        This function computes F using the w_tilde formalism, which is faster as it precomputes the PSF convolution
        of different noise-map pixels (see `curvature_matrix_via_w_tilde_curvature_preload_imaging_from`).

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        return inversion_util.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
            curvature_preload=self.w_tilde.curvature_preload,
            curvature_indexes=self.w_tilde.indexes,
            curvature_lengths=self.w_tilde.lengths,
            data_to_pix_unique=self.mapper.data_unique_mappings.data_to_pix_unique,
            data_weights=self.mapper.data_unique_mappings.data_weights,
            pix_lengths=self.mapper.data_unique_mappings.pix_lengths,
            pix_pixels=self.mapper.pixels,
        )

    @profile_func
    def mapped_reconstructed_image_from(self, reconstruction: np.ndarray) -> Array2D:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """

        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=self.mapper.data_unique_mappings.data_to_pix_unique,
            data_weights=self.mapper.data_unique_mappings.data_weights,
            pix_lengths=self.mapper.data_unique_mappings.pix_lengths,
            reconstruction=reconstruction,
        )

        mapped_reconstructed_image = Array2D(
            array=mapped_reconstructed_image,
            mask=self.mapper.source_grid_slim.mask.mask_sub_1,
        )

        return self.convolver.convolve_image_no_blurring(
            image=mapped_reconstructed_image
        )


class LinearEqnImagingMapping(AbstractLinearEqnImaging):
    def __init__(
        self,
        noise_map: Array2D,
        convolver: Convolver,
        mapper: Union[MapperRectangular, MapperVoronoi],
        preloads: Preloads = Preloads(),
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
        convolver : convolution.Convolver
            The convolver used to blur the mapping matrix with the PSF.
        mapper : inversion.Mapper
            The util between the image-pixels (via its / sub-grid) and pixelization pixels.
        """

        super().__init__(
            noise_map=noise_map,
            convolver=convolver,
            mapper=mapper,
            preloads=preloads,
            profiling_dict=profiling_dict,
        )

    @profile_func
    def data_vector_from(self, data: Array2D) -> np.ndarray:
        """
        __Data Vector (D)__

        To solve for the source pixel fluxes we now pose the problem as a linear inversion which we use the NumPy
        linear  algebra libraries to solve. The linear algebra is based on the
        paper https://arxiv.org/pdf/astro-ph/0302587.pdf .

        This requires us to convert the blurred mapping matrix and our data / noise map into matrices of certain
        dimensions.

        The `data_vector` D is the first such matrix, which is given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.
        """
        return inversion_util.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=self.blurred_mapping_matrix,
            image=data,
            noise_map=self.noise_map,
        )

    @property
    @profile_func
    def curvature_matrix(self) -> np.ndarray:
        """
        The `curvature_matrix` F is the second matrix, given by equation (4)
        in https://arxiv.org/pdf/astro-ph/0302587.pdf.

        This function computes F using the mapping matrix formalism, which is slower but must be used in circumstances
        where the noise-map is varying.

        The `curvature_matrix` computed here is overwritten in memory when the regularization matrix is added to it,
        because for large matrices this avoids overhead. For this reason, `curvature_matrix` is not a cached property
        to ensure if we access it after computing the `curvature_reg_matrix` it is correctly recalculated in a new
        array of memory.
        """
        if self.preloads.curvature_matrix_sparse_preload is None:

            return inversion_util.curvature_matrix_via_mapping_matrix_from(
                mapping_matrix=self.blurred_mapping_matrix, noise_map=self.noise_map
            )

        return inversion_util.curvature_matrix_via_sparse_preload_from(
            mapping_matrix=self.blurred_mapping_matrix,
            noise_map=self.noise_map,
            curvature_matrix_sparse_preload=self.preloads.curvature_matrix_sparse_preload,
            curvature_matrix_preload_counts=self.preloads.curvature_matrix_preload_counts,
        )

    @profile_func
    def mapped_reconstructed_image_from(self, reconstruction) -> Array2D:
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane (via
        the blurred mapping_matrix) and reconstruct the image data.

        This uses the blurring mapping matrix which describes the PSF convolved mappings of flux between every
        source pixel and image pixels, which is a quantity that is already computed when using the mapping formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """
        mapped_reconstructed_image = inversion_util.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=self.blurred_mapping_matrix, reconstruction=reconstruction
        )

        return Array2D(
            array=mapped_reconstructed_image,
            mask=self.mapper.source_grid_slim.mask.mask_sub_1,
        )
