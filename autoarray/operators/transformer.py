import copy
import jax
import jax.numpy as jnp
import numpy as np
import warnings
from typing import Tuple


class NUFFTPlaceholder:
    pass


try:
    from pynufft.linalg.nufft_cpu import NUFFT_cpu
except ModuleNotFoundError:
    NUFFT_cpu = NUFFTPlaceholder


from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.visibilities import Visibilities

from autoarray.structures.arrays import array_2d_util
from autoarray.operators import transformer_util


def pynufft_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to perform interferometer analysis.\n\n"
        "However, the optional library PyNUFFT (https://github.com/jyhmiinlin/pynufft) is not installed.\n\n"
        "Install it via the command `pip install pynufft==2022.2.2`.\n\n"
        "----------------------"
    )


class TransformerDFT:
    def __init__(
        self,
        uv_wavelengths: np.ndarray,
        real_space_mask: Mask2D,
        preload_transform: bool = True,
    ):
        """
        A direct Fourier transform (DFT) operator for radio interferometric imaging.

        This class performs the forward and inverse mapping between real-space images and
        complex visibilities measured by an interferometer. It uses a direct implementation
        of the Fourier transform (not FFT-based), making it suitable for irregular uv-coverage.

        Optionally, it precomputes and stores the sine and cosine terms used in the transform,
        which can significantly improve performance for repeated operations but at the cost of memory.

        Parameters
        ----------
        uv_wavelengths
            The (u, v) coordinates in wavelengths of the measured visibilities.
        real_space_mask
            The real-space mask that defines the image grid and which pixels are valid.
        preload_transform
            If True, precomputes and stores the cosine and sine terms for the Fourier transform.
            This accelerates repeated transforms but consumes additional memory (~1GB+ for large datasets).

        Attributes
        ----------
        grid : ndarray
            The unmasked real-space grid in radians.
        total_visibilities : int
            The number of measured visibilities.
        total_image_pixels : int
            The number of unmasked pixels in the real-space image grid.
        preload_real_transforms : ndarray, optional
            The precomputed cosine terms used in the real part of the DFT.
        preload_imag_transforms : ndarray, optional
            The precomputed sine terms used in the imaginary part of the DFT.
        real_space_pixels : int
            Alias for `total_image_pixels`.
        adjoint_scaling : float
            Scaling factor applied to the adjoint operator to normalize the inverse transform.
        """
        super().__init__()

        self.uv_wavelengths = uv_wavelengths.astype("float")
        self.real_space_mask = real_space_mask
        self.grid = self.real_space_mask.derive_grid.unmasked.in_radians

        self.total_visibilities = uv_wavelengths.shape[0]
        self.total_image_pixels = self.real_space_mask.pixels_in_mask

        self.preload_transform = preload_transform

        if preload_transform:

            self.preload_real_transforms = (
                transformer_util.preload_real_transforms_from(
                    grid_radians=np.array(self.grid.array),
                    uv_wavelengths=self.uv_wavelengths,
                )
            )

            self.preload_imag_transforms = (
                transformer_util.preload_imag_transforms_from(
                    grid_radians=np.array(self.grid.array),
                    uv_wavelengths=self.uv_wavelengths,
                )
            )

        self.real_space_pixels = self.real_space_mask.pixels_in_mask

        # NOTE: This is the scaling factor that needs to be applied to the adjoint operator
        self.adjoint_scaling = (2.0 * self.grid.shape_native[0]) * (
            2.0 * self.grid.shape_native[1]
        )

    def visibilities_from(self, image: Array2D) -> Visibilities:
        """
        Computes the visibilities from a real-space image using the direct Fourier transform (DFT).

        This method transforms the input image into the uv-plane (Fourier space), simulating the
        measurements made by an interferometer at specified uv-wavelengths.

        If `preload_transform` is True, it uses precomputed sine and cosine terms to accelerate the computation.

        Parameters
        ----------
        image
            The real-space image to be transformed to the uv-plane. Must be defined on the
            same grid and mask as this transformer's `real_space_mask`.

        Returns
        -------
        The complex visibilities resulting from the Fourier transform of the input image.
        """
        if self.preload_transform:
            visibilities = transformer_util.visibilities_via_preload_from(
                image_1d=image.array,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
            )
        else:
            visibilities = transformer_util.visibilities_from(
                image_1d=image.slim.array,
                grid_radians=self.grid.array,
                uv_wavelengths=self.uv_wavelengths,
            )

        return Visibilities(visibilities=jnp.array(visibilities))

    def image_from(
        self, visibilities: Visibilities, use_adjoint_scaling: bool = False
    ) -> Array2D:
        """
        Computes the real-space image from a set of visibilities using the adjoint of the DFT.

        This is not a true inverse Fourier transform, but rather the adjoint operation, which maps
        complex visibilities back into image space. This is typically used as the first step
        in inverse imaging algorithms like CLEAN or regularized reconstruction.

        Parameters
        ----------
        visibilities
            The complex visibilities to be transformed into a real-space image.
        use_adjoint_scaling
            If True, the result is scaled by a normalization factor. Currently unused.

        Returns
        -------
        The real-space image resulting from the adjoint DFT operation, defined on the same
        mask as this transformer's `real_space_mask`.
        """
        image_slim = transformer_util.image_direct_from(
            visibilities=visibilities.in_array,
            grid_radians=self.grid.array,
            uv_wavelengths=self.uv_wavelengths,
        )

        image_native = array_2d_util.array_2d_native_from(
            array_2d_slim=image_slim,
            mask_2d=self.real_space_mask,
        )

        return Array2D(values=image_native, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix: np.ndarray) -> np.ndarray:
        """
        Applies the DFT to a mapping matrix that maps source pixels to image pixels.

        This is used in linear inversion frameworks, where the transform of each source basis function
        (represented by a column of the mapping matrix) is computed individually. The result is a matrix
        mapping source pixels directly to visibilities.

        If `preload_transform` is True, the computation is accelerated using precomputed sine and cosine terms.

        Parameters
        ----------
        mapping_matrix
            A 2D array of shape (n_image_pixels, n_source_pixels) that maps source pixels to image-plane pixels.

        Returns
        -------
        A 2D complex-valued array of shape (n_visibilities, n_source_pixels) that maps source-plane basis
        functions directly to the visibilities.
        """
        if self.preload_transform:
            return transformer_util.transformed_mapping_matrix_via_preload_from(
                mapping_matrix=mapping_matrix,
                preloaded_reals=self.preload_real_transforms,
                preloaded_imags=self.preload_imag_transforms,
            )

        return transformer_util.transformed_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            grid_radians=self.grid.array,
            uv_wavelengths=self.uv_wavelengths,
        )


class TransformerNUFFT(NUFFT_cpu):
    def __init__(self, uv_wavelengths: np.ndarray, real_space_mask: Mask2D, **kwargs):
        """
        Performs the Non-Uniform Fast Fourier Transform (NUFFT) for interferometric image reconstruction.

        This transformer uses the PyNUFFT library to efficiently compute the Fourier transform
        of an image defined on a regular real-space grid to a set of non-uniform uv-plane (Fourier space)
        coordinates, as is typical in radio interferometry.

        It is initialized with the interferometer uv-wavelengths and a real-space mask, which defines
        the pixelized image domain.

        Parameters
        ----------
        uv_wavelengths
            The uv-coordinates (Fourier-space sampling points) corresponding to the measured visibilities.
            Should be an array of shape (n_vis, 2), where the two columns represent u and v coordinates in wavelengths.

        real_space_mask
            The 2D mask defining the real-space pixel grid on which the image is defined. Used to create the
            unmasked grid required for NUFFT planning.

        Notes
        -----
        - The `initialize_plan()` method builds the internal NUFFT plan based on the input grid and uv sampling.
        - A complex exponential `shift` factor is applied to align the center of the Fourier transform correctly,
          accounting for the pixel-center offset in the real-space grid.
        - The adjoint operation (used in inverse imaging) must be scaled by `adjoint_scaling` to normalize its output.
        - This transformer inherits directly from PyNUFFT's `NUFFT_cpu` base class.
        - If `NUFFTPlaceholder` is detected (indicating PyNUFFT is not available), an exception is raised.

        Attributes
        ----------
        grid : Grid2D
            The real-space pixel grid derived from the mask, in radians.
        native_index_for_slim_index : np.ndarray
            Index map converting from slim (1D) grid to native (2D) indexing, for image reshaping.
        shift : np.ndarray
            Complex exponential phase shift applied to account for real-space pixel centering.
        real_space_pixels : int
            Total number of valid real-space pixels defined by the mask.
        total_visibilities : int
            Total number of visibilities across all uv-wavelength components.
        adjoint_scaling : float
            Scaling factor for adjoint operations to normalize reconstructed images.
        """
        from astropy import units

        if isinstance(self, NUFFTPlaceholder):
            pynufft_exception()

        super(TransformerNUFFT, self).__init__()

        self.uv_wavelengths = uv_wavelengths
        self.real_space_mask = real_space_mask
        #        self.grid = self.real_space_mask.unmasked_grid.in_radians
        self.grid = Grid2D.from_mask(mask=self.real_space_mask).in_radians
        self.native_index_for_slim_index = copy.copy(
            real_space_mask.derive_indexes.native_for_slim.astype("int")
        )

        # NOTE: The plan need only be initialized once
        self.initialize_plan()

        # ...
        self.shift = np.exp(
            -2.0
            * np.pi
            * 1j
            * (
                self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * self.uv_wavelengths[:, 1]
                + self.grid.pixel_scales[0]
                / 2.0
                * units.arcsec.to(units.rad)
                * self.uv_wavelengths[:, 0]
            )
        )

        self.real_space_pixels = self.real_space_mask.pixels_in_mask

        # NOTE: If reshaped the shape of the operator is (2 x Nvis, Np) else it is (Nvis, Np)
        self.total_visibilities = int(uv_wavelengths.shape[0] * uv_wavelengths.shape[1])

        # NOTE: This is the scaling factor that needs to be applied to the adjoint operator
        self.adjoint_scaling = (2.0 * self.grid.shape_native[0]) * (
            2.0 * self.grid.shape_native[1]
        )

    def initialize_plan(self, ratio: int = 2, interp_kernel: Tuple[int, int] = (6, 6)):
        """
        Initializes the PyNUFFT plan for performing the NUFFT operation.

        This method precomputes the interpolation structure and gridding
        needed by the NUFFT algorithm to map between the regular real-space
        image grid and the non-uniform uv-plane sampling defined by the
        interferometric visibilities.

        Parameters
        ----------
        ratio
            The oversampling ratio used to pad the Fourier grid before interpolation.
            A higher value improves accuracy at the cost of increased memory and computation.
            Default is 2 (i.e., the Fourier grid is twice the size of the image grid).

        interp_kernel
            The interpolation kernel size along each axis, given as (Jy, Jx).
            This determines how many neighboring Fourier grid points are used
            to interpolate each uv-point.
            Default is (6, 6), a good trade-off between accuracy and performance.

        Notes
        -----
        - The uv-coordinates are normalized and rescaled into the range expected by PyNUFFT
          using the real-space grid’s pixel scale and the Nyquist frequency limit.
        - The plan must be initialized before performing any NUFFT operations (e.g., forward or adjoint).
        - This method modifies the internal state of the NUFFT object by calling `self.plan(...)`.
        """
        from astropy import units

        if not isinstance(ratio, int):
            ratio = int(ratio)

        # ... NOTE : The u,v coordinated should be given in the order ...
        visibilities_normalized = np.array(
            [
                self.uv_wavelengths[:, 1]
                / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad)))
                * np.pi,
                self.uv_wavelengths[:, 0]
                / (1.0 / (2.0 * self.grid.pixel_scales[0] * units.arcsec.to(units.rad)))
                * np.pi,
            ]
        ).T

        # NOTE:
        self.plan(
            om=visibilities_normalized,
            Nd=self.grid.shape_native,
            Kd=(ratio * self.grid.shape_native[0], ratio * self.grid.shape_native[1]),
            Jd=interp_kernel,
        )

    def visibilities_from(self, image: Array2D) -> Visibilities:
        """
        Computes visibilities from a real-space image using the NUFFT forward transform.

        Parameters
        ----------
        image
            The input image in real space, represented as a 2D array object.

        Returns
        -------
        The complex visibilities in the uv-plane computed via the NUFFT forward operation.

        Notes
        -----
        - The image is flipped vertically before transformation to account for PyNUFFT’s internal data layout.
        - Warnings during the NUFFT computation are suppressed for cleaner output.
        """

        warnings.filterwarnings("ignore")

        return Visibilities(
            visibilities=self.forward(
                image.native.array[::-1, :]
            )  # flip due to PyNUFFT internal flip
        )

    def image_from(
        self, visibilities: Visibilities, use_adjoint_scaling: bool = False
    ) -> Array2D:
        """
        Reconstructs a real-space image from visibilities using the NUFFT adjoint transform.

        Parameters
        ----------
        visibilities
            The complex visibilities in the uv-plane to be inverted.
        use_adjoint_scaling
            If True, apply a scaling factor to the adjoint result to improve accuracy.
            Default is False.

        Returns
        -------
        The reconstructed real-space image after applying the NUFFT adjoint transform.

        Notes
        -----
        - The output image is flipped vertically to align with the input image orientation.
        - Warnings during the adjoint operation are suppressed.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = np.real(self.adjoint(visibilities))[::-1, :]

        if use_adjoint_scaling:
            image *= self.adjoint_scaling

        return Array2D(values=image, mask=self.real_space_mask)

    def transform_mapping_matrix(self, mapping_matrix: np.ndarray) -> np.ndarray:
        """
            Applies the NUFFT forward transform to each column of a mapping matrix, producing transformed visibilities.

            Parameters
            ----------
            mapping_matrix
                A 2D array where each column corresponds to a source-plane pixel intensity distribution flattened into image space.

            Returns
        -------
            A complex-valued 2D array where each column contains the visibilities corresponding to the respective column
            in the input mapping matrix.

            Notes
            -----
            - Each column of the input mapping matrix is reshaped into the native 2D image grid before transformation.
            - This method repeatedly calls `visibilities_from` for each column, which may be computationally intensive.
        """
        transformed_mapping_matrix = 0 + 0j * np.zeros(
            (self.uv_wavelengths.shape[0], mapping_matrix.shape[1])
        )

        for source_pixel_1d_index in range(mapping_matrix.shape[1]):
            image_2d = array_2d_util.array_2d_native_from(
                array_2d_slim=mapping_matrix[:, source_pixel_1d_index],
                mask_2d=self.grid.mask,
            )

            image = Array2D(values=image_2d, mask=self.grid.mask)

            visibilities = self.visibilities_from(image=image)

            transformed_mapping_matrix[:, source_pixel_1d_index] = visibilities

        return transformed_mapping_matrix
