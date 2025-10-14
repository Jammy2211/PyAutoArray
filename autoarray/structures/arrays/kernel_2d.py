from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray import Mask2D

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import scipy
from typing import List, Optional, Tuple, Union
import warnings

from autoconf import conf
from autoconf.fitsable import header_obj_from

from autoarray.structures.arrays.uniform_2d import AbstractArray2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.header import Header

from autoarray import type as ty


class Kernel2D(AbstractArray2D):
    def __init__(
        self,
        values,
        mask,
        header=None,
        normalize: bool = False,
        store_native: bool = False,
        image_mask=None,
        blurring_mask=None,
        mask_shape=None,
        full_shape=None,
        fft_shape=None,
        use_fft: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """
        A 2D convolution kernel stored as an array of values paired to a uniform 2D mask.

        The ``Kernel2D`` is a subclass of ``Array2D`` with additional methods for performing
        point spread function (PSF) convolution of images or mapping matrices. Each entry of
        the kernel corresponds to a PSF value at the centre of a pixel in the unmasked grid.

        Two convolution modes are supported:

        - **Real-space convolution**: performed directly via sliding-window summation or
          ``jax.scipy.signal.convolve``. This is exact but can be slow for large kernels.
        - **FFT convolution**: performed by transforming both the kernel and the input image
          into Fourier space, multiplying, and transforming back. This is typically faster
          for kernels larger than ~5×5, but requires careful zero-padding.

        When using FFT convolution, the input image and mask are automatically padded such
        that the FFT avoids circular wrap-around artefacts. This padding is computed from the
        kernel size via :meth:`fft_shape_from`. The padded shape is stored in ``fft_shape``.
        If FFT convolution is attempted without precomputing and applying this padding,
        an exception is raised to avoid silent shape mismatches.

        Parameters
        ----------
        values
            The raw 2D kernel values. Can be normalised to sum to unity if ``normalize=True``.
        mask
            The 2D mask associated with the kernel, defining the pixels each kernel value is
            paired with.
        header
            Optional metadata (e.g. FITS header) associated with the kernel.
        normalize
            If True, the kernel values are rescaled such that they sum to 1.0.
        store_native
            If True, the kernel is stored in its full native 2D form as an attribute
            ``stored_native`` for re-use (e.g. when convolving repeatedly).
        image_mask
            Optional mask defining the unmasked image pixels when performing convolution.
            If not provided, defaults to the supplied ``mask``.
        blurring_mask
            Optional mask defining the "blurring region": pixels outside the image mask
            into which PSF flux can spread. Used to construct blurring images and
            blurring mapping matrices.
        mask_shape
            The shape of the (unpadded) mask region. Used when cropping back results after
            FFT convolution.
        full_shape
            The unpadded image + kernel shape (``image_shape + kernel_shape - 1``).
        fft_shape
            The padded shape used in FFT convolution, typically computed via
            ``scipy.fft.next_fast_len`` for efficiency. Must be precomputed before calling
            FFT convolution methods.
        use_fft
            If True, convolution is performed in Fourier space with zero-padding.
            If False, convolution is performed in real space.
            If None, a default choice is made: real space for small kernels,
            FFT for large kernels.
        *args, **kwargs
            Passed to the ``Array2D`` constructor.

        Notes
        -----
        - FFT padding can be disabled globally with ``disable_fft_pad=True`` when
          constructing ``Imaging`` objects, in which case convolution will either
          use real space or proceed without padding.
        - Blurring masks ensure that PSF flux spilling outside the main image mask
          is included correctly. Omitting them may lead to underestimated PSF wings.
        - For unit tests with tiny kernels, FFT and real-space convolution may differ
          slightly due to edge and truncation effects.
        """

        super().__init__(
            values=values,
            mask=mask,
            header=header,
            store_native=store_native,
        )

        if normalize:
            self._array = np.divide(self._array, np.sum(self._array))

        self.stored_native = self.native

        self.slim_to_native_tuple = None

        if image_mask is not None:

            slim_to_native = image_mask.derive_indexes.native_for_slim.astype("int32")
            self.slim_to_native_tuple = (slim_to_native[:, 0], slim_to_native[:, 1])

        self.slim_to_native_blurring_tuple = None

        if blurring_mask is not None:

            slim_to_native_blurring = (
                blurring_mask.derive_indexes.native_for_slim.astype("int32")
            )
            self.slim_to_native_blurring_tuple = (
                slim_to_native_blurring[:, 0],
                slim_to_native_blurring[:, 1],
            )

        self.fft_shape = fft_shape

        self.mask_shape = None
        self.full_shape = None
        self.fft_psf = None

        if self.fft_shape is not None:

            self.mask_shape = mask_shape
            self.full_shape = full_shape
            self.fft_psf = jnp.fft.rfft2(self.native.array, s=self.fft_shape)
            self.fft_psf_mapping = jnp.expand_dims(self.fft_psf, 2)

        self._use_fft = use_fft

    @property
    def use_fft(self):
        if self._use_fft is None:
            return conf.instance["general"]["psf"]["use_fft_default"]

        return self._use_fft

    @classmethod
    def no_mask(
        cls,
        values: Union[np.ndarray, List],
        pixel_scales: ty.PixelScales,
        shape_native: Tuple[int, int] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        header: Optional[Header] = None,
        normalize: bool = False,
        image_mask=None,
        blurring_mask=None,
        mask_shape=None,
        full_shape=None,
        fft_shape=None,
        use_fft: Optional[bool] = None,
    ):
        """
        Create a Kernel2D (see *Kernel2D.__new__*) by inputting the kernel values in 1D or 2D, automatically
        determining whether to use the 'manual_slim' or 'manual_native' methods.

        See the manual_slim and manual_native methods for examples.
        Parameters
        ----------
        values
            The values of the array input as an ndarray of shape [total_unmasked_pixels] or a list of
            lists.
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        values = Array2D.no_mask(
            values=values,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
        )
        return Kernel2D(
            values=values,
            mask=values.mask,
            header=header,
            normalize=normalize,
            image_mask=image_mask,
            blurring_mask=blurring_mask,
            mask_shape=mask_shape,
            full_shape=full_shape,
            fft_shape=fft_shape,
            use_fft=use_fft,
        )

    @classmethod
    def full(
        cls,
        fill_value: float,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Create a Kernel2D (see *Kernel2D.__new__*) where all values are filled with an input fill value, analogous to
        the method numpy ndarray.full.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        fill_value
            The value all array elements are filled with.
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        return Kernel2D.no_mask(
            values=np.full(fill_value=fill_value, shape=shape_native),
            pixel_scales=pixel_scales,
            origin=origin,
            normalize=normalize,
        )

    @classmethod
    def ones(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ):
        """
        Create an Kernel2D (see *Kernel2D.__new__*) where all values are filled with ones, analogous to the method numpy
        ndarray.ones.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        return cls.full(
            fill_value=1.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            normalize=normalize,
        )

    @classmethod
    def zeros(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        origin: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Create an Kernel2D (see *Kernel2D.__new__*) where all values are filled with zeros, analogous to the method numpy
        ndarray.ones.

        From 1D input the method cannot determine the 2D shape of the array and its mask, thus the shape_native must be
        input into this method. The mask is setup as a unmasked `Mask2D` of shape_native.

        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """
        return cls.full(
            fill_value=0.0,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            origin=origin,
            normalize=normalize,
        )

    @classmethod
    def no_blur(cls, pixel_scales):
        """
        Setup the Kernel2D as a kernel which does not convolve any signal, which is simply an array of shape (1, 1)
        with value 1.

        Parameters
        ----------
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        """

        array = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

        return cls.no_mask(values=array, pixel_scales=pixel_scales)

    @classmethod
    def from_gaussian(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        sigma: float,
        centre: Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Setup the Kernel2D as a 2D symmetric elliptical Gaussian profile, according to the equation:

        (1.0 / (sigma * sqrt(2.0*pi))) * exp(-0.5 * (r/sigma)**2)


        Parameters
        ----------
        shape_native
            The 2D shape of the mask the array is paired with.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        sigma
            The value of sigma in the equation, describing the size and full-width half maximum of the Gaussian.
        centre
            The (y,x) central coordinates of the Gaussian.
        axis_ratio
            The axis-ratio of the elliptical Gaussian.
        angle
            The rotational angle of the Gaussian's ellipse defined counter clockwise from the positive x-axis.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """

        grid = Grid2D.uniform(shape_native=shape_native, pixel_scales=pixel_scales)
        grid_shifted = np.subtract(grid.array, centre)
        grid_radius = np.sqrt(np.sum(grid_shifted**2.0, 1))
        theta_coordinate_to_profile = np.arctan2(
            grid_shifted[:, 0], grid_shifted[:, 1]
        ) - np.radians(angle)
        grid_transformed = np.vstack(
            (
                grid_radius * np.sin(theta_coordinate_to_profile),
                grid_radius * np.cos(theta_coordinate_to_profile),
            )
        ).T

        grid_elliptical_radii = np.sqrt(
            np.add(
                np.square(grid_transformed[:, 1]),
                np.square(np.divide(grid_transformed[:, 0], axis_ratio)),
            )
        )

        gaussian = np.multiply(
            np.divide(1.0, sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(grid_elliptical_radii, sigma))),
        )

        return cls.no_mask(
            values=gaussian,
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            normalize=normalize,
        )

    @classmethod
    def from_as_gaussian_via_alma_fits_header_parameters(
        cls,
        shape_native: Tuple[int, int],
        pixel_scales: ty.PixelScales,
        y_stddev: float,
        x_stddev: float,
        theta: float,
        centre: Tuple[float, float] = (0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":

        from astropy import units

        x_stddev = (
            x_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            y_stddev * (units.deg).to(units.arcsec) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        axis_ratio = x_stddev / y_stddev

        return Kernel2D.from_gaussian(
            shape_native=shape_native,
            pixel_scales=pixel_scales,
            sigma=y_stddev,
            axis_ratio=axis_ratio,
            angle=90.0 - theta,
            centre=centre,
            normalize=normalize,
        )

    @classmethod
    def from_fits(
        cls,
        file_path: Union[Path, str],
        hdu: int,
        pixel_scales,
        origin=(0.0, 0.0),
        normalize: bool = False,
    ) -> "Kernel2D":
        """
        Loads the Kernel2D from a .fits file.

        Parameters
        ----------
        file_path
            The path the file is loaded from, including the filename and the ``.fits`` extension,
            e.g. '/path/to/filename.fits'
        hdu
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales
            The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
        origin
            The (y,x) scaled units origin of the mask's coordinate system.
        normalize
            If True, the Kernel2D's array values are normalized such that they sum to 1.0.
        """

        array = Array2D.from_fits(
            file_path=file_path, hdu=hdu, pixel_scales=pixel_scales, origin=origin
        )

        header_sci_obj = header_obj_from(file_path=file_path, hdu=0)
        header_hdu_obj = header_obj_from(file_path=file_path, hdu=hdu)

        return Kernel2D(
            values=array[:],
            mask=array.mask,
            normalize=normalize,
            header=Header(header_sci_obj=header_sci_obj, header_hdu_obj=header_hdu_obj),
        )

    def fft_shape_from(
        self, mask: np.ndarray
    ) -> Union[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Compute the padded shapes required for FFT-based convolution with this kernel.

        FFT convolution requires the input image and kernel to be zero-padded so that
        the convolution is equivalent to linear convolution (not circular) and to avoid
        wrap-around artefacts. This method inspects the mask and the kernel shape to
        determine three key shapes:

        - ``mask_shape``: the rectangular bounding-box region of the mask that encloses
          all unmasked (False) pixels, padded by half the kernel size in each direction.
          This is the minimal region that must be retained for convolution.
        - ``full_shape``: the "linear convolution shape", equal to
          ``mask_shape + kernel_shape - 1``. This is the minimal padded size required
          for an exact linear convolution.
        - ``fft_shape``: the FFT-efficient padded shape, obtained by rounding each
          dimension of ``full_shape`` up to the next fast length for real FFTs
          (via ``scipy.fft.next_fast_len``). Using this ensures efficient FFT execution.

        Parameters
        ----------
        mask
            A 2D mask where False indicates unmasked pixels (valid data) and True
            indicates masked pixels. The bounding-box of the False region is used
            to compute the convolution region.

        Returns
        -------
        full_shape
            The unpadded linear convolution shape (mask region + kernel − 1).
        fft_shape
            The FFT-friendly padded shape for efficient convolution.
        mask_shape
            The rectangular mask region size including kernel padding.
        """

        ys, xs = np.where(~mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        (pad_y, pad_x) = self.shape_native

        mask_shape = (
            (y_max + pad_y // 2) - (y_min - pad_y // 2),
            (x_max + pad_x // 2) - (x_min - pad_x // 2),
        )

        full_shape = tuple(s1 + s2 - 1 for s1, s2 in zip(mask_shape, self.shape_native))
        fft_shape = tuple(scipy.fft.next_fast_len(s, real=True) for s in full_shape)

        return full_shape, fft_shape, mask_shape

    @property
    def normalized(self) -> "Kernel2D":
        """
        Normalize the Kernel2D such that its data_vector values sum to unity.
        """
        return Kernel2D(values=self, mask=self.mask, normalize=True)

    def mapping_matrix_native_from(
        self,
        mapping_matrix: jnp.ndarray,
        mask: "Mask2D",
        blurring_mapping_matrix: Optional[jnp.ndarray] = None,
        blurring_mask: Optional["Mask2D"] = None,
    ) -> jnp.ndarray:
        """
        Expand a slim mapping matrix (image-plane) and optional blurring mapping matrix
        into a full native 3D cube (ny, nx, n_src).

        This is primarily used for real-space convolution, where the pixel-to-source
        mapping must be represented on the full image grid.

        Parameters
        ----------
        mapping_matrix : ndarray (N_pix, N_src)
            Slim mapping matrix for unmasked image pixels, mapping each image pixel
            to source-plane pixels.
        mask : Mask2D
            Mask defining which image pixels are unmasked. Used to expand the slim
            mapping matrix into a native grid.
        blurring_mapping_matrix : ndarray (N_blur, N_src), optional
            Mapping matrix for blurring pixels outside the main mask (e.g. light
            spilling in from outside). If provided, it is also scattered into the
            native cube.
        blurring_mask : Mask2D, optional
            Mask defining the blurring region pixels. Must be provided if
            `blurring_mapping_matrix` is given and `slim_to_native_blurring_tuple`
            is not already cached.

        Returns
        -------
        ndarray (ny, nx, N_src)
            Native 3D mapping matrix cube with dimensions (image_y, image_x, sources).
            Contains contributions from both the main mapping matrix and, if provided,
            the blurring mapping matrix.
        """
        slim_to_native_tuple = self.slim_to_native_tuple
        if slim_to_native_tuple is None:
            slim_to_native_tuple = jnp.nonzero(
                jnp.logical_not(mask.array), size=mapping_matrix.shape[0]
            )

        n_src = mapping_matrix.shape[1]

        # Allocate full native grid (ny, nx, n_src)
        mapping_matrix_native = jnp.zeros(
            mask.shape + (n_src,), dtype=mapping_matrix.dtype
        )

        # Scatter main mapping matrix into native cube
        mapping_matrix_native = mapping_matrix_native.at[slim_to_native_tuple].set(
            mapping_matrix
        )

        # Optionally scatter blurring mapping matrix
        if blurring_mapping_matrix is not None:
            slim_to_native_blurring_tuple = self.slim_to_native_blurring_tuple

            if slim_to_native_blurring_tuple is None:
                if blurring_mask is None:
                    raise ValueError(
                        "blurring_mask must be provided if blurring_mapping_matrix is given "
                        "and slim_to_native_blurring_tuple is None."
                    )
                slim_to_native_blurring_tuple = jnp.nonzero(
                    jnp.logical_not(blurring_mask.array),
                    size=blurring_mapping_matrix.shape[0],
                )

            mapping_matrix_native = mapping_matrix_native.at[
                slim_to_native_blurring_tuple
            ].set(blurring_mapping_matrix)

        return mapping_matrix_native

    def convolved_image_from(self, image, blurring_image, jax_method="direct"):
        """
        Convolve an input masked image with this PSF.

        This method chooses between an FFT-based convolution (default if
        ``self.use_fft=True``) or a direct real-space convolution, depending on
        how the Kernel2D was configured.

        In the FFT branch:
        - The input image (and optional blurring image) are resized / padded to
          match the FFT-friendly padded shape (``fft_shape``) associated with this kernel.
        - The PSF and image are transformed to Fourier space via ``jax.numpy.fft.rfft2``.
        - Convolution is performed as elementwise multiplication.
        - The result is inverse-transformed and cropped back to the masked region.

        Padding ensures that the FFT implements *linear* convolution, not circular,
        and avoids wrap-around artefacts. The required padding is determined by
        ``fft_shape_from(mask)``. If no precomputed shapes exist, they are computed
        on the fly. For reproducible behaviour, precompute and set
        ``fft_shape``, ``full_shape``, and ``mask_shape`` on the kernel.

        If ``use_fft=False``, convolution falls back to
        :meth:`Kernel2D.convolved_image_via_real_space_from`.

        Parameters
        ----------
        image
            Masked 2D image array to convolve.
        blurring_image
            Masked image containing flux from outside the mask core that blurs
            into the masked region after convolution. If ``None``, only the direct
            image is convolved, which may be numerically incorrect if the mask
            excludes PSF wings.
        jax_method : {"direct", "fft"}
            Backend passed to ``jax.scipy.signal.convolve`` when in real-space mode.
            Ignored for FFT convolutions.

        Returns
        -------
        Array2D
            The convolved image in slim (1D masked) format.
        """

        if not self.use_fft:
            return self.convolved_image_via_real_space_from(
                image=image, blurring_image=blurring_image, jax_method=jax_method
            )

        if self.fft_shape is None:

            full_shape, fft_shape, mask_shape = self.fft_shape_from(mask=image.mask)
            fft_psf = jnp.fft.rfft2(self.stored_native.array, s=fft_shape)

            image_shape_original = image.shape_native

            image = image.resized_from(new_shape=fft_shape, mask_pad_value=1)
            if blurring_image is not None:
                blurring_image = blurring_image.resized_from(
                    new_shape=fft_shape, mask_pad_value=1
                )

        else:

            fft_shape = self.fft_shape
            full_shape = self.full_shape
            mask_shape = self.mask_shape
            fft_psf = self.fft_psf

        slim_to_native_tuple = self.slim_to_native_tuple
        slim_to_native_blurring_tuple = self.slim_to_native_blurring_tuple

        if slim_to_native_tuple is None:
            slim_to_native_tuple = jnp.nonzero(
                jnp.logical_not(image.mask.array), size=image.shape[0]
            )

        # start with native image padded with zeros
        image_both_native = jnp.zeros(image.mask.shape, dtype=image.dtype)
        image_both_native = image_both_native.at[slim_to_native_tuple].set(
            jnp.asarray(image.array)
        )

        # add blurring contribution if provided
        if blurring_image is not None:
            if slim_to_native_blurring_tuple is None:
                slim_to_native_blurring_tuple = jnp.nonzero(
                    jnp.logical_not(blurring_image.mask.array),
                    size=blurring_image.shape[0],
                )
            image_both_native = image_both_native.at[slim_to_native_blurring_tuple].set(
                jnp.asarray(blurring_image.array)
            )
        else:
            warnings.warn(
                "No blurring_image provided. Only the direct image will be convolved. "
                "This may change the correctness of the PSF convolution."
            )

        # FFT the combined image
        fft_image_native = jnp.fft.rfft2(image_both_native, s=fft_shape, axes=(0, 1))

        # Multiply by PSF in Fourier space and invert
        blurred_image_full = jnp.fft.irfft2(
            fft_psf * fft_image_native, s=fft_shape, axes=(0, 1)
        )

        # Crop back to mask_shape
        start_indices = tuple(
            (full_size - out_size) // 2
            for full_size, out_size in zip(full_shape, mask_shape)
        )
        out_shape_full = mask_shape
        blurred_image_native = jax.lax.dynamic_slice(
            blurred_image_full, start_indices, out_shape_full
        )

        blurred_image = Array2D(
            values=blurred_image_native[slim_to_native_tuple], mask=image.mask
        )

        if self.fft_shape is None:

            blurred_image = blurred_image.resized_from(
                new_shape=image_shape_original, mask_pad_value=0
            )

        return blurred_image

    def convolved_mapping_matrix_from(
        self,
        mapping_matrix,
        mask,
        blurring_mapping_matrix=None,
        blurring_mask: Optional[Mask2D] = None,
        jax_method="direct",
    ):
        """
        Convolve a source-plane mapping matrix with this PSF.

        A mapping matrix maps image-plane unmasked pixels to source-plane pixels.
        This method performs the equivalent operation of PSF convolution on the
        mapping matrix, so that model visibilities / images can be computed via
        matrix multiplication instead of explicit convolution.

        If ``use_fft=True``, convolution is performed in Fourier space:
        - The mapping matrix is scattered into a 3D native cube
          (ny, nx, n_src).
        - An FFT of this cube is multiplied by the precomputed FFT of the PSF.
        - The inverse FFT is taken and cropped to the mask region.
        - The slim (masked 1D) representation is returned.

        If ``use_fft=False``, convolution falls back to
        :meth:`Kernel2D.convolved_mapping_matrix_via_real_space_from`.

        Notes
        -----
        - FFT convolution requires that ``self.fft_shape`` and related padding
          attributes are precomputed. If not, a ``ValueError`` is raised with the
          expected vs actual shapes. This ensures the mapping matrix is padded
          consistently with the PSF.
        - The optional ``blurring_mapping_matrix`` plays the same role as
          ``blurring_image`` in :meth:`convolved_image_from`, accounting for PSF flux
          that falls into the masked region from outside.

        Parameters
        ----------
        mapping_matrix : ndarray of shape (N_pix, N_src)
            Slim mapping matrix from unmasked pixels to source pixels.
        mask : Mask2D
            Associated mask defining the image grid.
        blurring_mapping_matrix : ndarray of shape (N_blur, N_src), optional
            Mapping matrix for the blurring region, outside the mask core.
        jax_method : str
            Backend passed to real-space convolution if ``use_fft=False``.

        Returns
        -------
        ndarray of shape (N_pix, N_src)
            Convolved mapping matrix in slim form.
        """
        if not self.use_fft:
            return self.convolved_mapping_matrix_via_real_space_from(
                mapping_matrix=mapping_matrix,
                mask=mask,
                blurring_mapping_matrix=blurring_mapping_matrix,
                blurring_mask=blurring_mask,
                jax_method=jax_method,
            )

        if self.fft_shape is None:

            full_shape, fft_shape, mask_shape = self.fft_shape_from(mask=mask)

            raise ValueError(
                f"FFT convolution requires precomputed padded shapes, but `self.fft_shape` is None.\n"
                f"Expected mapping matrix padded to match FFT shape of PSF.\n"
                f"PSF fft_shape: {fft_shape}, mask shape: {mask.shape}, "
                f"mapping_matrix shape: {getattr(mapping_matrix, 'shape', 'unknown')}."
            )

        else:

            fft_shape = self.fft_shape
            full_shape = self.full_shape
            mask_shape = self.mask_shape
            fft_psf_mapping = self.fft_psf_mapping

        slim_to_native_tuple = self.slim_to_native_tuple

        if slim_to_native_tuple is None:
            slim_to_native_tuple = jnp.nonzero(
                jnp.logical_not(mask.array), size=mapping_matrix.shape[0]
            )

        mapping_matrix_native = self.mapping_matrix_native_from(
            mapping_matrix=mapping_matrix,
            mask=mask,
            blurring_mapping_matrix=blurring_mapping_matrix,
            blurring_mask=blurring_mask,
        )

        # FFT convolution
        fft_mapping_matrix_native = jnp.fft.rfft2(
            mapping_matrix_native, s=fft_shape, axes=(0, 1)
        )
        blurred_mapping_matrix_full = jnp.fft.irfft2(
            fft_psf_mapping * fft_mapping_matrix_native,
            s=fft_shape,
            axes=(0, 1),
        )

        # crop back
        start_indices = tuple(
            (full_size - out_size) // 2
            for full_size, out_size in zip(full_shape, mask_shape)
        ) + (0,)
        out_shape_full = mask_shape + (blurred_mapping_matrix_full.shape[2],)
        blurred_mapping_matrix_native = jax.lax.dynamic_slice(
            blurred_mapping_matrix_full, start_indices, out_shape_full
        )

        # return slim form
        return blurred_mapping_matrix_native[slim_to_native_tuple]

    def rescaled_with_odd_dimensions_from(
        self, rescale_factor: float, normalize: bool = False
    ) -> "Kernel2D":
        """
        Return a version of this kernel rescaled so both dimensions are odd-sized.

        Odd-sized kernels are often required for real space convolution operations
        (e.g. centered PSFs in imaging pipelines). If the kernel has one or two
        even-sized dimensions, they are rescaled (via interpolation) and padded
        so that both dimensions are odd.

        The kernel can also be scaled larger or smaller by changing
        ``rescale_factor``. Rescaling uses ``skimage.transform.rescale`` /
        ``resize``, which interpolate pixel values and may introduce small
        inaccuracies compared to native instrument PSFs. Where possible, users
        should generate odd-sized PSFs directly from data reduction.

        Parameters
        ----------
        rescale_factor
            Factor by which the kernel is rescaled. If 1.0, only adjusts size to
            nearest odd dimensions. Values > 1 enlarge, < 1 shrink the kernel.
        normalize
            If True, the returned kernel is normalized to sum to 1.0.

        Returns
        -------
        Kernel2D
            Rescaled kernel with odd-sized dimensions.
        """

        from skimage.transform import resize, rescale

        try:
            kernel_rescaled = rescale(
                self.native.array,
                rescale_factor,
                anti_aliasing=False,
                mode="constant",
                channel_axis=None,
            )
        except TypeError:
            kernel_rescaled = rescale(
                self.native.array,
                rescale_factor,
                anti_aliasing=False,
                mode="constant",
            )

        if kernel_rescaled.shape[0] % 2 == 0 and kernel_rescaled.shape[1] % 2 == 0:
            kernel_rescaled = resize(
                kernel_rescaled,
                output_shape=(
                    kernel_rescaled.shape[0] + 1,
                    kernel_rescaled.shape[1] + 1,
                ),
                anti_aliasing=False,
                mode="constant",
            )

        elif kernel_rescaled.shape[0] % 2 == 0 and kernel_rescaled.shape[1] % 2 != 0:
            kernel_rescaled = resize(
                kernel_rescaled,
                output_shape=(kernel_rescaled.shape[0] + 1, kernel_rescaled.shape[1]),
                anti_aliasing=False,
                mode="constant",
            )

        elif kernel_rescaled.shape[0] % 2 != 0 and kernel_rescaled.shape[1] % 2 == 0:
            kernel_rescaled = resize(
                kernel_rescaled,
                output_shape=(kernel_rescaled.shape[0], kernel_rescaled.shape[1] + 1),
                anti_aliasing=False,
                mode="constant",
            )

        if self.pixel_scales is not None:
            pixel_scale_factors = (
                self.mask.shape[0] / kernel_rescaled.shape[0],
                self.mask.shape[1] / kernel_rescaled.shape[1],
            )

            pixel_scales = (
                self.pixel_scales[0] * pixel_scale_factors[0],
                self.pixel_scales[1] * pixel_scale_factors[1],
            )

        else:
            pixel_scales = None

        return Kernel2D.no_mask(
            values=kernel_rescaled, pixel_scales=pixel_scales, normalize=normalize
        )

    def convolved_image_via_real_space_from(
        self,
        image: np.ndarray,
        blurring_image: Optional[np.ndarray] = None,
        jax_method: str = "direct",
    ):
        """
        Convolve an input masked image with this PSF in real space.

        This is the direct method (non-FFT) where convolution is explicitly
        performed using ``jax.scipy.signal.convolve`` with the kernel in native
        space.

        Unlike FFT convolution, this does not require padding shapes, but it is
        typically much slower for large kernels (> ~5x5).

        Parameters
        ----------
        image
            Masked image array to convolve.
        blurring_image
            Blurring contribution from outside the mask core. If None, only the
            direct image is convolved (which may be numerically incorrect).
        jax_method
            Method flag for JAX convolution backend (default "direct").

        Returns
        -------
        Array2D
            Convolved image in slim format.
        """

        slim_to_native_tuple = self.slim_to_native_tuple
        slim_to_native_blurring_tuple = self.slim_to_native_blurring_tuple

        if slim_to_native_tuple is None:
            slim_to_native_tuple = jnp.nonzero(
                jnp.logical_not(image.mask.array), size=image.shape[0]
            )

        # start with native array padded with zeros
        image_native = jnp.zeros(image.mask.shape, dtype=jnp.asarray(image.array).dtype)

        # set image pixels
        image_native = image_native.at[slim_to_native_tuple].set(
            jnp.asarray(image.array)
        )

        # add blurring contribution if provided
        if blurring_image is not None:
            if slim_to_native_blurring_tuple is None:
                slim_to_native_blurring_tuple = jnp.nonzero(
                    jnp.logical_not(blurring_image.mask.array),
                    size=blurring_image.shape[0],
                )
            image_native = image_native.at[slim_to_native_blurring_tuple].set(
                jnp.asarray(blurring_image.array)
            )
        else:
            warnings.warn(
                "No blurring_image provided. Only the direct image will be convolved. "
                "This may change the correctness of the PSF convolution."
            )

        # perform real-space convolution
        kernel = self.stored_native.array
        convolve_native = jax.scipy.signal.convolve(
            image_native, kernel, mode="same", method=jax_method
        )

        convolved_array_1d = convolve_native[slim_to_native_tuple]

        return Array2D(values=convolved_array_1d, mask=image.mask)

    def convolved_mapping_matrix_via_real_space_from(
        self,
        mapping_matrix: np.ndarray,
        mask,
        blurring_mapping_matrix: Optional[np.ndarray] = None,
        blurring_mask: Optional[Mask2D] = None,
        jax_method: str = "direct",
    ):
        """
        Convolve a source-plane mapping matrix with this PSF in real space.

        Equivalent to :meth:`convolved_mapping_matrix_from`, but using explicit
        real-space convolution rather than FFTs. This avoids FFT padding issues
        but is slower for large kernels.

        The mapping matrix is expanded into a native cube (ny, nx, n_src),
        convolved with the kernel (broadcast along the source axis),
        and reduced back to slim form.

        Parameters
        ----------
        mapping_matrix
            Slim mapping matrix from unmasked pixels to source pixels.
        mask
            Mask defining the pixelization grid.
        blurring_mapping_matrix : ndarray (N_blur, N_src), optional
            Mapping matrix for blurring region pixels outside the mask core.
        jax_method
            Backend passed to JAX convolution.

        Returns
        -------
        ndarray (N_pix, N_src)
            Convolved mapping matrix in slim form.
        """

        slim_to_native_tuple = self.slim_to_native_tuple

        if slim_to_native_tuple is None:
            slim_to_native_tuple = jnp.nonzero(
                jnp.logical_not(mask.array), size=mapping_matrix.shape[0]
            )

        mapping_matrix_native = self.mapping_matrix_native_from(
            mapping_matrix=mapping_matrix,
            mask=mask,
            blurring_mapping_matrix=blurring_mapping_matrix,
            blurring_mask=blurring_mask,
        )
        # 6) Real-space convolution, broadcast kernel over source axis
        kernel = self.stored_native.array
        blurred_mapping_matrix_native = jax.scipy.signal.convolve(
            mapping_matrix_native, kernel[..., None], mode="same", method=jax_method
        )

        # return slim form
        return blurred_mapping_matrix_native[slim_to_native_tuple]
