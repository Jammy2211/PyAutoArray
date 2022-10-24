import logging
from typing import Optional

from autoconf import cached_property

from autoarray.dataset.abstract_dataset import AbstractDataset
from autoarray.dataset.imaging.settings import SettingsImaging
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.arrays.kernel_2d import Kernel2D

from autoarray import exc
from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)


class AbstractImaging(AbstractDataset):
    def __init__(
        self,
        image: Array2D,
        noise_map: Optional[Array2D] = None,
        psf: Kernel2D = None,
        settings: SettingsImaging = SettingsImaging(),
        pad_for_convolver: bool = False,
    ):
        """
        An abstract class for an imaging dataset, including the image data, noise-map and a point spread function (PSF).

        Parameters
        ----------
        image
            The array of the image data, for example in units of electrons per second.
        noise_map
            An array describing the RMS standard deviation error in each pixel, for example in units of electrons per
            second.
        psf
            An array describing the Point Spread Function kernel of the image which accounts for diffraction due to the
            telescope optics via 2D convolution.
        settings
            Controls settings of how the dataset is set up (e.g. the types of grids used to perform calculations).
        """

        self.pad_for_convolver = pad_for_convolver

        if pad_for_convolver and psf is not None:

            try:
                image.mask.blurring_mask_from(kernel_shape_native=psf.shape_native)
            except exc.MaskException:
                image = image.padded_before_convolution_from(
                    kernel_shape=psf.shape_native, mask_pad_value=1
                )
                if noise_map is not None:
                    noise_map = noise_map.padded_before_convolution_from(
                        kernel_shape=psf.shape_native, mask_pad_value=1
                    )
                logger.info(
                    f"The image and noise map of the `Imaging` objected have been padded to the dimensions"
                    f"{image.shape}. This is because the blurring region around the mask (which defines where"
                    f"PSF flux may be convolved into the masked region) extended beyond the edge of the image."
                    f""
                    f"This can be prevented by using a smaller mask, smaller PSF kernel size or manually padding"
                    f"the image and noise-map yourself."
                )

        super().__init__(data=image, noise_map=noise_map, settings=settings)

        self.psf_unormalized = psf

        if psf is not None:

            self.psf_normalized = Kernel2D.manual_native(
                array=psf.native, pixel_scales=psf.pixel_scales, normalize=True
            )

    @property
    def shape_native(self):
        return self.data.shape_native

    @property
    def image(self):
        return self.data

    @property
    def pixel_scales(self):
        return self.data.pixel_scales

    @property
    def psf(self):
        if self.settings.use_normalized_psf:
            return self.psf_normalized
        return self.psf_unormalized

    @cached_property
    def blurring_grid(self) -> Grid2D:
        """
        Returns a blurring-grid from a mask and the 2D shape of the PSF kernel.

        A blurring grid consists of all pixels that are masked (and therefore have their values set to (0.0, 0.0)),
        but are close enough to the unmasked pixels that their values will be convolved into the unmasked those pixels.
        This when computing images from light profile objects.

        This uses lazy allocation such that the calculation is only performed when the blurring grid is used, ensuring
        efficient set up of the `Imaging` class.

        Returns
        -------
        The blurring grid given the mask of the imaging data.
        """

        return self.grid.blurring_grid_via_kernel_shape_from(
            kernel_shape_native=self.psf.shape_native
        )

    @cached_property
    def convolver(self):
        """
        Returns a `Convolver` from a mask and 2D PSF kernel.

        The `Convolver` stores in memory the array indexing between the mask and PSF, enabling efficient 2D PSF
        convolution of images and matrices used for linear algebra calculations (see `operators.convolver`).

        This uses lazy allocation such that the calculation is only performed when the convolver is used, ensuring
        efficient set up of the `Imaging` class.

        Returns
        -------
        Convolver
            The convolver given the masked imaging data's mask and PSF.
        """
        return Convolver(mask=self.mask, kernel=self.psf)

    @cached_property
    def w_tilde(self):
        """
        The w_tilde formalism of the linear algebra equations precomputes the convolution of every pair of masked
        noise-map values given the PSF (see `inversion.inversion_util`).

        The `WTilde` object stores these precomputed values in the imaging dataset ensuring they are only computed once
        per analysis.

        This uses lazy allocation such that the calculation is only performed when the wtilde matrices are used,
        ensuring efficient set up of the `Imaging` class.

        Returns
        -------
        WTildeImaging
            Precomputed values used for the w tilde formalism of linear algebra calculations.
        """

        logger.info("IMAGING - Computing W-Tilde... May take a moment.")

        (
            curvature_preload,
            indexes,
            lengths,
        ) = inversion_imaging_util.w_tilde_curvature_preload_imaging_from(
            noise_map_native=self.noise_map.native,
            kernel_native=self.psf.native,
            native_index_for_slim_index=self.mask.native_index_for_slim_index,
        )

        return WTildeImaging(
            curvature_preload=curvature_preload,
            indexes=indexes.astype("int"),
            lengths=lengths.astype("int"),
            noise_map_value=self.noise_map[0],
        )
