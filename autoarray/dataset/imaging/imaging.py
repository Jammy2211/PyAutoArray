import copy
import logging
import numpy as np

from autoconf import cached_property

from autoarray.dataset.abstract_dataset import AbstractDataset
from autoarray.dataset.imaging.abstract import AbstractImaging
from autoarray.dataset.imaging.settings import SettingsImaging
from autoarray.dataset.imaging.w_tilde import WTildeImaging
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.arrays.kernel_2d import Kernel2D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.inversion.inversion.imaging import inversion_imaging_util

logger = logging.getLogger(__name__)


class Imaging(AbstractImaging):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        psf: Kernel2D = None,
        settings=SettingsImaging(),
        name: str = None,
        pad_for_convolver=False,
    ):
        """
        A class containing an imaging dataset, including the image data, noise-map and a point spread function (PSF).

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

        self.unmasked = None

        super().__init__(
            image=image,
            noise_map=noise_map,
            psf=psf,
            settings=settings,
            name=name,
            pad_for_convolver=pad_for_convolver,
        )

    @classmethod
    def from_fits(
        cls,
        image_path,
        pixel_scales,
        noise_map_path,
        image_hdu=0,
        noise_map_hdu=0,
        psf_path=None,
        psf_hdu=0,
        name=None,
    ):
        """
        Factory for loading the imaging data_type from .fits files, as well as computing properties like the noise-map,
        exposure-time map, etc. from the imaging-data.

        This factory also includes a number of routines for converting the imaging-data from unit_label not
        supported by PyAutoLens (e.g. adus, electrons) to electrons per second.

        Parameters
        ----------
        noise_map_non_constant
        name
        image_path
            The path to the image .fits file containing the image (e.g. '/path/to/image.fits')
        pixel_scales
            The size of each pixel in scaled units.
        image_hdu
            The hdu the image is contained in the .fits file specified by *image_path*.
        psf_path
            The path to the psf .fits file containing the psf (e.g. '/path/to/psf.fits')
        psf_hdu
            The hdu the psf is contained in the .fits file specified by *psf_path*.
        noise_map_path
            The path to the noise_map .fits file containing the noise_map (e.g. '/path/to/noise_map.fits')
        noise_map_hdu
            The hdu the noise_map is contained in the .fits file specified by *noise_map_path*.
        """

        image = Array2D.from_fits(
            file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
        )

        noise_map = Array2D.from_fits(
            file_path=noise_map_path, hdu=noise_map_hdu, pixel_scales=pixel_scales
        )

        if psf_path is not None:

            psf = Kernel2D.from_fits(
                file_path=psf_path,
                hdu=psf_hdu,
                pixel_scales=pixel_scales,
                normalize=False,
            )

        else:

            psf = None

        return Imaging(image=image, noise_map=noise_map, psf=psf, name=name)

    def apply_mask(self, mask: Mask2D) -> "Imaging":
        """
        Apply a mask to the imaging dataset, whereby the mask is applied to the image data and noise-map one-by-one.

        The original unmasked imaging data is stored as the `self.unmasked` attribute. This is used to ensure that if
        the `apply_mask` function is called multiple times, every mask is always applied to the original unmasked
        imaging dataset.

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image.
        """
        if self.image.mask.is_all_false:
            unmasked_imaging = self
        else:
            unmasked_imaging = self.unmasked

        image = Array2D.manual_mask(
            array=unmasked_imaging.image.native, mask=mask.mask_sub_1
        )

        noise_map = Array2D.manual_mask(
            array=unmasked_imaging.noise_map.native, mask=mask.mask_sub_1
        )

        imaging = Imaging(
            image=image,
            noise_map=noise_map,
            psf=self.psf_unormalized,
            settings=self.settings,
            name=self.name,
            pad_for_convolver=True,
        )

        imaging.unmasked = unmasked_imaging

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return imaging

    def apply_settings(self, settings: SettingsImaging) -> "Imaging":
        """
        Returns a new instance of the imaging with the input `SettingsImaging` applied to them.

        This can be used to update settings like the types of grids associated with the dataset that are used
        to perform calculations or putting a limit of the dataset's signal-to-noise.

        Parameters
        ----------
        settings
            The settings for the imaging data that control things like the grids used for calculations.
        """
        return Imaging(
            image=self.image,
            noise_map=self.noise_map,
            psf=self.psf_unormalized,
            settings=settings,
            name=self.name,
            pad_for_convolver=self.pad_for_convolver,
        )

    def signal_to_noise_limited_from(self, signal_to_noise_limit, mask=None):

        imaging = copy.deepcopy(self)

        if mask is None:
            mask = Mask2D.unmasked(
                shape_native=self.shape_native, pixel_scales=self.pixel_scales
            )

        noise_map_limit = np.where(
            (self.signal_to_noise_map.native > signal_to_noise_limit) & (mask == False),
            np.abs(self.image.native) / signal_to_noise_limit,
            self.noise_map.native,
        )

        imaging.noise_map = Array2D.manual_mask(
            array=noise_map_limit, mask=self.image.mask
        )

        return imaging

    def output_to_fits(
        self, image_path, psf_path=None, noise_map_path=None, overwrite=False
    ):
        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)

        if self.psf is not None and psf_path is not None:
            self.psf.output_to_fits(file_path=psf_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
