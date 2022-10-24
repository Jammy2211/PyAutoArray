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


class ImagingCovariance(AbstractImaging):
    def __init__(
        self,
        image: Array2D,
        noise_covariance_matrix: np.ndarray,
        psf: Kernel2D = None,
        settings=SettingsImaging(),
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

        super().__init__(
            image=image,
            noise_map=None,
            psf=psf,
            settings=settings,
            pad_for_convolver=pad_for_convolver,
        )
