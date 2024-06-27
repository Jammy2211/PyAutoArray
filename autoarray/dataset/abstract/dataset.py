import copy
import logging
import numpy as np
import warnings
from typing import Optional, Union

from autoarray.dataset.over_sampling import OverSamplingDataset
from autoarray.dataset.grids import GridsDataset

from autoarray import exc
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.abstract_structure import Structure
from autoarray.structures.arrays.uniform_2d import Array2D
from autoconf import cached_property


logger = logging.getLogger(__name__)


class AbstractDataset:
    def __init__(
        self,
        data: Structure,
        noise_map: Structure,
        noise_covariance_matrix: Optional[np.ndarray] = None,
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
    ):
        """
        An abstract dataset, containing the image data, noise-map, PSF and associated quantities for calculations
        like the grid.

        This object is extended with specific dataset types, such as an `Imaging` or `Interferometer` dataset,
        and can be used for fitting data with model data and quantifying the goodness-of-fit.

        The following quantities are abstract and used by any dataset type:

        - `data`: The image data, which shows the signal that is analysed and fitted with a model image.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        The dataset also has a number of (y,x) grids of coordinates associated with it, which map to the centres
        of its image pixels. They are used for performing calculations which map directly to the data and have
        over sampling calculations built in which approximate the 2D line integral of these calculations within a
        pixel. This is explained in more detail in the `GridsDataset` class.

        Parameters
        ----------
        data
            The array of the image data containing the signal that is fitted (in PyAutoGalaxy and PyAutoLens the
            recommended units are electrons per second).
        noise_map
            An array describing the RMS standard deviation error in each pixel used for computing quantities like the
            chi-squared in a fit (in PyAutoGalaxy and PyAutoLens the recommended units are electrons per second).
        noise_covariance_matrix
            A noise-map covariance matrix representing the covariance between noise in every `data` value, which
            can be used via a bespoke fit to account for correlated noise in the data.
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grid_pixelization).
        """

        self.data = data

        self.noise_covariance_matrix = noise_covariance_matrix

        if noise_map is None:
            try:
                noise_map = Array2D.no_mask(
                    values=np.diag(noise_covariance_matrix),
                    shape_native=data.shape_native,
                    pixel_scales=data.shape_native,
                )

                logger.info(
                    """
                    No noise map was input into the Imaging class, but a `noise_covariance_matrix` was.
    
                    Using the diagonal of the `noise_covariance_matrix` to create the `noise_map`. 
    
                    This `noise-map` is used only for visualization where it is not appropriate to plot covariance.
                    """
                )

            except ValueError as e:
                raise exc.DatasetException(
                    """
                    No noise map or noise_covariance_matrix was passed to the Imaging object.
                    """
                ) from e

        self.noise_map = noise_map

        self.over_sampling = over_sampling

    @property
    def grid(self):
        return self.grids.uniform

    @cached_property
    def grids(self):
        return GridsDataset(mask=self.data.mask, over_sampling=self.over_sampling)

    @property
    def shape_native(self):
        return self.mask.shape_native

    @property
    def shape_slim(self):
        return self.data.shape_slim

    @property
    def pixel_scales(self):
        return self.mask.pixel_scales

    @property
    def mask(self) -> Union[Mask1D, Mask2D]:
        return self.data.mask

    @property
    def signal_to_noise_map(self) -> Structure:
        """
        The estimated signal-to-noise_maps mappers of the image.

        Warnings arise when masked native noise-maps are used, whose masked entries are given values of 0.0. We
        use the warnings module to suppress these RunTimeWarnings.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = self.data / self.noise_map
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self) -> float:
        """
        The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers.
        """
        return np.max(self.signal_to_noise_map)

    @cached_property
    def noise_covariance_matrix_inv(self) -> np.ndarray:
        """
        Returns the inverse of the noise covariance matrix, which is used when computing a chi-squared which accounts
        for covariance via a fit.
        """
        return np.linalg.inv(self.noise_covariance_matrix)

    def trimmed_after_convolution_from(self, kernel_shape) -> "AbstractDataset":
        dataset = copy.copy(self)

        dataset.data = dataset.data.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )
        dataset.noise_map = dataset.noise_map.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )

        return dataset
