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

        Datasets also contains following properties:

        - `grid`: A grids of (y,x) coordinates which align with the image pixels, whereby each coordinate corresponds to
        the centre of an image pixel. This may be used in fits to calculate the model image of the imaging data.

        - `grid_pixelization`: A grid of (y,x) coordinates which align with the pixels of a pixelization. This grid
        is specifically used for pixelizations computed via the `inversion` module, which often use different
        oversampling and sub-size values to the grid above.

        The `over_sampling` and `over_sampling_pixelization` define how over sampling is performed for these grids.

        This is used in the project PyAutoGalaxy to load imaging data of a galaxy and fit it with galaxy light profiles.
        It is used in PyAutoLens to load imaging data of a strong lens and fit it with a lens model.

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

        self.grids = GridsDataset(mask=data.mask, over_sampling=over_sampling)

    @property
    def over_sampling(self):
        return self.grids.over_sampling

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

    def apply_over_sampling(
        self,
        over_sampling: Optional[OverSamplingDataset] = OverSamplingDataset(),
    ) -> "AbstractDataset":
        """
        Apply new over sampling objects to the grid and grid pixelization of the dataset.

        This method is used to change the over sampling of the grid and grid pixelization, for example when the
        user wishes to perform over sampling with a higher sub grid size or with an iterative over sampling strategy.

        The `grid` and grid_pixelization` are cached properties which after use are stored in memory for efficiency.
        This function resets the cached properties so that the new over sampling is used in the grid and grid
        pixelization.

        The `default_galaxy_mode` parameter is used to set up default over sampling for galaxy light profiles in
        the project PyAutoGalaxy. This sets up the over sampling such that there is high over sampling in the centre
        of the mask, where the galaxy is located, and lower over sampling in the outer regions of the mask. It
        does this based on the pixel scale, which gives a good estimate of how large the central region
        requiring over sampling is.

        Parameters
        ----------
        over_sampling
            The over sampling schemes which divide the grids into sub grids of smaller pixels within their host image
            pixels when using the grid to evaluate a function (e.g. images) to better approximate the 2D line integral
            This class controls over sampling for all the different grids (e.g. `grid`, `grid_pixelization).
        """

        uniform = over_sampling.uniform or self.over_sampling.uniform
        non_uniform = over_sampling.non_uniform or self.over_sampling.non_uniform
        pixelization = over_sampling.pixelization or self.over_sampling.pixelization

        over_sampling = OverSamplingDataset(
            uniform=uniform,
            non_uniform=non_uniform,
            pixelization=pixelization,
        )

        self.grids = GridsDataset(mask=self.mask, over_sampling=over_sampling)

        return self
