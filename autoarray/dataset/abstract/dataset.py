import copy
import logging
import numpy as np
from typing import Optional, Union
import warnings

from autoconf import cached_property
from autoconf import conf

from autoarray.dataset.abstract.settings import AbstractSettingsDataset
from autoarray.structures.arrays.uniform_1d import Array1D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.vectors.uniform import VectorYX2D
from autoarray.structures.visibilities import Visibilities
from autoarray.structures.visibilities import VisibilitiesNoiseMap
from autoarray.mask.mask_1d import Mask1D
from autoarray.mask.mask_2d import Mask2D

from autoarray import exc

logger = logging.getLogger(__name__)


class AbstractDataset:
    def __init__(
        self,
        data: Union[Array1D, Array2D, VectorYX2D, Visibilities],
        noise_map: Union[Array1D, Array2D, VectorYX2D, VisibilitiesNoiseMap],
        noise_covariance_matrix: Optional[np.ndarray] = None,
        settings: AbstractSettingsDataset = AbstractSettingsDataset(),
    ):
        """
        A collection of abstract data structures for different types of data (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        dataucture
            The array of the image data, in units of electrons per second.
        noise_mapucture
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """

        self.data = data
        self.noise_map = noise_map
        self.settings = settings

        mask = self.mask

        self.noise_covariance_matrix = noise_covariance_matrix

        if noise_map is None and noise_covariance_matrix is not None:

            logger.info(
                """
                No noise map was input into the Imaging class, but a `noise_covariance_matrix` was.

                Using the diagonal of the `noise_covariance_matrix` to create the `noise_map`. 

                This `noise-map` is used only for visualization where it is not appropriate to plot covariance.
                """
            )

            noise_map = Array2D.manual_slim(
                array=np.diag(noise_covariance_matrix),
                shape_native=data.shape_native,
                pixel_scales=data.shape_native,
            )

        elif noise_map is None and noise_covariance_matrix is None:

            raise exc.DatasetException(
                """
                No noise map or noise_covariance_matrix was passed to the Imaging object.
                """
            )

        self.noise_map = noise_map

        if conf.instance["general"]["structures"]["use_dataset_grids"]:

            mask_grid = mask.mask_new_sub_size_from(
                mask=mask, sub_size=settings.sub_size
            )
            self.grid = settings.grid_from(mask=mask_grid)

            mask_inversion = mask.mask_new_sub_size_from(
                mask=mask, sub_size=settings.sub_size_pixelization
            )

            self.grid_pixelization = settings.grid_pixelization_from(
                mask=mask_inversion
            )

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
    def inverse_noise_map(self) -> Union[Array1D, Array2D]:
        return 1.0 / self.noise_map

    @property
    def signal_to_noise_map(self) -> Union[Array1D, Array2D]:
        """
        The estimated signal-to-noise_maps mappers of the image.

        Warnings airse when masked native noise-maps are used, whose masked entries are given values of 0.0. We
        uses the warnings module to surpress these RunTimeWarnings.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self) -> float:
        """
        The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers.
        """
        return np.max(self.signal_to_noise_map)

    @property
    def absolute_signal_to_noise_map(self) -> Union[Array1D, Array2D]:
        """
        The estimated absolute_signal-to-noise_maps mappers of the image.
        """
        return np.divide(np.abs(self.data), self.noise_map)

    @property
    def absolute_signal_to_noise_max(self) -> float:
        """
        The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers.
        """
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self) -> Union[Array1D, Array2D]:
        """
        The potential chi-squared-map of the imaging data_type. This represents how much each pixel can contribute to
        the chi-squared-map, assuming the model fails to fit it at all (e.g. model value = 0.0).
        """
        return np.square(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_max(self) -> float:
        """
        The maximum value of the potential chi-squared-map.
        """
        return np.max(self.potential_chi_squared_map)

    @cached_property
    def noise_covariance_matrix_inv(self) -> np.ndarray:
        """
        Returns the inverse of the noise covariance matrix, which is used when computing a chi-squared which accounts
        for covariance via a fit.

        Returns
        -------
        The inverse of the noise covariance matrix.
        """
        return np.linalg.inv(self.noise_covariance_matrix)

    def trimmed_after_convolution_from(self, kernel_shape) -> "AbstractDataset":

        imaging = copy.copy(self)

        imaging.data = imaging.data.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )
        imaging.noise_map = imaging.noise_map.trimmed_after_convolution_from(
            kernel_shape=kernel_shape
        )

        return imaging
