import pickle

import numpy as np
import copy

from autoarray.structures import arrays, grids


def grid_from_mask_and_grid_class(
    mask, grid_class, fractional_accuracy, sub_steps, pixel_scales_interp
):
    if mask.pixel_scales is None:
        return None

    if grid_class is grids.Grid:

        return grids.Grid.from_mask(mask=mask)

    elif grid_class is grids.GridIterate:

        return grids.GridIterate.from_mask(
            mask=mask, fractional_accuracy=fractional_accuracy, sub_steps=sub_steps
        )

    elif grid_class is grids.GridInterpolate:

        return grids.GridInterpolate.from_mask(
            mask=mask, pixel_scales_interp=pixel_scales_interp
        )


class AbstractDataset:
    def __init__(self, data, noise_map, positions=None, name=None):
        """A collection of abstract 2D for different data_type classes (an image, pixel-scale, noise-map, etc.)

        Parameters
        ----------
        data : arrays.Array
            The array of the image data, in units of electrons per second.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the image.
        noise_map : ndarray
            An array describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        self.data = data
        self.noise_map = noise_map
        self.positions = positions
        self._name = name if name is not None else "dataset"

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def load(cls, filename) -> "AbstractDataset":
        """
        Load the dataset at the specified filename

        Parameters
        ----------
        filename
            The filename containing the dataset

        Returns
        -------
        The dataset
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def mapping(self):
        return self.data.mask.mapping

    @property
    def geometry(self):
        return self.data.mask.geometry

    @property
    def inverse_noise_map(self):
        return 1.0 / self.noise_map

    @property
    def signal_to_noise_map(self):
        """The estimated signal-to-noise_maps mappers of the image."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def signal_to_noise_max(self):
        """The maximum value of signal-to-noise_maps in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.signal_to_noise_map)

    @property
    def absolute_signal_to_noise_map(self):
        """The estimated absolute_signal-to-noise_maps mappers of the image."""
        return arrays.Array(
            array=np.divide(np.abs(self.data), self.noise_map), mask=self.data.mask
        )

    @property
    def absolute_signal_to_noise_max(self):
        """The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self):
        """The potential chi-squared-map of the imaging data_type. This represents how much each pixel can contribute to \
        the chi-squared-map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return arrays.Array(
            array=np.square(self.absolute_signal_to_noise_map), mask=self.data.mask
        )

    @property
    def potential_chi_squared_max(self):
        """The maximum value of the potential chi-squared-map"""
        return np.max(self.potential_chi_squared_map)

    def modify_noise_map(self, noise_map):

        imaging = copy.deepcopy(self)

        imaging.noise_map = noise_map

        return imaging


class AbstractMaskedDatasetSettings:
    def __init__(
        self,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise-map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        pixel_scales_interp : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        self.grid_class = grid_class
        self.grid_inversion_class = grid_inversion_class
        self.fractional_accuracy = fractional_accuracy

        if sub_steps is None:
            sub_steps = [2, 4, 8, 16]

        self.sub_steps = sub_steps
        self.pixel_scales_interp = pixel_scales_interp

    def grid_from_mask(self, mask):
        return grid_from_mask_and_grid_class(
            mask=mask,
            grid_class=self.grid_class,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            pixel_scales_interp=self.pixel_scales_interp,
        )

    def grid_inversion_from_mask(self, mask):
        return grid_from_mask_and_grid_class(
            mask=mask,
            grid_class=self.grid_inversion_class,
            fractional_accuracy=self.fractional_accuracy,
            sub_steps=self.sub_steps,
            pixel_scales_interp=self.pixel_scales_interp,
        )


class AbstractMaskedDataset:
    def __init__(self, dataset, mask, settings=AbstractMaskedDatasetSettings()):

        self.dataset = dataset
        self.mask = mask
        self.settings = settings
        self.grid = settings.grid_from_mask(mask=mask)
        self.grid_inversion = settings.grid_inversion_from_mask(mask=mask)

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def positions(self):
        return self.dataset.positions

    def modify_noise_map(self, noise_map):

        masked_imaging = copy.deepcopy(self)

        masked_imaging.noise_map = noise_map

        return masked_imaging
