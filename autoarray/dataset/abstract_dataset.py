import pickle

import numpy as np

from autoarray.structures import arrays, grids


def grid_from_mask_and_grid_class(
    mask, grid_class, fractional_accuracy, sub_steps, pixel_scales_interp
):

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
        """A collection of abstract 2D for different data_type classes (an image, pixel-scale, noise map, etc.)

        Parameters
        ----------
        data : arrays.Array
            The array of the image data, in units of electrons per second.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An array describing the PSF kernel of the image.
        noise_map : NoiseMap | float | ndarray
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
        """The potential chi-squared map of the imaging data_type. This represents how much each pixel can contribute to \
        the chi-squared map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return arrays.Array(
            array=np.square(self.absolute_signal_to_noise_map), mask=self.data.mask
        )

    @property
    def potential_chi_squared_max(self):
        """The maximum value of the potential chi-squared map"""
        return np.max(self.potential_chi_squared_map)


class AbstractMaskedDataset:
    def __init__(
        self,
        dataset,
        mask,
        grid_class=grids.GridIterate,
        grid_inversion_class=grids.Grid,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):

        if sub_steps is None:
            sub_steps = [2, 4, 8, 16]

        self.dataset = dataset
        self.mask = mask

        self.pixel_scales_interp = pixel_scales_interp

        ### GRIDS ###

        if mask.pixel_scales is not None:

            self.grid = grid_from_mask_and_grid_class(
                mask=mask,
                grid_class=grid_class,
                fractional_accuracy=fractional_accuracy,
                sub_steps=sub_steps,
                pixel_scales_interp=pixel_scales_interp,
            )

            self.grid_inversion = grid_from_mask_and_grid_class(
                mask=mask,
                grid_class=grid_inversion_class,
                fractional_accuracy=fractional_accuracy,
                sub_steps=sub_steps,
                pixel_scales_interp=pixel_scales_interp,
            )

        else:

            self.grid = None
            self.grid_inversion = None

        self.inversion_pixel_limit = inversion_pixel_limit
        self.inversion_uses_border = inversion_uses_border

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def positions(self):
        return self.dataset.positions
