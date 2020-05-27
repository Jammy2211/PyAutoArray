import pickle

import numpy as np

from autoarray.structures import arrays, grids


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
        return self.mapping.array_stored_1d_from_array_1d(
            array_1d=np.divide(np.abs(self.data), self.noise_map)
        )

    @property
    def absolute_signal_to_noise_max(self):
        """The maximum value of absolute signal-to-noise_map in an image pixel in the image's signal-to-noise_maps mappers"""
        return np.max(self.absolute_signal_to_noise_map)

    @property
    def potential_chi_squared_map(self):
        """The potential chi-squared map of the imaging data_type. This represents how much each pixel can contribute to \
        the chi-squared map, assuming the model fails to fit it at all (e.g. model value = 0.0)."""
        return self.mapping.array_stored_1d_from_array_1d(
            array_1d=np.square(self.absolute_signal_to_noise_map)
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
        grid_class=grids.GridIterator,
        grid_inversion_class=grids.Grid,
        grid_fractional_accuracy=0.9999,
        grid_sub_steps=[2, 4, 8, 16],
        grid_interpolate_pixel_scale=None,
        inversion_pixel_limit=None,
        inversion_uses_border=True,
    ):

        self.dataset = dataset
        self.mask = mask

        self.grid_interpolation_pixel_scale = grid_interpolate_pixel_scale

        ### GRIDS ###

        if mask.pixel_scales is not None:

            if grid_class is grids.Grid:

                self.grid = grids.Grid.from_mask(mask=mask)

                if grid_interpolate_pixel_scale is not None:

                    self.grid = self.grid.new_grid_with_interpolator(
                        pixel_scale_interpolation_grid=grid_interpolate_pixel_scale
                    )

            elif grid_class is grids.GridIterator:

                self.grid = grids.GridIterator.from_mask(
                    mask=mask,
                    fractional_accuracy=grid_fractional_accuracy,
                    sub_steps=grid_sub_steps,
                )

            if grid_inversion_class is grids.Grid:

                self.grid_inversion = grids.Grid.from_mask(mask=mask)

                if grid_interpolate_pixel_scale is not None:

                    self.grid_inversion = self.grid_inversion.new_grid_with_interpolator(
                        pixel_scale_interpolation_grid=grid_interpolate_pixel_scale
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
