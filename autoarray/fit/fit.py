import numpy as np

from autoarray.util import fit_util


class DataFit(object):

    # noinspection PyUnresolvedReferences
    def __init__(self, mask, data, noise_map, model_data):
        """Class to fit data where the data structures are any dimension.

        Parameters
        -----------
        data : ndarray
            The observed data that is fitted.
        noise_map : ndarray
            The noise_map-map of the observed data.
        mask: msk.Mask
            The masks that is applied to the data.
        model_data : ndarray
            The model data the fitting image is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (datas - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((datas - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the data, summed over every data-point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to data (chi_squared / number of datas points), summed over \
            every data-point.
        noise_normalization : float
            The overall normalization term of the noise_map-map, summed over every data-point.
        likelihood : float
            The overall likelihood of the model's fit to the data, summed over evey data-point.
        """
        self.mask = mask
        self.data = data
        self.noise_map = noise_map
        self.model_data = model_data

        residual_map = fit_util.residual_map_from_data_and_model_data(
            data=data, model_data=model_data)

        self.residual_map = mask.scaled_array_from_array_1d(array_1d=residual_map)

        chi_squared_map = fit_util.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map)

        self.chi_squared_map = mask.scaled_array_from_array_1d(array_1d=chi_squared_map)

        self.chi_squared = fit_util.chi_squared_from_chi_squared_map(
            chi_squared_map=self.chi_squared_map)

        self.reduced_chi_squared = self.chi_squared / int(
            np.size(self.mask) - np.sum(self.mask))

        self.noise_normalization = fit_util.noise_normalization_from_noise_map(
            noise_map=self.noise_map)

        self.likelihood = fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization)

    @property
    def normalized_residual_map(self):
        normalized_residual_map = fit_util.normalized_residual_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map)
        return self.mask.scaled_array_from_array_1d(array_1d=normalized_residual_map)

    @property
    def signal_to_noise_map(self):
        """The signal-to-noise_map of the data and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return self.mask.scaled_array_from_array_1d(array_1d=signal_to_noise_map)