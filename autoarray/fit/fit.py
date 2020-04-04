import numpy as np

from autoarray.util import fit_util


class FitDataset:

    # noinspection PyUnresolvedReferences
    def __init__(self, masked_dataset, model_data, inversion=None):
        """Class to fit data where the dataset structures are any dimension.

        Parameters
        -----------
        data : ndarray
            The observed dataset that is fitted.
        noise_map : ndarray
            The noise_map of the observed dataset.
        mask: msk.Mask
            The mask that is applied to the dataset.
        model_data : ndarray
            The model data the data is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (datas - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((datas - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every simulator-point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every simulator-point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every simulator-point.
        likelihood : float
            The overall likelihood of the model's fit to the dataset, summed over evey simulator-point.
        """

        self.masked_dataset = masked_dataset
        self.model_data = model_data
        self.inversion = inversion

    @property
    def name(self):
        return self.masked_dataset.dataset.name

    @property
    def mask(self):
        return self.masked_dataset.mask

    @property
    def data(self):
        return self.masked_dataset.data

    @property
    def noise_map(self):
        return self.masked_dataset.noise_map

    @property
    def residual_map(self):
        return fit_util.residual_map_from_data_and_model_data(
            data=self.data, model_data=self.model_data
        )

    @property
    def normalized_residual_map(self):
        return fit_util.normalized_residual_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self):
        return fit_util.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def signal_to_noise_map(self):
        """The signal-to-noise_map of the dataset and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def chi_squared(self):
        return fit_util.chi_squared_from_chi_squared_map(
            chi_squared_map=self.chi_squared_map
        )

    @property
    def reduced_chi_squared(self):
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))

    @property
    def noise_normalization(self):
        return fit_util.noise_normalization_from_noise_map(noise_map=self.noise_map)

    @property
    def likelihood(self):
        return fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )

    @property
    def likelihood_with_regularization(self):
        if self.inversion is not None:
            return fit_util.likelihood_with_regularization_from_inversion_terms(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def evidence(self):
        if self.inversion is not None:
            return fit_util.evidence_from_inversion_terms(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def figure_of_merit(self):
        if self.inversion is None:
            return self.likelihood
        else:
            return self.evidence

    @property
    def total_inversions(self):
        if self.inversion is None:
            return 0
        else:
            return 1


class FitImaging(FitDataset):
    def __init__(self, masked_imaging, model_image, inversion=None):
        """Class to fit data where the dataset structures are any dimension.

        Parameters
        -----------
        simulator : ndarray
            The observed dataset that is fitted.
        noise_map : ndarray
            The noise_map of the observed dataset.
        mask: msk.Mask
            The masks that is applied to the dataset.
        model_data : ndarray
            The model data the data is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (datas - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((datas - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every simulator-point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every simulator-point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every simulator-point.
        likelihood : float
            The overall likelihood of the model's fit to the dataset, summed over evey simulator-point.
        """

        super(FitImaging, self).__init__(
            masked_dataset=masked_imaging, model_data=model_image, inversion=inversion
        )

    @property
    def masked_imaging(self):
        return self.masked_dataset

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data


class FitInterferometer(FitDataset):
    def __init__(self, masked_interferometer, model_visibilities, inversion=None):
        """Class to fit data where the dataset structures are any dimension.

        Parameters
        -----------
        simulator : ndarray
            The observed dataset that is fitted.
        noise_map : ndarray
            The noise_map of the observed dataset.
        visibilities_mask: msk.Mask
            The masks that is applied to the dataset.
        model_data : ndarray
            The model data the data is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (datas - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((datas - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every simulator-point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every simulator-point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every simulator-point.
        likelihood : float
            The overall likelihood of the model's fit to the dataset, summed over evey simulator-point.
        """

        super(FitInterferometer, self).__init__(
            masked_dataset=masked_interferometer,
            model_data=model_visibilities,
            inversion=inversion,
        )

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    @property
    def mask(self):
        return self.masked_interferometer.visibilities_mask

    @property
    def visibilities_mask(self):
        return self.masked_interferometer.visibilities_mask

    @property
    def visibilities(self):
        return self.data

    @property
    def model_visibilities(self):
        return self.model_data
