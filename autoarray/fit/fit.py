import numpy as np

from autoarray.util import fit_util


class FitDataset:

    # noinspection PyUnresolvedReferences
    def __init__(self, masked_dataset, model_data, inversion=None):
        """Class to fit a masked dataset where the dataset's data structures are any dimension.

        Parameters
        -----------
        masked_dataset : MaskedDataset
            The masked dataset (data, mask, noise-map, etc.) that is fitted.
        model_data : ndarray
            The model data the masked dataset is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual-map of the fit (data - model_data).
        chi_squared_map : ndarray
            The chi-squared-map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of data points), summed over \
            every data point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood : float
            The overall log likelihood of the model's fit to the dataset, summed over evey data point.
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
        """Compute the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """
        return fit_util.residual_map_from(data=self.data, model_data=self.model_data)

    @property
    def normalized_residual_map(self):
        """Compute the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return fit_util.normalized_residual_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self):
        """Compute the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return fit_util.chi_squared_map_from(
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
        """Compute the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return fit_util.chi_squared_from(chi_squared_map=self.chi_squared_map)

    @property
    def reduced_chi_squared(self):
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))

    @property
    def noise_normalization(self):
        """Compute the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return fit_util.noise_normalization_from(noise_map=self.noise_map)

    @property
    def log_likelihood(self):
        """Compute the log likelihood of each model data point's fit to the dataset, where:

        Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
        """
        return fit_util.log_likelihood_from(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )

    @property
    def log_likelihood_with_regularization(self):
        """Compute the log likelihood of an inversion's fit to the dataset, including a regularization term which \
        comes from an inversion:

        Log Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)
        """
        if self.inversion is not None:
            return fit_util.log_likelihood_with_regularization_from(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def log_evidence(self):
        """Compute the log evidence of the inversion's fit to a dataset, where the log evidence includes a number of terms
        which quantify the complexity of an inversion's reconstruction (see the *inversion* module):

        Log Evidence = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                           Log(Regularization_Matrix_Term) + Noise_Term]

        Parameters
        ----------
        chi_squared : float
            The chi-squared term of the inversion's fit to the data.
        regularization_term : float
            The regularization term of the inversion, which is the sum of the difference between reconstructed \
            flux of every pixel multiplied by the regularization coefficient.
        log_curvature_regularization_term : float
            The log of the determinant of the sum of the curvature and regularization matrices.
        log_regularization_term : float
            The log of the determinant o the regularization matrix.
        noise_normalization : float
            The normalization noise_map-term for the data's noise-map.
        """
        if self.inversion is not None:
            return fit_util.log_evidence_from(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def figure_of_merit(self):
        if self.inversion is None:
            return self.log_likelihood
        else:
            return self.log_evidence

    @property
    def total_inversions(self):
        if self.inversion is None:
            return 0
        else:
            return 1


class FitImaging(FitDataset):
    def __init__(self, masked_imaging, model_image, inversion=None):
        """Class to fit a masked imaging dataset.

        Parameters
        -----------
        masked_imaging : MaskedImaging
            The masked imaging dataset that is fitted.
        model_image : Array
            The model image the masked imaging is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual-map of the fit (data - model_data).
        chi_squared_map : ndarray
            The chi-squared-map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of data points), summed over \
            every data point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood : float
            The overall log likelihood of the model's fit to the dataset, summed over evey data point.
        """

        super().__init__(
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
        """Class to fit a masked interferometer dataset.

        Parameters
        -----------
        masked_interferometer : MaskedInterferometer
            The masked interferometer dataset that is fitted.
        model_visibilities : Visibilities
            The model visibilities the masked imaging is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual-map of the fit (data - model_data).
        chi_squared_map : ndarray
            The chi-squared-map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of data points), summed over \
            every data point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood : float
            The overall log likelihood of the model's fit to the dataset, summed over evey data point.
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
