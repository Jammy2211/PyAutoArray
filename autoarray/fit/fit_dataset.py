import numpy as np
from typing import Union

from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.fit.fit_data import FitData
from autoarray.fit.fit_data import FitDataComplex


class FitDataset:

    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, fit: Union[FitData, FitDataComplex]):
        """Class to fit a masked dataset where the dataset's data structures are any dimension.

        Parameters
        -----------
        dataset : MaskedDataset
            The masked dataset (data, mask, noise-map, etc.) that is fitted.
        model_data
            The model data the masked dataset is fitted with.
        inversion : Inversion
            If the fit uses an `Inversion` this is the instance of the object used to perform the fit. This determines
            if the `log_likelihood` or `log_evidence` is used as the `figure_of_merit`.
        use_mask_in_fit : bool
            If `True`, masked data points are omitted from the fit. If `False` they are not (in most use cases the
            `dataset` will have been processed to remove masked points, for example the `slim` representation).

        Attributes
        -----------
        residual_map
            The residual-map of the fit (data - model_data).
        chi_squared_map
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

        self.dataset = dataset
        self.fit = fit

    @property
    def data(self):
        return self.fit.data

    @property
    def noise_map(self):
        return self.fit.noise_map

    @property
    def model_data(self):
        return self.fit.model_data

    @property
    def mask(self):
        return self.fit.mask

    @property
    def residual_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """
        return self.fit.residual_map

    @property
    def normalized_residual_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return self.fit.normalized_residual_map

    @property
    def chi_squared_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return self.fit.chi_squared_map

    @property
    def signal_to_noise_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """The signal-to-noise_map of the dataset and noise-map which are fitted."""
        return self.fit.signal_to_noise_map

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return self.fit.chi_squared

    @property
    def reduced_chi_squared(self) -> float:
        return self.fit.reduced_chi_squared

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return self.fit.noise_normalization

    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of each model data point's fit to the dataset, where:

        Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
        """
        return self.fit.log_likelihood

    @property
    def log_likelihood_with_regularization(self) -> float:
        """
        Returns the log likelihood of an inversion's fit to the dataset, including a regularization term which \
        comes from an inversion:

        Log Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)
        """
        return self.fit.log_likelihood_with_regularization

    @property
    def log_evidence(self) -> float:
        """
        Returns the log evidence of the inversion's fit to a dataset, where the log evidence includes a number of terms
        which quantify the complexity of an inversion's reconstruction (see the `Inversion` module):

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
        return self.fit.log_evidence

    @property
    def figure_of_merit(self) -> float:
        return self.fit.figure_of_merit

    @property
    def inversion(self):
        return self.fit.inversion

    @property
    def total_inversions(self) -> int:
        return self.fit.total_inversions


class FitImaging(FitDataset):
    def __init__(self, imaging, fit: FitData):
        """Class to fit a masked imaging dataset.

        Parameters
        -----------
        imaging : MaskedImaging
            The masked imaging dataset that is fitted.
        model_image : Array2D
            The model image the masked imaging is fitted with.
        inversion : Inversion
            If the fit uses an `Inversion` this is the instance of the object used to perform the fit. This determines
            if the `log_likelihood` or `log_evidence` is used as the `figure_of_merit`.
        use_mask_in_fit : bool
            If `True`, masked data points are omitted from the fit. If `False` they are not (in most use cases the
            `dataset` will have been processed to remove masked points, for example the `slim` representation).

        Attributes
        -----------
        residual_map
            The residual-map of the fit (data - model_data).
        chi_squared_map
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

        super().__init__(dataset=imaging, fit=fit)

    @property
    def imaging(self):
        return self.dataset

    @property
    def image(self) -> Union[np.ndarray, Array1D, Array2D]:
        return self.fit.data

    @property
    def model_image(self) -> Union[np.ndarray, Array1D, Array2D]:
        return self.fit.model_data


class FitInterferometer(FitDataset):
    def __init__(self, interferometer, fit: FitDataComplex):
        """Class to fit a masked interferometer dataset.

        Parameters
        -----------
        interferometer : MaskedInterferometer
            The masked interferometer dataset that is fitted.
        model_visibilities : Visibilities
            The model visibilities the masked imaging is fitted with.
        inversion : Inversion
            If the fit uses an `Inversion` this is the instance of the object used to perform the fit. This determines
            if the `log_likelihood` or `log_evidence` is used as the `figure_of_merit`.
        use_mask_in_fit : bool
            If `True`, masked data points are omitted from the fit. If `False` they are not (in most use cases the
            `dataset` will have been processed to remove masked points, for example the `slim` representation).

        Attributes
        -----------
        residual_map
            The residual-map of the fit (data - model_data).
        chi_squared_map
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

        super().__init__(dataset=interferometer, fit=fit)

    @property
    def mask(self):
        return np.full(shape=self.visibilities.shape, fill_value=False)

    @property
    def interferometer(self):
        return self.dataset

    @property
    def transformer(self):
        return self.interferometer.transformer

    @property
    def visibilities(self) -> Union[np.ndarray, Array1D, Array2D]:
        return self.fit.data

    @property
    def model_visibilities(self) -> Union[np.ndarray, Array1D, Array2D]:
        return self.fit.model_data

    @property
    def dirty_image(self):
        return self.transformer.image_from_visibilities(visibilities=self.visibilities)

    @property
    def dirty_noise_map(self):
        return self.transformer.image_from_visibilities(visibilities=self.fit.noise_map)

    @property
    def dirty_signal_to_noise_map(self):
        return self.transformer.image_from_visibilities(
            visibilities=self.signal_to_noise_map
        )

    @property
    def dirty_model_image(self):
        return self.transformer.image_from_visibilities(
            visibilities=self.model_visibilities
        )

    @property
    def dirty_residual_map(self):
        return self.transformer.image_from_visibilities(visibilities=self.residual_map)

    @property
    def dirty_normalized_residual_map(self):
        return self.transformer.image_from_visibilities(
            visibilities=self.normalized_residual_map
        )

    @property
    def dirty_chi_squared_map(self):
        return self.transformer.image_from_visibilities(
            visibilities=self.chi_squared_map
        )
