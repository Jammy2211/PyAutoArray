from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import Union

from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D

from autoarray.fit import fit_util


class AbstractFit(ABC):
    def __init__(
        self,
        data,
        noise_map,
        model_data,
        mask=None,
        inversion=None,
        use_mask_in_fit=False,
    ):

        self.data = data
        self.noise_map = noise_map
        self.model_data = model_data
        self._mask = mask
        self.inversion = inversion
        self.use_mask_in_fit = use_mask_in_fit

    @property
    @abstractmethod
    def mask(self):
        pass

    @property
    @abstractmethod
    def signal_to_noise_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted.
        """

    @property
    def residual_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """

        if self.use_mask_in_fit:
            return fit_util.residual_map_with_mask_from(
                data=self.data, model_data=self.model_data, mask=self.mask
            )
        return fit_util.residual_map_from(data=self.data, model_data=self.model_data)

    @property
    @abstractmethod
    def normalized_residual_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """

    @property
    @abstractmethod
    def chi_squared_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """

    @property
    @abstractmethod
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """

    @property
    @abstractmethod
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """

    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of each model data point's fit to the dataset, where:

        Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
        """
        return fit_util.log_likelihood_from(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )

    @property
    def log_likelihood_with_regularization(self) -> float:
        """
        Returns the log likelihood of an inversion's fit to the dataset, including a regularization term which \
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
        if self.inversion is not None:
            return fit_util.log_evidence_from(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def figure_of_merit(self) -> float:
        if self.inversion is None:
            return self.log_likelihood
        return self.log_evidence

    @property
    def total_inversions(self) -> int:
        if self.inversion is None:
            return 0
        return 1

    @property
    def reduced_chi_squared(self) -> float:
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))


class FitData(AbstractFit):
    @property
    def mask(self):
        return self._mask

    @property
    def signal_to_noise_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """The signal-to-noise_map of the dataset and noise-map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def normalized_residual_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        if self.use_mask_in_fit:
            return fit_util.normalized_residual_map_with_mask_from(
                residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
            )
        return fit_util.normalized_residual_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        if self.use_mask_in_fit:
            return fit_util.chi_squared_map_with_mask_from(
                residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
            )
        return fit_util.chi_squared_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        if self.use_mask_in_fit:
            return fit_util.chi_squared_with_mask_from(
                chi_squared_map=self.chi_squared_map, mask=self.mask
            )
        return fit_util.chi_squared_from(chi_squared_map=self.chi_squared_map)

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        if self.use_mask_in_fit:
            return fit_util.noise_normalization_with_mask_from(
                noise_map=self.noise_map, mask=self.mask
            )
        return fit_util.noise_normalization_from(noise_map=self.noise_map)


class FitDataComplex(AbstractFit):
    @property
    def mask(self):
        return np.full(shape=self.data.shape, fill_value=False)

    @property
    def normalized_residual_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return fit_util.normalized_residual_map_complex_with_mask_from(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
        )

    @property
    def chi_squared_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return fit_util.chi_squared_map_complex_with_mask_from(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
        )

    @property
    def signal_to_noise_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """The signal-to-noise_map of the dataset and noise-map which are fitted."""
        signal_to_noise_map_real = np.divide(
            np.real(self.data), np.real(self.noise_map)
        )
        signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
        signal_to_noise_map_imag = np.divide(
            np.imag(self.data), np.imag(self.noise_map)
        )
        signal_to_noise_map_imag[signal_to_noise_map_imag < 0] = 0.0

        return signal_to_noise_map_real + 1.0j * signal_to_noise_map_imag

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return fit_util.chi_squared_complex_with_mask_from(
            chi_squared_map=self.chi_squared_map, mask=self.mask
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return fit_util.noise_normalization_complex_with_mask_from(
            noise_map=self.noise_map, mask=self.mask
        )
