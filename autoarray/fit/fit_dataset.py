from abc import ABC
from abc import abstractmethod
import numpy as np
from typing import Dict, Optional, Union

from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D

from autoarray.fit import fit_util
from autoarray.numba_util import profile_func


class FitDataset(ABC):

    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        dataset,
        use_mask_in_fit: bool = False,
        profiling_dict: Optional[Dict] = None,
    ):
        """Class to fit a masked dataset where the dataset's data structures are any dimension.

        Parameters
        -----------
        dataset : MaskedDataset
            The masked dataset (data, mask, noise-map, etc.) that is fitted.
        model_data
            The model data the masked dataset is fitted with.
        inversion : LEq
            If the fit uses an `LEq` this is the instance of the object used to perform the fit. This determines
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
        chi_squared
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of data points), summed over \
            every data point.
        noise_normalization
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood
            The overall log likelihood of the model's fit to the dataset, summed over evey data point.
        """
        self.profiling_dict = profiling_dict

        self.dataset = dataset
        self.use_mask_in_fit = use_mask_in_fit

    @property
    @abstractmethod
    def mask(self):
        """
        Overwrite this method so it returns the mask of the dataset which is fitted to the input data.
        """

    @property
    @abstractmethod
    def inversion(self):
        """
        Overwrite this method so it returns the inversion used to fit the dataset.
        """

    @property
    def data(self):
        return self.dataset.data

    @property
    def noise_map(self):
        return self.dataset.noise_map

    @property
    @abstractmethod
    def model_data(self):
        """
        Overwrite this method so it returns the model-data which is fitted to the input data.
        """

    @property
    @abstractmethod
    def signal_to_noise_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted.
        """

    @property
    @abstractmethod
    def potential_chi_squared_map(self) -> Union[np.ndarray, Array1D, Array2D]:
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
        which quantify the complexity of an inversion's reconstruction (see the `LEq` module):

        Log Evidence = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                           Log(Regularization_Matrix_Term) + Noise_Term]

        Parameters
        ----------
        chi_squared
            The chi-squared term of the inversion's fit to the data.
        regularization_term
            The regularization term of the inversion, which is the sum of the difference between reconstructed \
            flux of every pixel multiplied by the regularization coefficient.
        log_curvature_regularization_term
            The log of the determinant of the sum of the curvature and regularization matrices.
        log_regularization_term
            The log of the determinant o the regularization matrix.
        noise_normalization
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
    @profile_func
    def figure_of_merit(self) -> float:
        if self.inversion is None:
            return self.log_likelihood
        return self.log_evidence

    @property
    def total_mappers(self) -> int:
        if self.inversion is None:
            return 0
        return self.inversion.total_mappers

    @property
    def reduced_chi_squared(self) -> float:
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))
