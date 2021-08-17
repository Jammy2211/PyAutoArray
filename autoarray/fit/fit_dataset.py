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

        self.data = self.fit.data
        self.noise_map = self.fit.noise_map
        self.model_data = self.fit.model_data
        self.residual_map = self.fit.residual_map
        self.normalized_residual_map = self.fit.normalized_residual_map
        self.chi_squared_map = self.fit.chi_squared_map
        self.signal_to_noise_map = self.fit.signal_to_noise_map
        self.chi_squared = self.fit.chi_squared
        self.noise_normalization = self.fit.noise_normalization
        self.log_likelihood = self.fit.log_likelihood
        self.log_evidence = self.fit.log_evidence
        self.figure_of_merit = self.fit.figure_of_merit

        self.log_likelihood_with_regularization = (
            self.fit.log_likelihood_with_regularization
        )
        self.reduced_chi_squared = self.fit.reduced_chi_squared

        self.inversion = self.fit.inversion
        self.total_inversions = self.fit.total_inversions

    @property
    def mask(self):
        raise NotImplementedError


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
    def mask(self):
        return self.fit.mask

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
