import numpy as np
from typing import Dict, Optional, Union
import warnings

from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.fit.fit_dataset import FitDataset

from autoarray.fit import fit_util


class FitImaging(FitDataset):
    def __init__(
        self,
        dataset,
        use_mask_in_fit: bool = False,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        Class to fit a masked imaging dataset.

        Parameters
        -----------
        dataset : MaskedImaging
            The masked imaging dataset that is fitted.
        model_image : Array2D
            The model image the masked imaging is fitted with.
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

        super().__init__(
            dataset=dataset,
            use_mask_in_fit=use_mask_in_fit,
            profiling_dict=profiling_dict,
        )

    @property
    def imaging(self):
        return self.dataset

    @property
    def image(self) -> Union[np.ndarray, Array1D, Array2D]:
        return self.data

    @property
    def model_image(self) -> Union[np.ndarray, Array1D, Array2D]:
        return self.model_data

    @property
    def mask(self):
        return self.imaging.mask

    @property
    def signal_to_noise_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def potential_chi_squared_map(self) -> Union[np.ndarray, Array1D, Array2D]:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted.
        """
        warnings.filterwarnings("ignore")
        absolute_signal_to_noise_map = np.divide(np.abs(self.data), self.noise_map)
        return np.square(absolute_signal_to_noise_map)

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

    @property
    def blurred_image(self):
        raise NotImplementedError