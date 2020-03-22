import numpy as np

from autoarray.dataset import imaging, interferometer
from autoarray.util import fit_util


def fit_masked_dataset(masked_dataset, model_data, inversion=None):
    return fit(
        masked_dataset=masked_dataset, model_data=model_data, inversion=inversion
    )


def fit(masked_dataset, model_data, inversion=None):
    if isinstance(masked_dataset, imaging.MaskedImaging):
        return ImagingFit(
            mask=masked_dataset.mask,
            image=masked_dataset.image,
            noise_map=masked_dataset.noise_map,
            model_image=model_data,
            inversion=inversion,
        )
    elif isinstance(masked_dataset, interferometer.MaskedInterferometer):
        return InterferometerFit(
            visibilities_mask=masked_dataset.visibilities_mask,
            visibilities=masked_dataset.visibilities,
            noise_map=masked_dataset.noise_map,
            model_visibilities=model_data,
            inversion=inversion,
        )


class DatasetFit:

    # noinspection PyUnresolvedReferences
    def __init__(self, mask, data, noise_map, model_data, inversion=None):
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
        self.mask = mask
        self.data = data
        self.noise_map = noise_map
        self.model_data = model_data
        self.inversion = inversion

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


class ImagingFit(DatasetFit):
    def __init__(self, mask, image, noise_map, model_image, inversion=None):
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

        super(ImagingFit, self).__init__(
            mask=mask,
            data=image,
            noise_map=noise_map,
            model_data=model_image,
            inversion=inversion,
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


class InterferometerFit(DatasetFit):
    def __init__(
        self,
        visibilities_mask,
        visibilities,
        noise_map,
        model_visibilities,
        inversion=None,
    ):
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

        super(InterferometerFit, self).__init__(
            mask=visibilities_mask,
            data=visibilities,
            noise_map=noise_map,
            model_data=model_visibilities,
            inversion=inversion,
        )

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    @property
    def visibilities_mask(self):
        return self.mask

    @property
    def visibilities(self):
        return self.data

    @property
    def model_visibilities(self):
        return self.model_data
