import numpy as np


def residual_map_with_mask_from(*, data, mask, model_data):
    """Compute the residual-map between a masked dataset and model data, where:

    Residuals = (Data - Model_Data).
    
    The residual-map values in masked pixels are returned as zero.

    Parameters
    -----------
    data : np.ndarray
        The data that is fitted.
    mask : np.ndarray
        The mask applied to the dataset, where *False* entries are included in the calculation.
    model_data : np.ndarray
        The model data used to fit the data.
    """
    return np.subtract(
        data, model_data, out=np.zeros_like(data), where=np.asarray(mask) == 0
    )


def normalized_residual_map_with_mask_from(*, residual_map, noise_map, mask):
    """Compute the normalized residual-map between a masked dataset and model data, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    The normalized residual-map values in masked pixels are returned as zero.

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the dataset.
    noise_map : np.ndarray
        The noise-map of the dataset.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.divide(
        residual_map,
        noise_map,
        out=np.zeros_like(residual_map),
        where=np.asarray(mask) == 0,
    )


def chi_squared_map_with_mask_from(*, residual_map, noise_map, mask):
    """Computes the chi-squared-map between a masked residual-map and noise-map, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    The chi-squared-map values in masked pixels are returned as zero.

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the dataset.
    noise_map : np.ndarray
        The noise-map of the dataset.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.square(
        np.divide(
            residual_map,
            noise_map,
            out=np.zeros_like(residual_map),
            where=np.asarray(mask) == 0,
        )
    )


def chi_squared_with_mask_from(*, chi_squared_map, mask):
    """Compute the chi-squared terms of each model data's fit to a masked dataset, by summing the masked
    chi-squared-map of the fit.

    The chi-squared values in masked pixels are omitted from the calculation.

    Parameters
    ----------
    chi_squared_map : np.ndarray
        The chi-squared-map of values of the model-simulator fit to the dataset.
    mask : np.ndarray
        The mask applied to the chi-squared-map, where *False* entries are included in the calculation.
    """
    return np.sum(chi_squared_map[np.asarray(mask) == 0])


def noise_normalization_with_mask_from(*, noise_map, mask):
    """Compute the noise-map normalization terms of masked noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    The noise-map values in masked pixels are omitted from the calculation.

    Parameters
    ----------
    noise_map : np.ndarray
        The masked noise-map of the dataset.
    mask : np.ndarray
        The mask applied to the noise-map, where *False* entries are included in the calculation.
    """
    return np.sum(np.log(2 * np.pi * noise_map[np.asarray(mask) == 0] ** 2.0))


def residual_map_from(*, data, model_data):
    """Compute the residual-map between a masked dataset and model data, where:

    Residuals = (Data - Model_Data).

    Parameters
    -----------
    data : np.ndarray
        The data that is fitted.
    mask : np.ndarray
        The mask applied to the dataset, where *False* entries are included in the calculation.
    model_data : np.ndarray
        The model data used to fit the data.
    """
    return np.subtract(data, model_data, out=np.zeros_like(data))


def normalized_residual_map_from(*, residual_map, noise_map):
    """Compute the normalized residual-map between a masked dataset and model data, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the dataset.
    noise_map : np.ndarray
        The noise-map of the dataset.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))


def chi_squared_map_from(*, residual_map, noise_map):
    """Compute the chi-squared-map between a residual-map and noise-map, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the dataset.
    noise_map : np.ndarray
        The noise-map of the dataset.
    """
    return np.square(
        np.divide(residual_map, noise_map, out=np.zeros_like(residual_map))
    )


def chi_squared_from(*, chi_squared_map):
    """Compute the chi-squared terms of a model data's fit to an dataset, by summing the chi-squared-map.

    Parameters
    ----------
    chi_squared_map : np.ndarray
        The chi-squared-map of values of the model-simulator fit to the dataset.
    """
    return np.sum(chi_squared_map)


def noise_normalization_from(*, noise_map):
    """Compute the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : np.ndarray
        The masked noise-map of the dataset.
    """
    return np.sum(np.log(2 * np.pi * noise_map ** 2.0))


def log_likelihood_from(*, chi_squared, noise_normalization):
    """Compute the log likelihood of each model data point's fit to the dataset, where:

    Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term for the model-simulator fit to the dataset.
    noise_normalization : float
        The normalization noise_map-term for the dataset's noise-map.
    """
    return -0.5 * (chi_squared + noise_normalization)


def log_likelihood_with_regularization_from(
    *, chi_squared, regularization_term, noise_normalization
):
    """Compute the log likelihood of an inversion's fit to the dataset, including a regularization term which \
    comes from an inversion:

    Log Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the dataset.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    noise_normalization : float
        The normalization noise_map-term for the dataset's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + noise_normalization)


def log_evidence_from(
    *,
    chi_squared,
    regularization_term,
    log_curvature_regularization_term,
    log_regularization_term,
    noise_normalization,
):
    """Compute the log evidence of an inversion's fit to a dataset, where the log evidence includes a number of terms which
    quantify the complexity of an inversion's reconstruction (see the *inversion* module):

    Log Evidence = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                           Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared : float
        The chi-squared term of the inversion's fit to the dataset.
    regularization_term : float
        The regularization term of the inversion, which is the sum of the difference between reconstructed \
        flux of every pixel multiplied by the regularization coefficient.
    log_curvature_regularization_term : float
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term : float
        The log of the determinant o the regularization matrix.
    noise_normalization : float
        The normalization noise_map-term for the dataset's noise-map.
    """
    return -0.5 * (
        chi_squared
        + regularization_term
        + log_curvature_regularization_term
        - log_regularization_term
        + noise_normalization
    )
