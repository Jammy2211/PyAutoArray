from functools import wraps
import jax.numpy as jnp
import numpy as np

from autoarray.mask.abstract_mask import Mask

from autoarray import type as ty


def to_new_array(func):
    @wraps(func)
    def wrapper(**kwargs):
        result = func(**kwargs)
        try:
            return list(kwargs.values())[0].with_new_array(result)
        except AttributeError:
            return result

    return wrapper


def residual_map_from(*, data: ty.DataLike, model_data: ty.DataLike) -> ty.DataLike:
    """
    Returns the residual-map of the fit of model-data to a masked dataset, where:

    Residuals = (Data - Model_Data).

    Parameters
    ----------
    data
        The data that is fitted.
    mask
        The mask applied to the dataset, where `False` entries are included in the calculation.
    model_data
        The model data used to fit the data.
    """
    return data - model_data


def normalized_residual_map_from(
    *, residual_map: ty.DataLike, noise_map: ty.DataLike
) -> ty.DataLike:
    """
    Returns the normalized residual-map of the fit of model-data to a masked dataset, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_map
        The noise-map of the dataset.
    mask
        The mask applied to the residual-map, where `False` entries are included in the calculation.
    """
    return residual_map / noise_map


def chi_squared_map_from(
    *, residual_map: ty.DataLike, noise_map: ty.DataLike
) -> ty.DataLike:
    """
    Returns the chi-squared-map of the fit of model-data to a masked dataset, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_map
        The noise-map of the dataset.
    """
    return (residual_map / noise_map) ** 2.0


def chi_squared_from(*, chi_squared_map: ty.DataLike) -> float:
    """
    Returns the chi-squared terms of a model data's fit to an dataset, by summing the chi-squared-map.

    Parameters
    ----------
    chi_squared_map
        The chi-squared-map of values of the model-data fit to the dataset.
    """
    return jnp.sum(chi_squared_map)


def noise_normalization_from(*, noise_map: ty.DataLike) -> float:
    """
    Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map
        The masked noise-map of the dataset.
    """
    return jnp.sum(jnp.log(2 * jnp.pi * noise_map**2.0))


def normalized_residual_map_complex_from(
    *, residual_map: jnp.ndarray, noise_map: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns the normalized residual-map of the fit of complex model-data to a dataset, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_map
        The noise-map of the dataset.
    """
    normalized_residual_map_real = (residual_map.real / noise_map.real).astype(
        "complex128"
    )
    normalized_residual_map_imag = (residual_map.imag / noise_map.imag).astype(
        "complex128"
    )

    return normalized_residual_map_real + 1j * normalized_residual_map_imag


def chi_squared_map_complex_from(
    *, residual_map: jnp.ndarray, noise_map: jnp.ndarray
) -> jnp.ndarray:
    """
    Returnss the chi-squared-map of the fit of complex model-data to a dataset, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_map
        The noise-map of the dataset.
    """
    chi_squared_map_real = (residual_map.real / noise_map.real) ** 2
    chi_squared_map_imag = (residual_map.imag / noise_map.imag) ** 2
    return chi_squared_map_real + 1j * chi_squared_map_imag


def chi_squared_complex_from(*, chi_squared_map: jnp.ndarray) -> float:
    """
    Returns the chi-squared terms of each complex model data's fit to a masked dataset, by summing the masked
    chi-squared-map of the fit.

    The chi-squared values in masked pixels are omitted from the calculation.

    Parameters
    ----------
    chi_squared_map
        The chi-squared-map of values of the model-data fit to the dataset.
    """
    chi_squared_real = jnp.sum(chi_squared_map.real)
    chi_squared_imag = jnp.sum(chi_squared_map.imag)
    return chi_squared_real + chi_squared_imag


def noise_normalization_complex_from(*, noise_map: jnp.ndarray) -> float:
    """
    Returns the noise-map normalization terms of a complex noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map
        The masked noise-map of the dataset.
    """
    noise_normalization_real = jnp.sum(jnp.log(2 * jnp.pi * noise_map.real**2.0))
    noise_normalization_imag = jnp.sum(jnp.log(2 * jnp.pi * noise_map.imag**2.0))
    return noise_normalization_real + noise_normalization_imag


@to_new_array
def residual_map_with_mask_from(
    *, data: ty.DataLike, mask: Mask, model_data: ty.DataLike
) -> ty.DataLike:
    """
    Returns the residual-map of the fit of model-data to a masked dataset, where:

    Residuals = (Data - Model_Data).

    The residual-map values in masked pixels are returned as zero.

    Parameters
    ----------
    data
        The data that is fitted.
    mask
        The mask applied to the dataset, where `False` entries are included in the calculation.
    model_data
        The model data used to fit the data.
    """
    return jnp.where(jnp.asarray(mask) == 0, jnp.subtract(data, model_data), 0)


@to_new_array
def normalized_residual_map_with_mask_from(
    *, residual_map: ty.DataLike, noise_map: ty.DataLike, mask: Mask
) -> ty.DataLike:
    """
    Returns the normalized residual-map of the fit of model-data to a masked dataset, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    The normalized residual-map values in masked pixels are returned as zero.

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_map
        The noise-map of the dataset.
    mask
        The mask applied to the residual-map, where `False` entries are included in the calculation.
    """
    return jnp.where(jnp.asarray(mask) == 0, jnp.divide(residual_map, noise_map), 0)


@to_new_array
def chi_squared_map_with_mask_from(
    *, residual_map: ty.DataLike, noise_map: ty.DataLike, mask: Mask
) -> ty.DataLike:
    """
    Returnss the chi-squared-map of the fit of model-data to a masked dataset, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    The chi-squared-map values in masked pixels are returned as zero.

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_map
        The noise-map of the dataset.
    mask
        The mask applied to the residual-map, where `False` entries are included in the calculation.
    """
    return jnp.where(jnp.asarray(mask) == 0, jnp.square(residual_map / noise_map), 0)


def chi_squared_with_mask_from(*, chi_squared_map: ty.DataLike, mask: Mask) -> float:
    """
    Returns the chi-squared terms of each model data's fit to a masked dataset, by summing the masked
    chi-squared-map of the fit.

    The chi-squared values in masked pixels are omitted from the calculation.

    Parameters
    ----------
    chi_squared_map
        The chi-squared-map of values of the model-data fit to the dataset.
    mask
        The mask applied to the chi-squared-map, where `False` entries are included in the calculation.
    """
    return float(jnp.sum(chi_squared_map[jnp.asarray(mask) == 0]))


def chi_squared_with_mask_fast_from(
    *, data: ty.DataLike, mask: Mask, model_data: ty.DataLike, noise_map: ty.DataLike
) -> float:
    """
    Returns the chi-squared terms of each model data's fit to a masked dataset, by summing the masked
    chi-squared-map of the fit.

    The chi-squared values in masked pixels are omitted from the calculation.

    Other methods in `fit_util` compute the `chi-squared` from the `residual_map` and `chi_squared_map`, which requires
    that the `ndarrays` which are used to do this are created and stored in memory. For problems with large datasets
    this can be computationally slow.

    This function computes the `chi_squared` directly from the data, avoiding the need to store the data in memory
    and offering faster tune times.

    Parameters
    ----------
    chi_squared_map
        The chi-squared-map of values of the model-data fit to the dataset.
    mask
        The mask applied to the chi-squared-map, where `False` entries are included in the calculation.
    """
    return float(
        jnp.sum(
            jnp.square(
                jnp.divide(
                    jnp.subtract(
                        data,
                        model_data,
                    )[jnp.asarray(mask) == 0],
                    noise_map[jnp.asarray(mask) == 0],
                )
            )
        )
    )


def noise_normalization_with_mask_from(*, noise_map: ty.DataLike, mask: Mask) -> float:
    """
    Returns the noise-map normalization terms of masked noise-map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    The noise-map values in masked pixels are omitted from the calculation.

    Parameters
    ----------
    noise_map
        The masked noise-map of the dataset.
    mask
        The mask applied to the noise-map, where `False` entries are included in the calculation.
    """
    return float(
        jnp.sum(jnp.log(2 * jnp.pi * noise_map[jnp.asarray(mask) == 0] ** 2.0))
    )


def chi_squared_with_noise_covariance_from(
    *, residual_map: ty.DataLike, noise_covariance_matrix_inv: jnp.ndarray
) -> float:
    """
    Returns the chi-squared value of the fit of model-data to a masked dataset, where
    the noise correlation is described by a covariance matrix of n^2 x n^2 dimensions.

    Chi_Squared = r C^{-1} r, where C^{-1} is the inverse of the covariance matrix

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    noise_covariance_matrix_inv
        The inverse of the noise covariance matrix.
    """

    return residual_map.array @ noise_covariance_matrix_inv @ residual_map.array


def log_likelihood_from(*, chi_squared: float, noise_normalization: float) -> float:
    """
    Returns the log likelihood of each model data point's fit to the dataset, where:

    Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared
        The chi-squared term for the model-data fit to the dataset.
    noise_normalization
        The normalization noise_map-term for the dataset's noise-map.
    """
    return -0.5 * (chi_squared + noise_normalization)


def log_likelihood_with_regularization_from(
    *, chi_squared: float, regularization_term: float, noise_normalization: float
) -> float:
    """
    Returns the log likelihood of an inversion's fit to the dataset, including a regularization term which comes from
    an inversion:

    Log Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term]

    Parameters
    ----------
    chi_squared
        The chi-squared term of the inversion's fit to the dataset.
    regularization_term
        The regularization term of the inversion, which is the sum of the difference between reconstructed
        flux of every pixel multiplied by the regularization coefficient.
    noise_normalization
        The normalization noise_map-term for the dataset's noise-map.
    """
    return -0.5 * (chi_squared + regularization_term + noise_normalization)


def log_evidence_from(
    *,
    chi_squared: float,
    regularization_term: float,
    log_curvature_regularization_term: float,
    log_regularization_term: float,
    noise_normalization: float,
) -> float:
    """
    Returns the log evidence of an inversion's fit to a dataset, where the log evidence includes a number of terms which
    quantify the complexity of an inversion's reconstruction (see the `Inversion` module):

    Log Evidence = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                           Log(Regularization_Matrix_Term) + Noise_Term]

    Parameters
    ----------
    chi_squared
        The chi-squared term of the inversion's fit to the dataset.
    regularization_term
        The regularization term of the inversion, which is the sum of the difference between reconstructed
        flux of every pixel multiplied by the regularization coefficient.
    log_curvature_regularization_term
        The log of the determinant of the sum of the curvature and regularization matrices.
    log_regularization_term
        The log of the determinant o the regularization matrix.
    noise_normalization
        The normalization noise_map-term for the dataset's noise-map.
    """
    return -0.5 * (
        chi_squared
        + regularization_term
        + log_curvature_regularization_term
        - log_regularization_term
        + noise_normalization
    )


def residual_flux_fraction_map_from(
    *, residual_map: jnp.ndarray, data: jnp.ndarray
) -> jnp.ndarray:
    """
    Returns the residual flux fraction map of the fit of model-data to a masked dataset, where:

    Residual_Flux_Fraction = Residuals / Data = (Data - Model)/Data

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    data
        The data of the dataset.
    """
    return jnp.where(data != 0, residual_map / data, 0)


def residual_flux_fraction_map_with_mask_from(
    *, residual_map: jnp.ndarray, data: jnp.ndarray, mask: Mask
) -> jnp.ndarray:
    """
    Returnss the residual flux fraction map of the fit of model-data to a masked dataset, where:

    Residual_Flux_Fraction = Residuals / Data = (Data - Model)/Data

    The residual flux fraction map values in masked pixels are returned as zero.

    Parameters
    ----------
    residual_map
        The residual-map of the model-data fit to the dataset.
    data
        The data of the dataset.
    mask
        The mask applied to the residual-map, where `False` entries are included in the calculation.
    """
    return jnp.where(mask == 0, residual_map / data, 0)
