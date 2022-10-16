import autoarray as aa

import numpy as np
import pytest


def test__residual_map_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    assert (residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()

    data = np.array([10.0, 10.0, 10.0, 10.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    assert (residual_map == np.array([-1.0, 0.0, 1.0, 2.0])).all()


def test__residual_map_with_mask_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    assert (residual_map == np.array([0.0, 0.0, 1.0, 0.0])).all()


def test__normalized_residual_map_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    normalized_residual_map = aa.util.fit.normalized_residual_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (normalized_residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()

    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    normalized_residual_map = aa.util.fit.normalized_residual_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (
        normalized_residual_map
        == np.array([-(1.0 / 2.0), 0.0, (1.0 / 2.0), (2.0 / 2.0)])
    ).all()


def test__normalized_residual_map_with_mask_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    normalized_residual_map = aa.util.fit.normalized_residual_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    assert (normalized_residual_map == np.array([0.0, 0.0, (1.0 / 2.0), 0.0])).all()


def test__normalized_residual_map_complex_from():

    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    noise_map = np.array([2.0 + 2.0j, 2.0 + 2.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 12.0j])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    normalized_residual_map = aa.util.fit.normalized_residual_map_complex_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (normalized_residual_map == np.array([0.5 - 1.0j, 0.5 - 1.0j])).all()


def test__normalized_residual_map_complex_with_mask_from():

    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    mask = np.array([False, True])
    noise_map = np.array([2.0 + 2.0j, 2.0 + 0.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 8.0j])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    normalized_residual_map = (
        aa.util.fit.normalized_residual_map_complex_with_mask_from(
            residual_map=residual_map, mask=mask, noise_map=noise_map
        )
    )

    assert (normalized_residual_map == np.array([0.5 - 1.0j, 0.0 + 0.0j])).all()


def test__chi_squared_map_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()

    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (
        chi_squared_map
        == np.array([(1.0 / 2.0) ** 2.0, 0.0, (1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0])
    ).all()


def test__chi_squared_map_with_mask_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.0, 0.0, (1.0 / 2.0) ** 2.0, 0.0])).all()

    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.0, 0.0, (1.0 / 2.0) ** 2.0, 0.0])).all()


def test__chi_squared_map_complex_from():

    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    noise_map = np.array([2.0 + 2.0j, 2.0 + 2.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 12.0j])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
        residual_map=residual_map, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.25 + 1.0j, 0.25 + 1.0j])).all()


def test__chi_squared_map_complex_with_mask_from():

    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    mask = np.array([False, True])
    noise_map = np.array([2.0 + 2.0j, 2.0 + 0.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 8.0j])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    chi_squared_map = aa.util.fit.chi_squared_map_complex_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    assert (chi_squared_map == np.array([0.25 + 1.0j, 0.0 + 0.0j])).all()


def test__log_likelihood_from():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    noise_map = np.array([2.0, 2.0, 2.0, 2.0])
    model_data = np.array([10.0, 10.0, 10.0, 10.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

    noise_normalization = aa.util.fit.noise_normalization_from(noise_map=noise_map)

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    chi_squared = 0.0
    noise_normalization = (
        np.log(2.0 * np.pi * (2.0**2.0))
        + np.log(2.0 * np.pi * (2.0**2.0))
        + np.log(2.0 * np.pi * (2.0**2.0))
        + np.log(2.0 * np.pi * (2.0**2.0))
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (chi_squared + noise_normalization), 1.0e-4
    )

    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

    noise_normalization = aa.util.fit.noise_normalization_from(noise_map=noise_map)

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0.25, 0, 0.25, 1.0
    # log_likelihood = -0.5*(0.25+0+0.25+1.0)

    chi_squared = (
        ((1.0 / 2.0) ** 2.0) + 0.0 + ((1.0 / 2.0) ** 2.0) + ((2.0 / 2.0) ** 2.0)
    )
    noise_normalization = (
        np.log(2.0 * np.pi * (2.0**2.0))
        + np.log(2.0 * np.pi * (2.0**2.0))
        + np.log(2.0 * np.pi * (2.0**2.0))
        + np.log(2.0 * np.pi * (2.0**2.0))
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (chi_squared + noise_normalization), 1.0e-4
    )

    noise_map = np.array([1.0, 2.0, 3.0, 4.0])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_from(
        residual_map=residual_map, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

    noise_normalization = aa.util.fit.noise_normalization_from(noise_map=noise_map)

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = (1.0/1.0)**2, (0.0), (-1.0/3.0)**2.0, (2.0/4.0)**2.0

    chi_squared = 1.0 + (1.0 / (3.0**2.0)) + 0.25
    noise_normalization = (
        np.log(2 * np.pi * (1.0**2.0))
        + np.log(2 * np.pi * (2.0**2.0))
        + np.log(2 * np.pi * (3.0**2.0))
        + np.log(2 * np.pi * (4.0**2.0))
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (chi_squared + noise_normalization), 1e-4
    )


def test__log_likelihood_from__with_mask():

    data = np.array([10.0, 10.0, 10.0, 10.0])
    mask = np.array([True, False, False, True])
    noise_map = np.array([1.0, 2.0, 3.0, 4.0])
    model_data = np.array([11.0, 10.0, 9.0, 8.0])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )

    noise_normalization = aa.util.fit.noise_normalization_with_mask_from(
        mask=mask, noise_map=noise_map
    )

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

    chi_squared = 0.0 + (1.0 / 3.0) ** 2.0
    noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (3.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (chi_squared + noise_normalization), 1e-4
    )

    data = np.array([[10.0, 10.0], [10.0, 10.0]])
    mask = np.array([[True, False], [False, True]])
    noise_map = np.array([[1.0, 2.0], [3.0, 4.0]])
    model_data = np.array([[11.0, 10.0], [9.0, 8.0]])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    chi_squared_map = aa.util.fit.chi_squared_map_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )

    noise_normalization = aa.util.fit.noise_normalization_with_mask_from(
        mask=mask, noise_map=noise_map
    )

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0, 0.25, (0.25 and 1.0 are masked)

    chi_squared = 0.0 + (1.0 / 3.0) ** 2.0
    noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (3.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (chi_squared + noise_normalization), 1e-4
    )


def test__log_likelihood_from__complex_data():

    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    noise_map = np.array([2.0 + 1.0j, 2.0 + 1.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 12.0j])

    residual_map = aa.util.fit.residual_map_from(data=data, model_data=model_data)

    chi_squared_map = aa.util.fit.chi_squared_map_complex_from(
        residual_map=residual_map, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_complex_from(chi_squared_map=chi_squared_map)

    noise_normalization = aa.util.fit.noise_normalization_complex_from(
        noise_map=noise_map
    )

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0.25 and 4.0

    chi_squared = 4.25
    noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (1.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * 2.0 * (chi_squared + noise_normalization), 1e-4
    )

    data = np.array([10.0 + 10.0j, 10.0 + 10.0j])
    mask = np.array([False, True])
    noise_map = np.array([2.0 + 1.0j, 2.0 + 0.0j])
    model_data = np.array([9.0 + 12.0j, 9.0 + 8.0j])

    residual_map = aa.util.fit.residual_map_with_mask_from(
        data=data, mask=mask, model_data=model_data
    )

    chi_squared_map = aa.util.fit.chi_squared_map_complex_with_mask_from(
        residual_map=residual_map, mask=mask, noise_map=noise_map
    )

    chi_squared = aa.util.fit.chi_squared_complex_with_mask_from(
        mask=mask, chi_squared_map=chi_squared_map
    )

    noise_normalization = aa.util.fit.noise_normalization_complex_with_mask_from(
        mask=mask, noise_map=noise_map
    )

    log_likelihood = aa.util.fit.log_likelihood_from(
        chi_squared=chi_squared, noise_normalization=noise_normalization
    )

    # chi squared = 0.25 and 4.0

    chi_squared = 4.25
    noise_normalization = np.log(2 * np.pi * (2.0**2.0)) + np.log(
        2 * np.pi * (1.0**2.0)
    )

    assert log_likelihood == pytest.approx(
        -0.5 * (chi_squared + noise_normalization), 1e-4
    )


def test__log_evidence_from():

    likelihood_with_regularization_terms = (
        aa.util.fit.log_likelihood_with_regularization_from(
            chi_squared=3.0, regularization_term=6.0, noise_normalization=2.0
        )
    )

    assert likelihood_with_regularization_terms == -0.5 * (3.0 + 6.0 + 2.0)

    evidences = aa.util.fit.log_evidence_from(
        chi_squared=3.0,
        regularization_term=6.0,
        log_curvature_regularization_term=9.0,
        log_regularization_term=10.0,
        noise_normalization=30.0,
    )

    assert evidences == -0.5 * (3.0 + 6.0 + 9.0 - 10.0 + 30.0)
