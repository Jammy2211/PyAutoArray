import numpy as np
import pytest

import autoarray as aa


def test__visibilities_and_model_are_identical__no_masking__check_values_are_correct():

    real_space_mask = aa.Mask2D.manual(
        mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
    )

    data = aa.Visibilities.manual_slim(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
    noise_map = aa.VisibilitiesNoiseMap.manual_slim(
        visibilities=[2.0 + 2.0j, 2.0 + 2.0j]
    )

    interferometer = aa.Interferometer(
        visibilities=data,
        noise_map=noise_map,
        uv_wavelengths=np.ones(shape=(2, 2)),
        real_space_mask=real_space_mask,
    )

    model_visibilities = aa.Visibilities.manual_slim(
        visibilities=[1.0 + 2.0j, 3.0 + 4.0j]
    )

    fit = aa.m.MockFitInterferometer(
        dataset=interferometer, use_mask_in_fit=False, model_data=model_visibilities
    )

    assert (fit.visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()

    assert (fit.noise_map.slim == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()

    assert (fit.signal_to_noise_map.slim == np.array([0.5 + 1.0j, 1.5 + 2.0j])).all()

    assert (fit.model_visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()

    assert (fit.residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()

    assert (
        fit.normalized_residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])
    ).all()

    assert (fit.chi_squared_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()

    assert fit.chi_squared == 0.0
    assert fit.reduced_chi_squared == 0.0
    assert fit.noise_normalization == pytest.approx(
        4.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


def test__visibilities_and_model_are_different__no_masking__check_values_are_correct():

    real_space_mask = aa.Mask2D.manual(
        mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
    )

    data = aa.Visibilities.manual_slim(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
    noise_map = aa.VisibilitiesNoiseMap.manual_slim(
        visibilities=[2.0 + 2.0j, 2.0 + 2.0j]
    )

    interferometer = aa.Interferometer(
        visibilities=data,
        noise_map=noise_map,
        uv_wavelengths=np.ones(shape=(2, 2)),
        real_space_mask=real_space_mask,
    )

    model_visibilities = aa.Visibilities.manual_slim(
        visibilities=[1.0 + 2.0j, 3.0 + 3.0j]
    )

    fit = aa.m.MockFitInterferometer(
        dataset=interferometer, use_mask_in_fit=False, model_data=model_visibilities
    )

    assert (fit.visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()

    assert (fit.noise_map.slim == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()

    assert (fit.signal_to_noise_map.slim == np.array([0.5 + 1.0j, 1.5 + 2.0j])).all()

    assert (fit.model_visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 3.0j])).all()

    assert (fit.residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 1.0j])).all()

    assert (
        fit.normalized_residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.5j])
    ).all()

    assert (fit.chi_squared_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.25j])).all()

    assert fit.chi_squared == 0.25
    assert fit.reduced_chi_squared == 0.25 / 2.0
    assert fit.noise_normalization == pytest.approx(
        4.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


def test__visibilities_and_model_are_identical__inversion_included__changes_certain_properties():

    real_space_mask = aa.Mask2D.manual(
        mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
    )

    data = aa.Visibilities.manual_slim(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
    noise_map = aa.VisibilitiesNoiseMap.manual_slim(
        visibilities=[2.0 + 2.0j, 2.0 + 2.0j]
    )

    interferometer = aa.Interferometer(
        visibilities=data,
        noise_map=noise_map,
        uv_wavelengths=np.ones(shape=(2, 2)),
        real_space_mask=real_space_mask,
    )

    model_visibilities = aa.Visibilities.manual_slim(
        visibilities=[1.0 + 2.0j, 3.0 + 4.0j]
    )

    inversion = aa.m.MockInversion(linear_obj_list=[aa.m.MockMapper()], data_vector=1)

    inversion = aa.m.MockInversion(
        inversion=inversion,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
    )

    fit = aa.m.MockFitInterferometer(
        dataset=interferometer,
        use_mask_in_fit=False,
        model_data=model_visibilities,
        inversion=inversion,
    )

    assert fit.chi_squared == 0.0
    assert fit.reduced_chi_squared == 0.0
    assert fit.noise_normalization == pytest.approx(
        4.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    assert fit.log_likelihood_with_regularization == -0.5 * (
        fit.chi_squared + 2.0 + fit.noise_normalization
    )
    assert fit.log_evidence == -0.5 * (
        fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
    )
    assert fit.figure_of_merit == fit.log_evidence


def test__dirty_quantities(transformer_7x7_7, interferometer_7, fit_interferometer_7):

    fit_interferometer_7.dataset.transformer = transformer_7x7_7

    dirty_image = transformer_7x7_7.image_from(
        visibilities=interferometer_7.visibilities
    )
    assert (fit_interferometer_7.dirty_image == dirty_image).all()

    dirty_noise_map = transformer_7x7_7.image_from(
        visibilities=interferometer_7.noise_map
    )
    assert (fit_interferometer_7.dirty_noise_map == dirty_noise_map).all()

    dirty_signal_to_noise_map = transformer_7x7_7.image_from(
        visibilities=interferometer_7.signal_to_noise_map
    )
    assert (
        fit_interferometer_7.dirty_signal_to_noise_map == dirty_signal_to_noise_map
    ).all()

    dirty_model_image = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.model_visibilities
    )
    assert (fit_interferometer_7.dirty_model_image == dirty_model_image).all()

    dirty_residual_map = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.residual_map
    )
    assert (fit_interferometer_7.dirty_residual_map == dirty_residual_map).all()

    dirty_normalized_residual_map = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.normalized_residual_map
    )
    assert (
        fit_interferometer_7.dirty_normalized_residual_map
        == dirty_normalized_residual_map
    ).all()

    dirty_chi_squared_map = transformer_7x7_7.image_from(
        visibilities=fit_interferometer_7.chi_squared_map
    )
    assert (fit_interferometer_7.dirty_chi_squared_map == dirty_chi_squared_map).all()
