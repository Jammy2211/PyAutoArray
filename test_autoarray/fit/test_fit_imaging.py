import numpy as np

import autoarray as aa


# ---------------------------------------------------------------------------
# Helper: build the "identical model, no masking" fit used by multiple tests
# ---------------------------------------------------------------------------


def _make_identical_fit_no_mask():
    mask = aa.Mask2D(mask=[[False, False], [False, False]], pixel_scales=(1.0, 1.0))
    data = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)
    noise_map = aa.Array2D(values=[2.0, 2.0, 2.0, 2.0], mask=mask)
    dataset = aa.Imaging(data=data, noise_map=noise_map)
    dataset = dataset.apply_mask(mask=mask)
    model_data = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)
    fit = aa.m.MockFitImaging(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )
    return fit, noise_map


def _make_different_fit_with_mask():
    mask = aa.Mask2D(mask=[[False, False], [True, False]], pixel_scales=(1.0, 1.0))
    data = aa.Array2D(values=[1.0, 2.0, 4.0], mask=mask)
    noise_map = aa.Array2D(values=[2.0, 2.0, 2.0], mask=mask)
    dataset = aa.Imaging(data=data, noise_map=noise_map)
    model_data = aa.Array2D(values=[1.0, 2.0, 3.0], mask=mask)
    fit = aa.m.MockFitImaging(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )
    return fit, noise_map


def _make_identical_fit_with_inversion():
    mask = aa.Mask2D(mask=[[False, False], [False, False]], pixel_scales=(1.0, 1.0))
    data = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)
    noise_map = aa.Array2D(values=[2.0, 2.0, 2.0, 2.0], mask=mask)
    dataset = aa.Imaging(data=data, noise_map=noise_map)
    dataset = dataset.apply_mask(mask=mask)
    model_data = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)
    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper()],
        data_vector=1,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
    )
    fit = aa.m.MockFitImaging(
        dataset=dataset,
        use_mask_in_fit=False,
        model_data=model_data,
        inversion=inversion,
    )
    return fit, noise_map


# ---------------------------------------------------------------------------
# Tests: identical data and model, no masking
# ---------------------------------------------------------------------------


def test__mask__no_masking__returns_2x2_all_false_mask():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.mask == np.array([[False, False], [False, False]])).all()


def test__data__no_masking__returns_correct_data_values():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.data == np.array([1.0, 2.0, 3.0, 4.0])).all()


def test__noise_map__no_masking__returns_correct_noise_map_values():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()


def test__signal_to_noise_map__no_masking__returns_correct_signal_to_noise_values():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()


def test__residual_map__identical_data_and_model__all_zero_residuals():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__normalized_residual_map__identical_data_and_model__all_zero_normalized_residuals():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.normalized_residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__chi_squared_map__identical_data_and_model__all_zero_chi_squared_map():
    fit, _ = _make_identical_fit_no_mask()

    assert (fit.chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()


def test__chi_squared__identical_data_and_model__is_zero():
    fit, _ = _make_identical_fit_no_mask()

    assert fit.chi_squared == 0.0


def test__reduced_chi_squared__identical_data_and_model__is_zero():
    fit, _ = _make_identical_fit_no_mask()

    assert fit.reduced_chi_squared == 0.0


def test__noise_normalization__uniform_noise_map__correct_log_sum_formula():
    fit, noise_map = _make_identical_fit_no_mask()

    assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map.array**2.0))


def test__log_likelihood__identical_data_and_model__negative_half_noise_normalization():
    fit, _ = _make_identical_fit_no_mask()

    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


# ---------------------------------------------------------------------------
# Tests: different data and model with partial mask
# ---------------------------------------------------------------------------


def test__residual_map__different_data_and_model_with_partial_mask__correct_slim_residuals():
    fit, _ = _make_different_fit_with_mask()

    assert (fit.residual_map.slim == np.array([0.0, 0.0, 1.0])).all()


def test__normalized_residual_map__different_data_and_model_with_partial_mask__correct_slim_values():
    fit, _ = _make_different_fit_with_mask()

    assert (fit.normalized_residual_map.slim == np.array([0.0, 0.0, 0.5])).all()


def test__chi_squared_map__different_data_and_model_with_partial_mask__correct_slim_chi_squared():
    fit, _ = _make_different_fit_with_mask()

    assert (fit.chi_squared_map.slim == np.array([0.0, 0.0, 0.25])).all()


def test__chi_squared__different_model_with_masked_data__correct_value():
    fit, _ = _make_different_fit_with_mask()

    assert fit.chi_squared == 0.25


def test__reduced_chi_squared__different_model_with_masked_data__divided_by_unmasked_pixel_count():
    fit, _ = _make_different_fit_with_mask()

    assert fit.reduced_chi_squared == 0.25 / 3.0


def test__log_likelihood__different_model_with_masked_data__correct_value():
    fit, noise_map = _make_different_fit_with_mask()

    assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map.array**2.0))
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


# ---------------------------------------------------------------------------
# Tests: identical data and model with inversion
# ---------------------------------------------------------------------------


def test__chi_squared__identical_data_and_model_with_inversion__is_zero():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.chi_squared == 0.0


def test__reduced_chi_squared__identical_data_and_model_with_inversion__is_zero():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.reduced_chi_squared == 0.0


def test__log_likelihood_with_regularization__with_inversion__adds_regularization_term():
    fit, noise_map = _make_identical_fit_with_inversion()

    assert fit.log_likelihood_with_regularization == -0.5 * (
        fit.chi_squared + 2.0 + fit.noise_normalization
    )


def test__log_evidence__with_inversion__uses_chi_squared_reg_and_determinant_terms():
    fit, noise_map = _make_identical_fit_with_inversion()

    assert fit.log_evidence == -0.5 * (
        fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
    )


def test__figure_of_merit__with_inversion__equals_log_evidence():
    fit, _ = _make_identical_fit_with_inversion()

    assert fit.figure_of_merit == fit.log_evidence
