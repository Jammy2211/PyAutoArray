import numpy as np

import autoarray as aa


def test__data_and_model_are_identical__no_masking__check_values_are_correct():
    mask = aa.Mask2D(mask=[[False, False], [False, False]], pixel_scales=(1.0, 1.0))

    data = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)
    noise_map = aa.Array2D(values=[2.0, 2.0, 2.0, 2.0], mask=mask)

    dataset = aa.Imaging(data=data, noise_map=noise_map)

    dataset = dataset.apply_mask(mask=mask)

    model_data = aa.Array2D(values=[1.0, 2.0, 3.0, 4.0], mask=mask)

    fit = aa.m.MockFitImaging(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )

    assert (fit.mask == np.array([[False, False], [False, False]])).all()
    assert (fit.data == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (fit.noise_map == np.array([2.0, 2.0, 2.0, 2.0])).all()
    assert (fit.signal_to_noise_map == np.array([0.5, 1.0, 1.5, 2.0])).all()
    assert (fit.model_data == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (fit.residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
    assert (fit.normalized_residual_map == np.array([0.0, 0.0, 0.0, 0.0])).all()
    assert (fit.chi_squared_map == np.array([0.0, 0.0, 0.0, 0.0])).all()

    assert fit.chi_squared == 0.0
    assert fit.reduced_chi_squared == 0.0
    assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map.array**2.0))
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


def test__data_and_model_are_different__include_masking__check_values_are_correct():
    mask = aa.Mask2D(mask=[[False, False], [True, False]], pixel_scales=(1.0, 1.0))

    data = aa.Array2D(values=[1.0, 2.0, 4.0], mask=mask)
    noise_map = aa.Array2D(values=[2.0, 2.0, 2.0], mask=mask)

    dataset = aa.Imaging(data=data, noise_map=noise_map)

    model_data = aa.Array2D(values=[1.0, 2.0, 3.0], mask=mask)

    fit = aa.m.MockFitImaging(
        dataset=dataset, use_mask_in_fit=False, model_data=model_data
    )

    assert (fit.mask == np.array([[False, False], [True, False]])).all()
    assert (fit.data.slim == np.array([1.0, 2.0, 4.0])).all()
    assert (fit.noise_map.slim == np.array([2.0, 2.0, 2.0])).all()
    assert (fit.signal_to_noise_map.slim == np.array([0.5, 1.0, 2.0])).all()
    assert (fit.model_data.slim == np.array([1.0, 2.0, 3.0])).all()
    assert (fit.residual_map.slim == np.array([0.0, 0.0, 1.0])).all()
    assert (fit.normalized_residual_map.slim == np.array([0.0, 0.0, 0.5])).all()
    assert (fit.chi_squared_map.slim == np.array([0.0, 0.0, 0.25])).all()

    assert fit.chi_squared == 0.25
    assert fit.reduced_chi_squared == 0.25 / 3.0
    assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map.array**2.0))
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)


def test__data_and_model_are_identical__inversion_included__changes_certain_properties():
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

    assert fit.chi_squared == 0.0
    assert fit.reduced_chi_squared == 0.0
    assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map.array**2.0))
    assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    assert fit.log_likelihood_with_regularization == -0.5 * (
        fit.chi_squared + 2.0 + fit.noise_normalization
    )
    assert fit.log_evidence == -0.5 * (
        fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
    )
    assert fit.figure_of_merit == fit.log_evidence

