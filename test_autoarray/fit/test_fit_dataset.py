import pytest

import autoarray as aa


def test__figure_of_merit__with_inversion(masked_imaging_7x7, model_image_7x7):
    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(regularization=aa.m.MockRegularization())],
        data_vector=1,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_evidence

    inversion = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockLinearObj(regularization=None)], data_vector=1
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_likelihood


def test__figure_of_merit__with_noise_covariance_matrix_in_dataset(
    masked_imaging_covariance_7x7, model_image_7x7, masked_imaging_7x7
):
    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_covariance_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
    )

    chi_squared = aa.util.fit.chi_squared_with_noise_covariance_from(
        residual_map=fit.residual_map,
        noise_covariance_matrix_inv=masked_imaging_covariance_7x7.noise_covariance_matrix_inv,
    )

    assert fit.chi_squared == chi_squared

    assert fit.figure_of_merit == pytest.approx(
        -0.5 * (fit.chi_squared + fit.noise_normalization), 1.0e-4
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
    )

    assert fit.chi_squared != pytest.approx(chi_squared, 1.0e-4)


def test__grid_offset_via_data_model(imaging_7x7, mask_2d_7x7, model_image_7x7):

    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7, disable_fft_pad=True)

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        dataset_model=aa.DatasetModel(grid_offset=(1.0, 2.0)),
    )

    assert fit.dataset_model.grid_offset == (1.0, 2.0)

    assert fit.grids.lp[0] == pytest.approx((0.0, -3.0), 1.0e-4)
    assert fit.grids.pixelization[0] == pytest.approx((0.0, -3.0), 1.0e-4)
