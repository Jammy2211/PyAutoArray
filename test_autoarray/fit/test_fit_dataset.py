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


# def test__figure_of_merit__with_noise_covariance_matrix_in_dataset(masked_imaging_7x7, model_image_7x7):
