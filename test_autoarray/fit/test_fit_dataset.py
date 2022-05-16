import numpy as np

import autoarray as aa


def test__inversion_figure_of_merit(masked_imaging_7x7, model_image_7x7):

    leq = aa.m.MockLEq(linear_obj_list=[aa.m.MockMapper()], data_vector=1)

    inversion = aa.m.MockInversion(
        leq=leq,
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

    leq = aa.m.MockLEq(linear_obj_list=[aa.m.MockLinearObjFunc()], data_vector=1)

    inversion = aa.m.MockInversion(
        leq=leq,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
        settings=aa.SettingsInversion(linear_func_only_use_evidence=False),
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_likelihood

    inversion = aa.m.MockInversion(
        leq=leq,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
        settings=aa.SettingsInversion(linear_func_only_use_evidence=True),
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_evidence

    leq = aa.m.MockLEq(
        linear_obj_list=[aa.m.MockMapper(), aa.m.MockLinearObjFunc()], data_vector=1
    )

    inversion = aa.m.MockInversion(
        leq=leq,
        regularization_term=2.0,
        log_det_curvature_reg_matrix_term=3.0,
        log_det_regularization_matrix_term=4.0,
        settings=aa.SettingsInversion(linear_func_only_use_evidence=True),
    )

    fit = aa.m.MockFitImaging(
        dataset=masked_imaging_7x7,
        use_mask_in_fit=False,
        model_data=model_image_7x7,
        inversion=inversion,
    )

    assert fit.figure_of_merit == fit.log_evidence
