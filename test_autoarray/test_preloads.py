import numpy as np

import autoarray as aa


def test__set_w_tilde():
    # fit inversion is None, so no need to bother with w_tilde.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fit are different but there is an inversion, so we should not preload w_tilde and use w_tilde.

    inversion = aa.m.MockInversion(linear_obj_list=[aa.m.MockMapper()])

    fit_0 = aa.m.MockFitDataset(
        inversion=inversion,
        noise_map=aa.Array2D.zeros(shape_native=(3, 1), pixel_scales=0.1),
    )
    fit_1 = aa.m.MockFitDataset(
        inversion=inversion,
        noise_map=aa.Array2D.ones(shape_native=(3, 1), pixel_scales=0.1),
    )

    preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fits are the same so preload w_tilde and use it.

    noise_map = aa.Array2D.ones(shape_native=(5, 5), pixel_scales=0.1)

    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        ),
        pixel_scales=noise_map.pixel_scales,
    )

    dataset = aa.m.MockDataset(psf=aa.Kernel2D.no_blur(pixel_scales=1.0), mask=mask)

    fit_0 = aa.m.MockFitDataset(
        inversion=inversion, dataset=dataset, noise_map=noise_map
    )
    fit_1 = aa.m.MockFitDataset(
        inversion=inversion, dataset=dataset, noise_map=noise_map
    )

    preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    (
        curvature_preload,
        indexes,
        lengths,
    ) = aa.util.inversion_imaging.w_tilde_curvature_preload_imaging_from(
        noise_map_native=np.array(fit_0.noise_map.native),
        kernel_native=np.array(fit_0.dataset.psf.native),
        native_index_for_slim_index=np.array(
            fit_0.dataset.mask.derive_indexes.native_for_slim
        ),
    )

    assert preloads.w_tilde.curvature_preload[0] == curvature_preload[0]
    assert preloads.w_tilde.indexes[0] == indexes[0]
    assert preloads.w_tilde.lengths[0] == lengths[0]
    assert preloads.w_tilde.noise_map_value == 1.0
    assert preloads.use_w_tilde == True


def test__set_relocated_grid():
    # Inversion is None so there is no mapper, thus preload mapper to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    inversion_0 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(source_plane_data_grid=np.ones((3, 2)))]
    )
    inversion_1 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(source_plane_data_grid=2.0 * np.ones((3, 2)))]
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert preloads.relocated_grid is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    inversion_0 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(source_plane_data_grid=np.ones((3, 2)))]
    )
    inversion_1 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(source_plane_data_grid=np.ones((3, 2)))]
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(relocated_grid=1)
    preloads.set_relocated_grid(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.relocated_grid == np.ones((3, 2))).all()


def test__set_mapper_list():
    # Inversion is None so there is no mapper, thus preload mapper to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper_list is None

    # Mapper's mapping matrices are different, thus preload mapper to None.

    inversion_0 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(mapping_matrix=np.ones((3, 2)))]
    )
    inversion_1 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(mapping_matrix=2.0 * np.ones((3, 2)))]
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert preloads.mapper_list is None

    # Mapper's mapping matrices are the same, thus preload mapper.

    inversion_0 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(mapping_matrix=np.ones((3, 2)))]
    )
    inversion_1 = aa.m.MockInversion(
        linear_obj_list=[aa.m.MockMapper(mapping_matrix=np.ones((3, 2)))]
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.mapper_list[0].mapping_matrix == np.ones((3, 2))).all()

    # Multiple mappers pre inversion still preloads full mapper list.

    inversion_0 = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockMapper(mapping_matrix=np.ones((3, 2))),
            aa.m.MockMapper(mapping_matrix=np.ones((3, 2))),
        ]
    )
    inversion_1 = aa.m.MockInversion(
        linear_obj_list=[
            aa.m.MockMapper(mapping_matrix=np.ones((3, 2))),
            aa.m.MockMapper(mapping_matrix=np.ones((3, 2))),
        ]
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(mapper_list=1)
    preloads.set_mapper_list(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.mapper_list[0].mapping_matrix == np.ones((3, 2))).all()
    assert (preloads.mapper_list[1].mapping_matrix == np.ones((3, 2))).all()


def test__set_operated_mapping_matrix_with_preloads():
    # Inversion is None thus preload it to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(
        operated_mapping_matrix=1,
    )
    preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)

    assert preloads.operated_mapping_matrix is None

    # Inversion's blurred mapping matrices are different thus no preloading.

    operated_mapping_matrix_0 = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    operated_mapping_matrix_1 = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    inversion_0 = aa.m.MockInversionImaging(
        operated_mapping_matrix=operated_mapping_matrix_0
    )
    inversion_1 = aa.m.MockInversionImaging(
        operated_mapping_matrix=operated_mapping_matrix_1
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(
        operated_mapping_matrix=1,
    )
    preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)

    assert preloads.operated_mapping_matrix is None

    # Inversion's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    inversion_0 = aa.m.MockInversionImaging(
        operated_mapping_matrix=operated_mapping_matrix_0,
    )
    inversion_1 = aa.m.MockInversionImaging(
        operated_mapping_matrix=operated_mapping_matrix_0,
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(
        operated_mapping_matrix=1,
    )
    preloads.set_operated_mapping_matrix_with_preloads(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.operated_mapping_matrix == operated_mapping_matrix_0).all()


def test__set_linear_func_operated_mapping_matrix_dict():
    # Inversion is None thus preload it to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(
        linear_func_operated_mapping_matrix_dict=0,
    )
    preloads.set_linear_func_inversion_dicts(fit_0=fit_0, fit_1=fit_1)

    assert preloads.linear_func_operated_mapping_matrix_dict is None
    assert preloads.data_linear_func_matrix_dict is None

    # Inversion's blurred mapping matrices are different thus no preloading.

    dict_0 = {"key0": np.array([1.0, 2.0])}
    dict_1 = {"key1": np.array([1.0, 3.0])}

    inversion_0 = aa.m.MockInversionImaging(
        linear_obj_list=[aa.m.MockLinearObjFuncList()],
        linear_func_operated_mapping_matrix_dict=dict_0,
    )
    inversion_1 = aa.m.MockInversionImaging(
        linear_obj_list=[aa.m.MockLinearObjFuncList()],
        linear_func_operated_mapping_matrix_dict=dict_1,
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads()
    preloads.set_linear_func_inversion_dicts(fit_0=fit_0, fit_1=fit_1)

    assert preloads.linear_func_operated_mapping_matrix_dict is None
    assert preloads.data_linear_func_matrix_dict is None

    # Inversion's blurred mapping matrices are the same therefore preload it and the curvature sparse terms.

    inversion_0 = aa.m.MockInversionImaging(
        linear_obj_list=[aa.m.MockLinearObjFuncList()],
        linear_func_operated_mapping_matrix_dict=dict_0,
        data_linear_func_matrix_dict=dict_0,
    )
    inversion_1 = aa.m.MockInversionImaging(
        linear_obj_list=[aa.m.MockLinearObjFuncList()],
        linear_func_operated_mapping_matrix_dict=dict_0,
        data_linear_func_matrix_dict=dict_0,
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads()
    preloads.set_linear_func_inversion_dicts(fit_0=fit_0, fit_1=fit_1)

    assert (
        preloads.linear_func_operated_mapping_matrix_dict["key0"] == dict_0["key0"]
    ).all()
    assert (preloads.data_linear_func_matrix_dict["key0"] == dict_0["key0"]).all()


def test__set_curvature_matrix():
    # Inversion is None thus preload curvature_matrix to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(
        curvature_matrix=1, data_vector_mapper=1, curvature_matrix_mapper_diag=1
    )
    preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)

    assert preloads.curvature_matrix is None

    # Inversion's curvature matrices are different thus no preloading.

    curvature_matrix_0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    curvature_matrix_1 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    fit_0 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_0,
            data_vector_mapper=1,
            curvature_matrix_mapper_diag=1,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )
    fit_1 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_1,
            data_vector_mapper=1,
            curvature_matrix_mapper_diag=1,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )

    preloads = aa.Preloads(curvature_matrix=1)
    preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)

    assert preloads.curvature_matrix is None

    # Inversion's curvature matrices are the same therefore preload it and the curvature sparse terms.

    preloads = aa.Preloads(curvature_matrix=2)

    fit_0 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_0,
            data_vector_mapper=1,
            curvature_matrix_mapper_diag=1,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )
    fit_1 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_0,
            data_vector_mapper=1,
            curvature_matrix_mapper_diag=1,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )

    preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.curvature_matrix == curvature_matrix_0).all()


def test__set_curvature_matrix__curvature_matrix_mapper_diag():
    # Inversion is None thus preload curvature_matrix to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(data_vector_mapper=0, curvature_matrix_mapper_diag=1)
    preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)

    assert preloads.data_vector_mapper is None
    assert preloads.curvature_matrix_mapper_diag is None
    assert preloads.mapper_operated_mapping_matrix_dict is None

    # Inversion's curvature matrices are different thus no preloading.

    curvature_matrix_0 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    curvature_matrix_1 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    fit_0 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_0,
            curvature_matrix_mapper_diag=curvature_matrix_0,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )
    fit_1 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_1,
            curvature_matrix_mapper_diag=curvature_matrix_1,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )

    preloads = aa.Preloads(
        data_vector_mapper=0,
        curvature_matrix_mapper_diag=1,
        mapper_operated_mapping_matrix_dict=2,
    )
    preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)

    assert preloads.data_vector_mapper is None
    assert preloads.curvature_matrix_mapper_diag is None
    assert preloads.mapper_operated_mapping_matrix_dict is None

    # Inversion's curvature matrices are the same therefore preload it and the curvature sparse terms.

    preloads = aa.Preloads(data_vector_mapper=10, curvature_matrix_mapper_diag=2)

    fit_0 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_0,
            data_vector_mapper=0,
            curvature_matrix_mapper_diag=curvature_matrix_0,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )
    fit_1 = aa.m.MockFitDataset(
        inversion=aa.m.MockInversion(
            curvature_matrix=curvature_matrix_1,
            data_vector_mapper=0,
            curvature_matrix_mapper_diag=curvature_matrix_0,
            mapper_operated_mapping_matrix_dict={"key0": 1},
        )
    )

    preloads.set_curvature_matrix(fit_0=fit_0, fit_1=fit_1)

    assert preloads.data_vector_mapper == 0
    assert (preloads.curvature_matrix_mapper_diag == curvature_matrix_0).all()
    assert preloads.mapper_operated_mapping_matrix_dict == {"key0": 1}


def test__set_regularization_matrix_and_term():
    regularization = aa.m.MockRegularization(regularization_matrix=np.eye(2))

    # Inversion is None thus preload log_det_regularization_matrix_term to None.

    fit_0 = aa.m.MockFitDataset(inversion=None)
    fit_1 = aa.m.MockFitDataset(inversion=None)

    preloads = aa.Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's log_det_regularization_matrix_term are different thus no preloading.

    inversion_0 = aa.m.MockInversion(
        log_det_regularization_matrix_term=0,
        linear_obj_list=[aa.m.MockLinearObj(regularization=regularization)],
    )

    inversion_1 = aa.m.MockInversion(
        log_det_regularization_matrix_term=1,
        linear_obj_list=[aa.m.MockLinearObj(regularization=regularization)],
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads = aa.Preloads(log_det_regularization_matrix_term=1)
    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert preloads.regularization_matrix is None
    assert preloads.log_det_regularization_matrix_term is None

    # Inversion's regularization matrix terms are the same therefore preload it and the regularization matrix.

    preloads = aa.Preloads(log_det_regularization_matrix_term=2)

    inversion_0 = aa.m.MockInversion(
        log_det_regularization_matrix_term=1,
        linear_obj_list=[aa.m.MockMapper(regularization=regularization)],
    )

    inversion_1 = aa.m.MockInversion(
        log_det_regularization_matrix_term=1,
        linear_obj_list=[aa.m.MockMapper(regularization=regularization)],
    )

    fit_0 = aa.m.MockFitDataset(inversion=inversion_0)
    fit_1 = aa.m.MockFitDataset(inversion=inversion_1)

    preloads.set_regularization_matrix_and_term(fit_0=fit_0, fit_1=fit_1)

    assert (preloads.regularization_matrix == np.eye(2)).all()
    assert preloads.log_det_regularization_matrix_term == 1
