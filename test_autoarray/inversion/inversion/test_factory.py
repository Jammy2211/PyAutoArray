import copy
import numpy as np
import pytest

import autoarray as aa


def test__inversion_imaging__via_linear_obj_func_list(masked_imaging_7x7_no_blur):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 1))
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)
    assert inversion.reconstruction == pytest.approx(np.array([2.0]), 1.0e-4)

    # Overwrites use_w_tilde to false.

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)
    assert inversion.reconstruction == pytest.approx(np.array([2.0]), 1.0e-4)

    # Works with multiple parameters

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 2))
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)
    assert inversion.reconstruction == pytest.approx(np.array([1.0, 1.0]), 1.0e-4)


def test__inversion_imaging__via_mapper(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperRectangularNoInterp)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(6.9546, 1.0e-4)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperRectangularNoInterp)
    assert isinstance(inversion, aa.InversionImagingWTilde)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(6.9546, 1.0e-4)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[delaunay_mapper_9_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperDelaunay)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(10.6674, 1.0e-4)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[delaunay_mapper_9_3x3],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperDelaunay)
    assert isinstance(inversion, aa.InversionImagingWTilde)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(10.6674, 1.0e-4)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)


def test__inversion_imaging__via_regularizations(
    masked_imaging_7x7_no_blur,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
    voronoi_mapper_nn_9_3x3,
    regularization_constant,
    regularization_constant_split,
    regularization_adaptive_brightness,
    regularization_adaptive_brightness_split,
    regularization_gaussian_kernel,
    regularization_exponential_kernel,
):
    mapper = copy.copy(delaunay_mapper_9_3x3)
    mapper.regularization = regularization_constant

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[mapper],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperDelaunay)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        10.66747, 1.0e-4
    )
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    mapper = copy.copy(delaunay_mapper_9_3x3)
    mapper.regularization = regularization_adaptive_brightness

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[mapper],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperDelaunay)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        47.410169, 1.0e-4
    )
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    # Have to do this because NN library is optional.

    try:

        mapper = copy.copy(voronoi_mapper_nn_9_3x3)
        mapper.regularization = regularization_constant

        inversion = aa.Inversion(
            dataset=masked_imaging_7x7_no_blur,
            linear_obj_list=[mapper],
            settings=aa.SettingsInversion(use_w_tilde=True),
        )

        assert isinstance(inversion.linear_obj_list[0], aa.MapperVoronoi)
        assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
            10.66505, 1.0e-4
        )
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

        mapper = copy.copy(voronoi_mapper_nn_9_3x3)
        mapper.regularization = regularization_constant_split

        inversion = aa.Inversion(
            dataset=masked_imaging_7x7_no_blur,
            linear_obj_list=[mapper],
            settings=aa.SettingsInversion(use_w_tilde=True),
        )

        assert isinstance(inversion.linear_obj_list[0], aa.MapperVoronoi)
        assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
            10.37955, 1.0e-4
        )
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    except AttributeError:
        pass


def test__inversion_imaging__via_linear_obj_func_and_mapper(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObj(
        parameters=1,
        grid=grid,
        mapping_matrix=np.full(fill_value=0.5, shape=(9, 1)),
        regularization=None,
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(
            use_w_tilde=False,
            no_regularization_add_to_curvature_diag_value=False,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert isinstance(inversion.linear_obj_list[1], aa.MapperRectangularNoInterp)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        6.95465245, 1.0e-4
    )
    assert inversion.reconstruction_dict[linear_obj] == pytest.approx(
        np.array([2.0]), 1.0e-4
    )
    assert inversion.reconstruction_dict[rectangular_mapper_7x7_3x3][0] < 1.0e-4
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)


def test__inversion_imaging__via_linear_obj_func_and_mapper__force_edge_pixels_to_zero(
    masked_imaging_7x7_no_blur,
    voronoi_mapper_9_3x3,
):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObj(
        parameters=1,
        grid=grid,
        mapping_matrix=np.full(fill_value=0.5, shape=(9, 1)),
        regularization=None,
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, voronoi_mapper_9_3x3],
        settings=aa.SettingsInversion(
            use_w_tilde=False,
            no_regularization_add_to_curvature_diag_value=False,
            force_edge_pixels_to_zeros=True,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert isinstance(inversion.linear_obj_list[1], aa.MapperVoronoi)
    assert isinstance(inversion, aa.InversionImagingMapping)

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, voronoi_mapper_9_3x3],
        settings=aa.SettingsInversion(
            use_w_tilde=False,
            use_positive_only_solver=True,
            no_regularization_add_to_curvature_diag_value=False,
            force_edge_pixels_to_zeros=True,
        ),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObj)
    assert isinstance(inversion.linear_obj_list[1], aa.MapperVoronoi)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.reconstruction == pytest.approx(
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.0e-4
    )


def test__inversion_imaging__compare_mapping_and_w_tilde_values(
    masked_imaging_7x7, voronoi_mapper_9_3x3
):
    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[voronoi_mapper_9_3x3],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[voronoi_mapper_9_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert inversion_w_tilde._curvature_matrix_mapper_diag == pytest.approx(
        inversion_mapping._curvature_matrix_mapper_diag, 1.0e-4
    )
    assert inversion_w_tilde.reconstruction == pytest.approx(
        inversion_mapping.reconstruction, 1.0e-4
    )
    assert inversion_w_tilde.mapped_reconstructed_image == pytest.approx(
        inversion_mapping.mapped_reconstructed_image, 1.0e-4
    )
    assert inversion_w_tilde.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_mapping.log_det_curvature_reg_matrix_term
    )


def test__inversion_imaging__linear_obj_func_and_non_func_give_same_terms(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    masked_imaging_7x7_no_blur = copy.copy(masked_imaging_7x7_no_blur)
    masked_imaging_7x7_no_blur.data[4] = 2.0

    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObj(
        parameters=2, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 2))
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    masked_imaging_7x7_no_blur = copy.copy(masked_imaging_7x7_no_blur)

    masked_imaging_7x7_no_blur.data -= inversion.mapped_reconstructed_data_dict[
        linear_obj
    ]

    inversion_no_linear_func = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.regularization_term == pytest.approx(
        inversion_no_linear_func.regularization_term, 1.0e-4
    )
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        inversion_no_linear_func.log_det_curvature_reg_matrix_term, 1.0e-4
    )
    assert inversion.log_det_regularization_matrix_term == pytest.approx(
        inversion_no_linear_func.log_det_regularization_matrix_term, 1.0e-4
    )


def test__inversion_imaging__linear_obj_func_with_w_tilde(
    masked_imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    masked_imaging_7x7 = copy.copy(masked_imaging_7x7)
    masked_imaging_7x7.data[4] = 2.0
    masked_imaging_7x7.noise_map[3] = 4.0
    masked_imaging_7x7.psf[0] = 0.1
    masked_imaging_7x7.psf[4] = 0.9

    mask = masked_imaging_7x7.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    mapping_matrix = np.full(fill_value=0.5, shape=(9, 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[linear_obj, rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert inversion_mapping.data_vector == pytest.approx(
        inversion_w_tilde.data_vector, 1.0e-4
    )
    assert inversion_mapping.curvature_matrix == pytest.approx(
        inversion_w_tilde.curvature_matrix, 1.0e-4
    )
    assert inversion_mapping.mapped_reconstructed_image == pytest.approx(
        inversion_w_tilde.mapped_reconstructed_image, 1.0e-4
    )

    linear_obj_1 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    linear_obj_2 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            voronoi_mapper_9_3x3,
            linear_obj_2,
        ],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            voronoi_mapper_9_3x3,
            linear_obj_2,
        ],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert inversion_mapping.data_vector == pytest.approx(
        inversion_w_tilde.data_vector, 1.0e-4
    )
    assert inversion_mapping.curvature_matrix == pytest.approx(
        inversion_w_tilde.curvature_matrix, 1.0e-4
    )


def test__inversion_imaging__linear_obj_func_with_w_tilde__include_preload_data_linear_func_matrix(
    masked_imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    masked_imaging_7x7 = copy.copy(masked_imaging_7x7)
    masked_imaging_7x7.data[4] = 2.0
    masked_imaging_7x7.noise_map[3] = 4.0
    masked_imaging_7x7.psf[0] = 0.1
    masked_imaging_7x7.psf[4] = 0.9

    mask = masked_imaging_7x7.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    mapping_matrix = np.full(fill_value=0.5, shape=(9, 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    linear_obj_1 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    linear_obj_2 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            voronoi_mapper_9_3x3,
            linear_obj_2,
        ],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    preloads = aa.Preloads(
        data_linear_func_matrix_dict=inversion_mapping.data_linear_func_matrix_dict
    )

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            voronoi_mapper_9_3x3,
            linear_obj_2,
        ],
        preloads=preloads,
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert inversion_mapping.data_vector == pytest.approx(
        inversion_w_tilde.data_vector, 1.0e-4
    )
    assert inversion_mapping.curvature_matrix == pytest.approx(
        inversion_w_tilde.curvature_matrix, 1.0e-4
    )


def test__inversion_imaging__linear_obj_func_with_w_tilde__include_preload_mapper_operated_mapping_matrix_dict(
    masked_imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    masked_imaging_7x7 = copy.copy(masked_imaging_7x7)
    masked_imaging_7x7.data[4] = 2.0
    masked_imaging_7x7.noise_map[3] = 4.0
    masked_imaging_7x7.psf[0] = 0.1
    masked_imaging_7x7.psf[4] = 0.9

    mask = masked_imaging_7x7.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    mapping_matrix = np.full(fill_value=0.5, shape=(9, 2))
    mapping_matrix[0, 0] = 0.8
    mapping_matrix[1, 1] = 0.4

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    linear_obj_1 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    linear_obj_2 = aa.m.MockLinearObjFuncList(
        parameters=2, grid=grid, mapping_matrix=mapping_matrix
    )

    inversion_mapping = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            voronoi_mapper_9_3x3,
            linear_obj_2,
        ],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    preloads = aa.Preloads(
        mapper_operated_mapping_matrix_dict=inversion_mapping.mapper_operated_mapping_matrix_dict
    )

    inversion_w_tilde = aa.Inversion(
        dataset=masked_imaging_7x7,
        linear_obj_list=[
            rectangular_mapper_7x7_3x3,
            linear_obj,
            delaunay_mapper_9_3x3,
            linear_obj_1,
            voronoi_mapper_9_3x3,
            linear_obj_2,
        ],
        preloads=preloads,
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    assert inversion_mapping.data_vector == pytest.approx(
        inversion_w_tilde.data_vector, 1.0e-4
    )

    assert inversion_mapping.curvature_matrix == pytest.approx(
        inversion_w_tilde.curvature_matrix, 1.0e-4
    )


def test__inversion_interferometer__via_mapper(
    interferometer_7_no_fft,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
):
    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperRectangularNoInterp)
    assert isinstance(inversion, aa.InversionInterferometerMapping)
    assert inversion.mapped_reconstructed_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_data) > 0.0).all()
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(10.2116, 1.0e-4)

    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[delaunay_mapper_9_3x3],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperDelaunay)
    assert isinstance(inversion, aa.InversionInterferometerMapping)
    assert inversion.mapped_reconstructed_data == pytest.approx(
        1.0 + 0.0j * np.ones(shape=(7,)), 1.0e-4
    )
    assert (np.imag(inversion.mapped_reconstructed_data) < 0.0001).all()
    assert (np.imag(inversion.mapped_reconstructed_data) > 0.0).all()
    assert inversion.log_det_curvature_reg_matrix_term == pytest.approx(
        14.49772, 1.0e-4
    )


def test__inversion_matrices__x2_mappers(
    masked_imaging_7x7_no_blur,
    rectangular_mapper_7x7_3x3,
    voronoi_mapper_9_3x3,
    regularization_constant,
):
    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[rectangular_mapper_7x7_3x3, voronoi_mapper_9_3x3],
    )

    assert (
        inversion.operated_mapping_matrix[0:9, 0:9]
        == rectangular_mapper_7x7_3x3.mapping_matrix
    ).all()
    assert (
        inversion.operated_mapping_matrix[0:9, 9:18]
        == voronoi_mapper_9_3x3.mapping_matrix
    ).all()

    operated_mapping_matrix = np.hstack(
        [rectangular_mapper_7x7_3x3.mapping_matrix, voronoi_mapper_9_3x3.mapping_matrix]
    )

    assert inversion.operated_mapping_matrix == pytest.approx(
        operated_mapping_matrix, 1.0e-4
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=operated_mapping_matrix, noise_map=inversion.noise_map
    )

    assert inversion.curvature_matrix == pytest.approx(curvature_matrix, 1.0e-4)

    regularization_matrix_of_reg_0 = regularization_constant.regularization_matrix_from(
        linear_obj=rectangular_mapper_7x7_3x3
    )
    regularization_matrix_of_reg_1 = regularization_constant.regularization_matrix_from(
        linear_obj=voronoi_mapper_9_3x3
    )

    assert (
        inversion.regularization_matrix[0:9, 0:9] == regularization_matrix_of_reg_0
    ).all()
    assert (
        inversion.regularization_matrix[9:18, 9:18] == regularization_matrix_of_reg_1
    ).all()
    assert (inversion.regularization_matrix[0:9, 9:18] == np.zeros((9, 9))).all()
    assert (inversion.regularization_matrix[9:18, 0:9] == np.zeros((9, 9))).all()

    reconstruction_0 = 0.5 * np.ones(9)
    reconstruction_1 = 0.5 * np.ones(9)

    assert inversion.reconstruction_dict[rectangular_mapper_7x7_3x3] == pytest.approx(
        reconstruction_0, 1.0e-4
    )
    assert inversion.reconstruction_dict[voronoi_mapper_9_3x3] == pytest.approx(
        reconstruction_1, 1.0e-4
    )
    assert inversion.reconstruction == pytest.approx(
        np.concatenate([reconstruction_0, reconstruction_1]), 1.0e-4
    )

    assert inversion.mapped_reconstructed_data_dict[
        rectangular_mapper_7x7_3x3
    ] == pytest.approx(0.5 * np.ones(9), 1.0e-4)
    assert inversion.mapped_reconstructed_data_dict[
        voronoi_mapper_9_3x3
    ] == pytest.approx(0.5 * np.ones(9), 1.0e-4)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)


def test__inversion_imaging__positive_only_solver(masked_imaging_7x7_no_blur):
    mask = masked_imaging_7x7_no_blur.mask

    grid = aa.Grid2D.from_mask(mask=mask)

    linear_obj = aa.m.MockLinearObjFuncList(
        parameters=1, grid=grid, mapping_matrix=np.full(fill_value=0.5, shape=(9, 1))
    )

    inversion = aa.Inversion(
        dataset=masked_imaging_7x7_no_blur,
        linear_obj_list=[linear_obj],
        settings=aa.SettingsInversion(use_w_tilde=False, use_positive_only_solver=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.m.MockLinearObjFuncList)
    assert isinstance(inversion, aa.InversionImagingMapping)
    assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)
    assert inversion.reconstruction == pytest.approx(np.array([2.0]), 1.0e-4)
