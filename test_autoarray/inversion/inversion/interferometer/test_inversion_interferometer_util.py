import autoarray as aa
import numpy as np
import pytest


def test__data_vector_via_transformed_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    data_real = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_real = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_real_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_real,
            noise_map=noise_map_real,
        )
    )

    data_imag = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_imag = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_imag_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_imag,
            noise_map=noise_map_imag,
        )
    )

    data_vector_complex_via_blurred = (
        data_vector_real_via_blurred + data_vector_imag_via_blurred
    )

    transformed_mapping_matrix = np.array(
        [
            [1.0 + 1.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 1.0 + 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    data = np.array(
        [4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j, 16.0 + 16.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )
    noise_map = np.array(
        [2.0 + 2.0j, 1.0 + 1.0j, 1.0 + 1.0j, 4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )

    data_vector_via_transformed = aa.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=data,
        noise_map=noise_map,
    )

    assert (data_vector_complex_via_blurred == data_vector_via_transformed).all()


def test__w_tilde_curvature_interferometer_from():
    noise_map = np.array([1.0, 2.0, 3.0])
    uv_wavelengths = np.array([[0.0001, 2.0, 3000.0], [3000.0, 2.0, 0.0001]])

    grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=0.0005)

    w_tilde = aa.util.inversion_interferometer.w_tilde_curvature_interferometer_from(
        noise_map_real=np.array(noise_map),
        uv_wavelengths=np.array(uv_wavelengths),
        grid_radians_slim=np.array(grid),
    )

    assert w_tilde == pytest.approx(
        np.array(
            [
                [1.25, 0.75, 1.24997, 0.74998],
                [0.75, 1.25, 0.74998, 1.24997],
                [1.24994, 0.74998, 1.25, 0.75],
                [0.74998, 1.24997, 0.75, 1.25],
            ]
        ),
        1.0e-4,
    )


def test__curvature_matrix_via_w_tilde_preload_from():
    noise_map = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    uv_wavelengths = np.array(
        [[0.0001, 2.0, 3000.0, 50000.0, 200000.0], [3000.0, 2.0, 0.0001, 10.0, 5000.0]]
    )

    grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.0005)

    w_tilde = aa.util.inversion_interferometer.w_tilde_curvature_interferometer_from(
        noise_map_real=np.array(noise_map),
        uv_wavelengths=np.array(uv_wavelengths),
        grid_radians_slim=np.array(grid),
    )

    mapping_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    curvature_matrix_via_w_tilde = aa.util.inversion.curvature_matrix_via_w_tilde_from(
        w_tilde=w_tilde, mapping_matrix=mapping_matrix
    )

    w_tilde_preload = (
        aa.util.inversion_interferometer.w_tilde_curvature_preload_interferometer_from(
            noise_map_real=np.array(noise_map),
            uv_wavelengths=np.array(uv_wavelengths),
            shape_masked_pixels_2d=(3, 3),
            grid_radians_2d=np.array(grid.native),
        )
    )

    pix_indexes_for_sub_slim_index = np.array(
        [[0], [2], [1], [1], [2], [2], [0], [2], [0]]
    )

    pix_size_for_sub_slim_index = np.ones(shape=(9,)).astype("int")
    pix_weights_for_sub_slim_index = np.ones(shape=(9, 1))

    native_index_for_slim_index = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    curvature_matrix_via_preload = aa.util.inversion_interferometer.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
        curvature_preload=w_tilde_preload,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        native_index_for_slim_index=native_index_for_slim_index,
        pix_pixels=3,
    )

    assert curvature_matrix_via_w_tilde == pytest.approx(
        curvature_matrix_via_preload, 1.0e-4
    )


def test__curvature_matrix_via_w_tilde_two_methods_agree():
    noise_map = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    uv_wavelengths = np.array(
        [[0.0001, 2.0, 3000.0, 50000.0, 200000.0], [3000.0, 2.0, 0.0001, 10.0, 5000.0]]
    )

    grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.0005)

    w_tilde = aa.util.inversion_interferometer.w_tilde_curvature_interferometer_from(
        noise_map_real=np.array(noise_map),
        uv_wavelengths=np.array(uv_wavelengths),
        grid_radians_slim=np.array(grid),
    )

    w_tilde_preload = (
        aa.util.inversion_interferometer.w_tilde_curvature_preload_interferometer_from(
            noise_map_real=np.array(noise_map),
            uv_wavelengths=np.array(uv_wavelengths),
            shape_masked_pixels_2d=(3, 3),
            grid_radians_2d=np.array(grid.native),
        )
    )

    native_index_for_slim_index = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    w_tilde_via_preload = aa.util.inversion_interferometer.w_tilde_via_preload_from(
        w_tilde_preload=w_tilde_preload,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert (w_tilde == w_tilde_via_preload).all()


def test__identical_inversion_values_for_two_methods():
    real_space_mask = aa.Mask2D.all_false(
        shape_native=(7, 7),
        pixel_scales=0.1,
    )

    grid = aa.Grid2D.from_mask(mask=real_space_mask, over_sample_size=1)

    mesh = aa.mesh.Delaunay()

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mesh_grid = aa.Grid2DIrregular(values=mesh_grid)

    mapper_grids = mesh.mapper_grids_from(
        mask=real_space_mask,
        border_relocator=None,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=mesh_grid,
    )

    reg = aa.reg.Constant(coefficient=0.0)

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=reg)

    visibilities = aa.Visibilities(
        visibilities=[
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        ]
    )
    noise_map = aa.VisibilitiesNoiseMap.ones(shape_slim=(7,))
    uv_wavelengths = np.ones(shape=(7, 2))

    dataset = aa.Interferometer(
        data=visibilities,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        transformer_class=aa.TransformerDFT,
    )

    inversion_w_tilde = aa.Inversion(
        dataset=dataset,
        linear_obj_list=[mapper],
        settings=aa.SettingsInversion(use_w_tilde=True),
    )

    inversion_mapping_matrices = aa.Inversion(
        dataset=dataset,
        linear_obj_list=[mapper],
        settings=aa.SettingsInversion(use_w_tilde=False),
    )

    assert (inversion_w_tilde.data == inversion_mapping_matrices.data).all()
    assert (inversion_w_tilde.noise_map == inversion_mapping_matrices.noise_map).all()
    assert (
        inversion_w_tilde.linear_obj_list[0]
        == inversion_mapping_matrices.linear_obj_list[0]
    )
    assert (
        inversion_w_tilde.regularization_list[0]
        == inversion_mapping_matrices.regularization_list[0]
    )
    assert (
        inversion_w_tilde.regularization_matrix
        == inversion_mapping_matrices.regularization_matrix
    ).all()

    assert inversion_w_tilde.data_vector == pytest.approx(
        inversion_mapping_matrices.data_vector, 1.0e-8
    )
    assert inversion_w_tilde.curvature_matrix == pytest.approx(
        inversion_mapping_matrices.curvature_matrix, 1.0e-8
    )
    assert inversion_w_tilde.curvature_reg_matrix == pytest.approx(
        inversion_mapping_matrices.curvature_reg_matrix, 1.0e-8
    )

    assert inversion_w_tilde.reconstruction == pytest.approx(
        inversion_mapping_matrices.reconstruction, abs=1.0e-1
    )
    assert inversion_w_tilde.mapped_reconstructed_image == pytest.approx(
        inversion_mapping_matrices.mapped_reconstructed_image, abs=1.0e-1
    )
    assert inversion_w_tilde.mapped_reconstructed_data == pytest.approx(
        inversion_mapping_matrices.mapped_reconstructed_data, abs=1.0e-1
    )


def test__identical_inversion_source_and_image_loops():
    real_space_mask = aa.Mask2D.all_false(
        shape_native=(7, 7),
        pixel_scales=0.1,
    )

    grid = aa.Grid2D.from_mask(mask=real_space_mask, over_sample_size=1)

    mesh = aa.mesh.Delaunay()

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    mesh_grid = aa.Grid2DIrregular(values=mesh_grid)

    mapper_grids = mesh.mapper_grids_from(
        mask=real_space_mask,
        border_relocator=None,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=mesh_grid,
    )

    reg = aa.reg.Constant(coefficient=0.0)

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=reg)

    visibilities = aa.Visibilities(
        visibilities=[
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
            1.0 + 0.0j,
        ]
    )
    noise_map = aa.VisibilitiesNoiseMap.ones(shape_slim=(7,))
    uv_wavelengths = np.ones(shape=(7, 2))

    dataset = aa.Interferometer(
        data=visibilities,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        transformer_class=aa.TransformerDFT,
    )

    inversion_image_loop = aa.Inversion(
        dataset=dataset,
        linear_obj_list=[mapper],
        settings=aa.SettingsInversion(use_w_tilde=True, use_source_loop=False),
    )

    inversion_source_loop = aa.Inversion(
        dataset=dataset,
        linear_obj_list=[mapper],
        settings=aa.SettingsInversion(use_w_tilde=True, use_source_loop=True),
    )

    assert (inversion_image_loop.data == inversion_source_loop.data).all()
    assert (inversion_image_loop.noise_map == inversion_source_loop.noise_map).all()
    assert (
        inversion_image_loop.linear_obj_list[0]
        == inversion_source_loop.linear_obj_list[0]
    )
    assert (
        inversion_image_loop.regularization_list[0]
        == inversion_source_loop.regularization_list[0]
    )
    assert (
        inversion_image_loop.regularization_matrix
        == inversion_source_loop.regularization_matrix
    ).all()

    assert inversion_image_loop.curvature_matrix == pytest.approx(
        inversion_source_loop.curvature_matrix, 1.0e-8
    )
    assert inversion_image_loop.curvature_reg_matrix == pytest.approx(
        inversion_source_loop.curvature_reg_matrix, 1.0e-8
    )
    assert inversion_image_loop.reconstruction == pytest.approx(
        inversion_source_loop.reconstruction, 1.0e-2
    )
    assert inversion_image_loop.mapped_reconstructed_image == pytest.approx(
        inversion_source_loop.mapped_reconstructed_image, 1.0e-2
    )
    assert inversion_image_loop.mapped_reconstructed_data == pytest.approx(
        inversion_source_loop.mapped_reconstructed_data, 1.0e-2
    )
