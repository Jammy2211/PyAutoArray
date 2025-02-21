import autoarray as aa
import numpy as np
import pytest


def test__curvature_matrix_from_w_tilde():
    w_tilde = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )

    mapping_matrix = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_w_tilde_from(
        w_tilde=w_tilde, mapping_matrix=mapping_matrix
    )

    assert (
        curvature_matrix
        == np.array([[6.0, 8.0, 0.0], [8.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()


def test__curvature_matrix_via_mapping_matrix_from():
    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    ).all()

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    noise_map = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[1.25, 0.25, 0.0], [0.25, 2.25, 1.0], [0.0, 1.0, 1.0]])
    ).all()


def test__reconstruction_positive_negative_from():
    data_vector = np.array([1.0, 1.0, 2.0])

    curvature_reg_matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])

    reconstruction = aa.util.inversion.reconstruction_positive_negative_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
        mapper_param_range_list=[[0, 3]],
    )

    assert reconstruction == pytest.approx(np.array([1.0, -1.0, 3.0]), 1.0e-4)


def test__reconstruction_positive_negative_from__check_solution_raises_error_cause_all_values_identical():
    data_vector = np.array([1.0, 1.0, 1.0])

    curvature_reg_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # reconstruction = np.array([1.0, 1.0, 1.0])

    with pytest.raises(aa.exc.InversionException):
        aa.util.inversion.reconstruction_positive_negative_from(
            data_vector=data_vector,
            curvature_reg_matrix=curvature_reg_matrix,
            mapper_param_range_list=[[0, 3]],
            force_check_reconstruction=True,
        )


def test__mapped_reconstructed_data_via_mapping_matrix_from():
    mapping_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_data == np.array([1.0, 1.0, 2.0])).all()

    mapping_matrix = np.array([[0.25, 0.50, 0.25], [0.0, 1.0, 0.0], [0.0, 0.25, 0.75]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_data == np.array([1.25, 1.0, 1.75])).all()


def test__mapped_reconstructed_data_via_image_to_pix_unique_from():
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pix_indexes_for_sub_slim_index_sizes = np.array([1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array([[1.0], [1.0], [1.0]])

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=np.array([1, 1, 1]),
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            reconstruction=reconstruction,
        )
    )

    assert (mapped_reconstructed_data == np.array([1.0, 1.0, 2.0])).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0], [1], [1], [2], [1], [1], [1], [1], [1], [2], [2], [2]]
    )
    pix_indexes_for_sub_slim_index_sizes = np.ones(shape=(12,)).astype("int")
    pix_weights_for_sub_slim_index = np.ones(shape=(12, 1))

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=np.array([2, 2, 2]),
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_data = (
        aa.util.inversion.mapped_reconstructed_data_via_image_to_pix_unique_from(
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            reconstruction=reconstruction,
        )
    )

    assert (mapped_reconstructed_data == np.array([1.25, 1.0, 1.75])).all()


def test__preconditioner_matrix_via_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=1.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    ).all()

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    ).all()

    regularization_matrix = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=regularization_matrix,
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[5.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 13.0]])
    ).all()


def test__inversion_interferometer__via_mapper(
    interferometer_7_no_fft,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
    regularization_constant,
):
    inversion = aa.Inversion(
        dataset=interferometer_7_no_fft,
        linear_obj_list=[rectangular_mapper_7x7_3x3],
        settings=aa.SettingsInversion(use_linear_operators=True),
    )

    assert isinstance(inversion.linear_obj_list[0], aa.MapperRectangular)
    assert isinstance(inversion, aa.InversionInterferometerMappingPyLops)


def test__w_tilde_curvature_interferometer_from():
    noise_map = np.array([1.0, 2.0, 3.0])
    uv_wavelengths = np.array([[0.0001, 2.0, 3000.0], [3000.0, 2.0, 0.0001]])

    grid = aa.Grid2D.uniform(shape_native=(2, 2), pixel_scales=0.0005)

    w_tilde = aa.util.inversion.w_tilde_curvature_interferometer_from(
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

    w_tilde = aa.util.inversion.w_tilde_curvature_interferometer_from(
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
        aa.util.inversion.w_tilde_curvature_preload_interferometer_from(
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

    curvature_matrix_via_preload = aa.util.inversion.curvature_matrix_via_w_tilde_curvature_preload_interferometer_from(
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

    w_tilde = aa.util.inversion.w_tilde_curvature_interferometer_from(
        noise_map_real=np.array(noise_map),
        uv_wavelengths=np.array(uv_wavelengths),
        grid_radians_slim=np.array(grid),
    )

    w_tilde_preload = (
        aa.util.inversion.w_tilde_curvature_preload_interferometer_from(
            noise_map_real=np.array(noise_map),
            uv_wavelengths=np.array(uv_wavelengths),
            shape_masked_pixels_2d=(3, 3),
            grid_radians_2d=np.array(grid.native),
        )
    )

    native_index_for_slim_index = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    )

    w_tilde_via_preload = aa.util.inversion.w_tilde_via_preload_from(
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

    assert inversion_w_tilde.curvature_matrix == pytest.approx(
        inversion_mapping_matrices.curvature_matrix, 1.0e-8
    )
    assert inversion_w_tilde.curvature_reg_matrix == pytest.approx(
        inversion_mapping_matrices.curvature_reg_matrix, 1.0e-8
    )
    assert inversion_w_tilde.reconstruction == pytest.approx(
        inversion_mapping_matrices.reconstruction, 1.0e-2
    )
    assert inversion_w_tilde.mapped_reconstructed_image == pytest.approx(
        inversion_mapping_matrices.mapped_reconstructed_image, 1.0e-2
    )
    assert inversion_w_tilde.mapped_reconstructed_data == pytest.approx(
        inversion_mapping_matrices.mapped_reconstructed_data, 1.0e-2
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

