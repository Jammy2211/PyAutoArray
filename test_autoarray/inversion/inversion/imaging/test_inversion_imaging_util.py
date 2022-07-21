import autoarray as aa
import numpy as np
import pytest


def test__w_tilde_imaging_from():

    noise_map_2d = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    w_tilde = aa.util.inversion_imaging.w_tilde_curvature_imaging_from(
        noise_map_native=noise_map_2d,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert w_tilde == pytest.approx(
        np.array(
            [
                [2.5, 1.625, 0.5, 0.375],
                [1.625, 1.3125, 0.125, 0.0625],
                [0.5, 0.125, 0.5, 0.375],
                [0.375, 0.0625, 0.375, 0.3125],
            ]
        ),
        1.0e-4,
    )


def test__w_tilde_data_imaging_from():

    image_2d = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    noise_map_2d = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    w_tilde_data = aa.util.inversion_imaging.w_tilde_data_imaging_from(
        image_native=image_2d,
        noise_map_native=noise_map_2d,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert (w_tilde_data == np.array([5.0, 5.0, 1.5, 1.5])).all()


def test__w_tilde_curvature_preload_imaging_from():

    noise_map_2d = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    w_tilde_preload, w_tilde_indexes, w_tilde_lengths = aa.util.inversion_imaging.w_tilde_curvature_preload_imaging_from(
        noise_map_native=noise_map_2d,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert w_tilde_preload == pytest.approx(
        np.array(
            [1.25, 1.625, 0.5, 0.375, 0.65625, 0.125, 0.0625, 0.25, 0.375, 0.15625]
        ),
        1.0e-4,
    )
    assert w_tilde_indexes == pytest.approx(
        np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]), 1.0e-4
    )

    assert w_tilde_lengths == pytest.approx(np.array([4, 3, 2, 1]), 1.0e-4)


def test__data_vector_via_blurred_mapping_matrix_from():

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

    image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix, image=image, noise_map=noise_map
    )

    assert (data_vector == np.array([2.0, 3.0, 1.0])).all()

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

    image = np.array([3.0, 1.0, 1.0, 10.0, 1.0, 1.0])
    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix, image=image, noise_map=noise_map
    )

    assert (data_vector == np.array([4.0, 14.0, 10.0])).all()

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

    image = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix, image=image, noise_map=noise_map
    )

    assert (data_vector == np.array([2.0, 3.0, 1.0])).all()


def test__data_vector_via_w_tilde_data_two_methods_agree():

    mask = aa.Mask2D.circular(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=1, radius=2.0
    )

    image = np.random.uniform(size=mask.shape_native)
    image = aa.Array2D.manual_mask(array=image, mask=mask)

    noise_map = np.random.uniform(size=mask.shape_native)
    noise_map = aa.Array2D.manual_mask(array=noise_map, mask=mask)

    kernel = aa.Kernel2D.from_gaussian(
        shape_native=(7, 7), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
    )

    convolver = aa.Convolver(mask=mask, kernel=kernel)

    pixelization = aa.pix.Rectangular(shape=(20, 20))

    for sub_size in range(1, 3):

        mask_sub = mask.mask_new_sub_size_from(mask=mask, sub_size=sub_size)

        grid = aa.Grid2D.from_mask(mask=mask_sub)

        mapper = pixelization.mapper_from(source_grid_slim=grid)

        mapping_matrix = mapper.mapping_matrix

        blurred_mapping_matrix = convolver.convolve_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=blurred_mapping_matrix,
            image=image,
            noise_map=noise_map,
        )

        w_tilde_data = aa.util.inversion_imaging.w_tilde_data_imaging_from(
            image_native=image.native,
            noise_map_native=noise_map.native,
            kernel_native=kernel.native,
            native_index_for_slim_index=mask.native_index_for_slim_index,
        )

        data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
            data_pixels=w_tilde_data.shape[0],
            pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
            sub_size=sub_size,
        )

        data_vector_via_w_tilde = aa.util.inversion_imaging.data_vector_via_w_tilde_data_imaging_from(
            w_tilde_data=w_tilde_data,
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            pix_pixels=pixelization.pixels,
        )

        # print(data_vector_via_w_tilde)
        # print(data_vector)

        assert data_vector_via_w_tilde == pytest.approx(data_vector, 1.0e-4)


def test__curvature_matrix_via_w_tilde_two_methods_agree():

    mask = aa.Mask2D.circular(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=1, radius=2.0
    )

    noise_map = np.random.uniform(size=mask.shape_native)
    noise_map = aa.Array2D.manual_mask(array=noise_map, mask=mask)

    kernel = aa.Kernel2D.from_gaussian(
        shape_native=(7, 7), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
    )

    convolver = aa.Convolver(mask=mask, kernel=kernel)

    pixelization = aa.pix.Rectangular(shape=(20, 20))

    mapper = pixelization.mapper_from(source_grid_slim=mask.masked_grid_sub_1)

    mapping_matrix = mapper.mapping_matrix

    w_tilde = aa.util.inversion_imaging.w_tilde_curvature_imaging_from(
        noise_map_native=noise_map.native,
        kernel_native=kernel.native,
        native_index_for_slim_index=mask.native_index_for_slim_index,
    )

    curvature_matrix_via_w_tilde = aa.util.inversion.curvature_matrix_via_w_tilde_from(
        w_tilde=w_tilde, mapping_matrix=mapping_matrix
    )

    blurred_mapping_matrix = convolver.convolve_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )
    assert curvature_matrix_via_w_tilde == pytest.approx(curvature_matrix, 1.0e-4)


def test__curvature_matrix_via_w_tilde_preload_two_methods_agree():

    mask = aa.Mask2D.circular(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=1, radius=2.0
    )

    noise_map = np.random.uniform(size=mask.shape_native)
    noise_map = aa.Array2D.manual_mask(array=noise_map, mask=mask)

    kernel = aa.Kernel2D.from_gaussian(
        shape_native=(7, 7), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
    )

    convolver = aa.Convolver(mask=mask, kernel=kernel)

    pixelization = aa.pix.Rectangular(shape=(20, 20))

    for sub_size in range(1, 2, 3):

        mask_sub = mask.mask_new_sub_size_from(mask=mask, sub_size=sub_size)

        grid = aa.Grid2D.from_mask(mask=mask_sub)

        mapper = pixelization.mapper_from(source_grid_slim=grid)

        mapping_matrix = mapper.mapping_matrix

        w_tilde_preload, w_tilde_indexes, w_tilde_lengths = aa.util.inversion_imaging.w_tilde_curvature_preload_imaging_from(
            noise_map_native=noise_map.native,
            kernel_native=kernel.native,
            native_index_for_slim_index=mask.native_index_for_slim_index,
        )

        data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
            data_pixels=w_tilde_lengths.shape[0],
            pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
            pix_sizes_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
            pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,
            sub_size=sub_size,
        )

        curvature_matrix_via_w_tilde = aa.util.inversion_imaging.curvature_matrix_via_w_tilde_curvature_preload_imaging_from(
            curvature_preload=w_tilde_preload,
            curvature_indexes=w_tilde_indexes.astype("int"),
            curvature_lengths=w_tilde_lengths.astype("int"),
            data_to_pix_unique=data_to_pix_unique.astype("int"),
            data_weights=data_weights,
            pix_lengths=pix_lengths.astype("int"),
            pix_pixels=pixelization.pixels,
        )

        blurred_mapping_matrix = convolver.convolve_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
            mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
        )

        assert curvature_matrix_via_w_tilde == pytest.approx(curvature_matrix, 1.0e-4)
