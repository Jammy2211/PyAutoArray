import autoarray as aa
import numpy as np
import pytest


def test__psf_weighted_noise_imaging_from():
    noise_map = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    psf_weighted_noise = aa.util.inversion_imaging_numba.psf_precision_operator_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert psf_weighted_noise == pytest.approx(
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


def test__psf_weighted_data_from():

    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    data = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask,
    )

    noise_map = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask,
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    weight_map = data / (noise_map**2)
    weight_map = aa.Array2D(values=weight_map, mask=mask)

    psf_weighted_data = aa.util.inversion_imaging.psf_weighted_data_from(
        weight_map_native=weight_map.native.array,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert (psf_weighted_data == np.array([5.0, 5.0, 1.5, 1.5])).all()


def test__psf_precision_operator_sparse_from():
    noise_map = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    (
        psf_weighted_noise_preload,
        psf_weighted_noise_indexes,
        psf_weighted_noise_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert psf_weighted_noise_preload == pytest.approx(
        np.array(
            [1.25, 1.625, 0.5, 0.375, 0.65625, 0.125, 0.0625, 0.25, 0.375, 0.15625]
        ),
        1.0e-4,
    )
    assert psf_weighted_noise_indexes == pytest.approx(
        np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]), 1.0e-4
    )

    assert psf_weighted_noise_lengths == pytest.approx(np.array([4, 3, 2, 1]), 1.0e-4)


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


def test__data_vector_via_weighted_data_two_methods_agree():
    mask = aa.Mask2D.circular(shape_native=(51, 51), pixel_scales=0.1, radius=2.0)

    image = np.random.uniform(size=mask.shape_native)
    image = aa.Array2D(values=image, mask=mask)

    noise_map = np.random.uniform(size=mask.shape_native)
    noise_map = aa.Array2D(values=noise_map, mask=mask)

    kernel = aa.Kernel2D.from_gaussian(
        shape_native=(7, 7), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
    )

    psf = kernel

    pixelization = aa.mesh.RectangularUniform(shape=(20, 20))

    # TODO : Use pytest.parameterize

    for sub_size in range(1, 3):

        print(sub_size)

        grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=sub_size)

        mapper_grids = pixelization.mapper_grids_from(
            mask=mask,
            border_relocator=None,
            source_plane_data_grid=grid,
        )

        mapper = aa.Mapper(
            mapper_grids=mapper_grids,
            regularization=None,
        )

        mapping_matrix = mapper.mapping_matrix

        blurred_mapping_matrix = psf.convolved_mapping_matrix_from(
            mapping_matrix=mapping_matrix, mask=mask
        )

        data_vector = (
            aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
                blurred_mapping_matrix=blurred_mapping_matrix,
                image=image,
                noise_map=noise_map,
            )
        )

        rows, cols, vals = aa.util.mapper.sparse_triplets_from(
            pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
            pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
            slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
            fft_index_for_masked_pixel=mask.fft_index_for_masked_pixel,
            sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
        )

        weight_map = image.array / (noise_map.array**2)
        weight_map = aa.Array2D(values=weight_map, mask=noise_map.mask)

        psf_weighted_data = aa.util.inversion_imaging.psf_weighted_data_from(
            weight_map_native=weight_map.native.array,
            kernel_native=kernel.native.array,
            native_index_for_slim_index=mask.derive_indexes.native_for_slim.astype(
                "int"
            ),
        )

        data_vector_via_psf_weighted_noise = (
            aa.util.inversion_imaging.data_vector_via_psf_weighted_data_from(
                psf_weighted_data=psf_weighted_data,
                rows=rows,
                cols=cols,
                vals=vals,
                S=pixelization.pixels,
            )
        )

        assert data_vector_via_psf_weighted_noise == pytest.approx(data_vector, 1.0e-4)


def test__curvature_matrix_via_psf_weighted_noise_two_methods_agree():

    mask = aa.Mask2D.circular(shape_native=(51, 51), pixel_scales=0.1, radius=2.0)

    noise_map = np.random.uniform(size=mask.shape_native)
    noise_map = aa.Array2D(values=noise_map, mask=mask)

    kernel = aa.Kernel2D.from_gaussian(
        shape_native=(7, 7), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
    )

    psf = kernel

    sparse_operator = aa.ImagingSparseOperator.from_noise_map_and_psf(
        data=noise_map,
        noise_map=noise_map,
        psf=psf.native,
    )

    mesh = aa.mesh.RectangularAdaptDensity(shape=(20, 20))

    mapper_grids = mesh.mapper_grids_from(
        mask=mask,
        border_relocator=None,
        source_plane_data_grid=mask.derive_grid.unmasked,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    mapping_matrix = mapper.mapping_matrix

    rows, cols, vals = aa.util.mapper.sparse_triplets_from(
        pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
        pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
        fft_index_for_masked_pixel=mask.fft_index_for_masked_pixel,
        sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
        return_rows_slim=False,
    )

    curvature_matrix_via_sparse_operator = sparse_operator.curvature_matrix_diag_from(
        rows,
        cols,
        vals,
        S=mesh.shape[0] * mesh.shape[1],
    )

    curvature_matrix_via_sparse_operator = (
        aa.util.inversion_imaging.curvature_matrix_mirrored_from(
            curvature_matrix=curvature_matrix_via_sparse_operator,
        )
    )

    blurred_mapping_matrix = psf.convolved_mapping_matrix_from(
        mapping_matrix=mapping_matrix, mask=mask
    )

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix,
        noise_map=noise_map,
    )

    assert curvature_matrix_via_sparse_operator == pytest.approx(
        curvature_matrix, rel=1.0e-3
    )
