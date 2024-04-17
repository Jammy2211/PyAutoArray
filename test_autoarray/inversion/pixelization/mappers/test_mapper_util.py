import autoarray as aa
import numpy as np
import pytest


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


def grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers):
    def compute_squared_separation(coordinate1, coordinate2):
        """
        Returns the squared separation of two grid (no square root for efficiency).
        """
        return (coordinate1[0] - coordinate2[0]) ** 2 + (
            coordinate1[1] - coordinate2[1]
        ) ** 2

    image_pixels = grid.shape[0]

    image_to_pixelization = np.zeros((image_pixels,))

    for image_index, image_coordinate in enumerate(grid):
        distances = list(
            map(
                lambda centers: compute_squared_separation(image_coordinate, centers),
                pixel_centers,
            )
        )

        image_to_pixelization[image_index] = np.argmin(distances)

    return image_to_pixelization


def _test__sub_slim_indexes_for_pix_index():
    pix_indexes_for_sub_slim_index = np.array(
        [[0, 4], [1, 4], [2, 4], [0, 4], [1, 4], [3, 4], [0, 4], [3, 4]]
    ).astype("int")
    pix_pixels = 5
    pix_weights_for_sub_slim_index = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
        ]
    )

    (
        sub_slim_indexes_for_pix_index,
        sub_slim_sizes_for_pix_index,
        sub_slim_weights_for_pix_index,
    ) = aa.util.mapper.sub_slim_indexes_for_pix_index(
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=pix_pixels,
    )

    assert (
        sub_slim_indexes_for_pix_index
        == np.array(
            [
                [0, 3, 6, -1, -1, -1, -1, -1],
                [1, 4, -1, -1, -1, -1, -1, -1],
                [2, -1, -1, -1, -1, -1, -1, -1],
                [5, 7, -1, -1, -1, -1, -1, -1],
                [0, 1, 2, 3, 4, 5, 6, 7],
            ]
        )
    ).all()
    assert (sub_slim_sizes_for_pix_index == np.array([3, 2, 1, 2, 8])).all()

    assert (
        sub_slim_weights_for_pix_index
        == np.array(
            [
                [0.1, 0.4, 0.7, -1, -1, -1, -1, -1],
                [0.2, 0.5, -1, -1, -1, -1, -1, -1],
                [0.3, -1, -1, -1, -1, -1, -1, -1],
                [0.6, 0.8, -1, -1, -1, -1, -1, -1],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            ]
        )
    ).all()


def test__mapping_matrix(three_pixels, five_pixels):
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    slim_index_for_sub_slim_index = np.array([0, 1, 2])

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((3, 1), dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(3, dtype="int"),
        pixels=6,
        total_mask_pixels=3,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([1.0, 1.0, 1.0]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [1, 0, 0, 0, 0, 0],  # Imaging pixel 0 maps to pix pixel 0.
                [0, 1, 0, 0, 0, 0],  # Imaging pixel 1 maps to pix pixel 1.
                [0, 0, 1, 0, 0, 0],
            ]
        )
    ).all()  # Imaging pixel 2 maps to pix pixel 2

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2], [7], [6]])
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4])

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(5, dtype="int"),
        pix_weights_for_sub_slim_index=np.ones((5, 1), dtype="int"),
        pixels=8,
        total_mask_pixels=5,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
    ).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]]
    ).T
    slim_index_for_sub_slim_index = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    )
    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((20, 1), dtype="int"),
        pix_size_for_sub_slim_index=np.ones(20, dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pixels=8,
        total_mask_pixels=5,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
                [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0],
                [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25],
            ]
        )
    ).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]]
    ).T
    slim_index_for_sub_slim_index = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    )

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((20, 1), dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(20, dtype="int"),
        pixels=8,
        total_mask_pixels=5,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [0.75, 0.25, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25],
            ]
        )
    ).all()

    pix_indexes_for_sub_slim_index = np.array(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
            ]
        ]
    ).T

    slim_index_for_sub_slim_index = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((48, 1), dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(48, dtype="int"),
        pixels=6,
        total_mask_pixels=3,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [0.75, 0.25, 0, 0, 0, 0],
                [0, 0, 1.0, 0, 0, 0],
                [0.1875, 0.1875, 0.1875, 0.1875, 0.125, 0.125],
            ]
        )
    ).all()


def test__data_to_pix_unique_from():
    image_pixels = 2
    sub_size = np.array([2, 2])

    pix_indexes_for_sub_slim_index = np.array(
        [[0, -1], [0, -1], [0, -1], [1, -1], [2, -1], [1, -1], [0, -1], [2, -1]]
    ).astype("int")
    pix_size_for_sub_slim_index = np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array(
        [
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
        ]
    )

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=image_pixels,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=sub_size,
    )

    assert (data_to_pix_unique[0, :] == np.array([0, 1, -1, -1])).all()
    assert (data_weights[0, :] == np.array([0.75, 0.25, 0.0, 0.0])).all()
    assert (data_to_pix_unique[1, :] == np.array([2, 1, 0, -1])).all()
    assert (data_weights[1, :] == np.array([0.5, 0.25, 0.25, 0.0])).all()
    assert (pix_lengths == np.array([2, 3])).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0, 1], [0, 1], [0, 2], [1, -1], [2, -1], [1, -1], [0, -1], [2, -1]]
    ).astype("int")
    pix_size_for_sub_slim_index = np.array([2, 2, 2, 1, 1, 1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.75],
            [0.75, 0.25],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
        ]
    )

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper.data_slim_to_pixelization_unique_from(
        data_pixels=image_pixels,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=sub_size,
    )

    assert (data_to_pix_unique[0, :] == np.array([0, 1, 2, -1, -1, -1, -1, -1])).all()
    assert (
        data_weights[0, :] == np.array([0.375, 0.5625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all()
    assert (data_to_pix_unique[1, :] == np.array([2, 1, 0, -1, -1, -1, -1, -1])).all()
    assert (
        data_weights[1, :] == np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all()
    assert (pix_lengths == np.array([3, 3])).all()


def test__weights():
    source_plane_data_grid = np.array([[0.1, 0.1], [1.0, 1.0]])

    source_plane_mesh_grid = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    slim_index_for_sub_slim_index = np.array([0, 1])

    pix_indexes_for_sub_slim_index = np.array([[0, 1, 2], [2, -1, -1]])

    pixel_weights = aa.util.mapper.pixel_weights_delaunay_from(
        source_plane_data_grid=source_plane_data_grid,
        source_plane_mesh_grid=source_plane_mesh_grid,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    )

    assert (pixel_weights == np.array([[0.25, 0.5, 0.25], [1.0, 0.0, 0.0]])).all()


def test__grid_to_pixel_pixels_via_nearest_neighbour(grid_2d_7x7):
    pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    grid = aa.Grid2D.no_mask(
        values=[[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]],
        shape_native=(2, 2),
        pixel_scales=1.0,
    )

    sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

    assert sub_to_pix[0] == 0
    assert sub_to_pix[1] == 1
    assert sub_to_pix[2] == 2
    assert sub_to_pix[3] == 3

    pixel_centers = np.array(
        [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]]
    )
    grid = aa.Grid2D.no_mask(
        [
            [0.1, 0.1],
            [-0.1, -0.1],
            [0.49, 0.49],
            [0.51, 0.51],
            [1.01, 1.01],
            [1.51, 1.51],
        ],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

    assert sub_to_pix[0] == 4
    assert sub_to_pix[1] == 4
    assert sub_to_pix[2] == 4
    assert sub_to_pix[3] == 0
    assert sub_to_pix[4] == 0
    assert sub_to_pix[5] == 5

    mesh_grid = aa.Grid2D.no_mask(
        [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
    )

    sub_to_pix_nearest_neighbour = np.array(
        [grid_to_pixel_pixels_via_nearest_neighbour(grid_2d_7x7, mesh_grid)]
    ).T

    mesh_grid = aa.Mesh2DVoronoi(
        values=mesh_grid,
    )

    mapper_grids = aa.MapperGrids(
        mask=grid_2d_7x7.mask,
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=mesh_grid,
    )

    over_sampler = aa.OverSamplerUniform(mask=grid_2d_7x7.mask, sub_size=1)

    mapper = aa.Mapper(
        mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
    )

    assert (mapper.pix_indexes_for_sub_slim_index == sub_to_pix_nearest_neighbour).all()


def test__adaptive_pixel_signals_from():
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pixel_weights = np.ones((3, 1), dtype="int")
    pixel_sizes = np.ones(3, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2])
    galaxy_image = np.array([1.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2], [0]])
    pixel_weights = np.ones((4, 1), dtype="int")
    pixel_sizes = np.ones(4, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 0])
    galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pixel_weights = np.ones((3, 1), dtype="int")
    pixel_sizes = np.ones(3, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2])
    galaxy_image = np.array([2.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 0.5, 0.5])).all()

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pixel_weights = np.ones((3, 1), dtype="int")
    pixel_sizes = np.ones(3, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2])
    galaxy_image = np.array([2.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=2.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 0.25, 0.25])).all()


def test_mapped_to_source_via_mapping_matrix_from():
    mapping_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    array_slim = np.array([1.0, 2.0, 3.0])

    mapped_to_source = aa.util.mapper.mapped_to_source_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix, array_slim=array_slim
    )

    assert (mapped_to_source == np.array([1.0, 2.5])).all()

    mapping_matrix = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.0, 0.5, 0.5],
            [0.0, 0.25, 0.75],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
        ]
    )

    array_slim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    mapped_to_source = aa.util.mapper.mapped_to_source_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix, array_slim=array_slim
    )


#    assert (mapped_to_source == np.array([3.5, 8.0, 3.5])).all()
