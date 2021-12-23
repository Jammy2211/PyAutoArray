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
        Returns the squared separation of two grid (no square root for efficiency)"""
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


class TestMappingMatrix:
    def test__3_image_pixels__6_pixel_pixels__sub_grid_1x1(self, three_pixels):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([[0], [1], [2]])
        slim_index_for_sub_slim_index = np.array([0, 1, 2])

        mapping_matrix = aa.util.mapper.mapping_matrix_from(
            pix_weights_for_sub_slim_index=np.ones((3, 1), dtype="int"),
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=np.ones(3, dtype="int"),
            pixels=6,
            total_mask_sub_pixels=3,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_fraction=1.0,
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

    def test__5_image_pixels__8_pixel_pixels__sub_grid_1x1(self, five_pixels):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
            [[0], [1], [2], [7], [6]]
        )
        slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4])

        mapping_matrix = aa.util.mapper.mapping_matrix_from(
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_weights_for_sub_slim_index=np.ones((5, 1), dtype="int"),
            pix_size_for_sub_slim_index=np.ones(3, dtype="int"),
            pixels=8,
            total_mask_sub_pixels=5,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_fraction=1.0,
        )

        assert (
            mapping_matrix
            == np.array(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],  # Imaging image_to_pixel 0 and 3 mappers to pix pixel 0.
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],  # Imaging image_to_pixel 1 and 4 mappers to pix pixel 1.
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )
        ).all()  # Imaging image_to_pixel 2 and 5 mappers to pix pixel 2

    def test__5_image_pixels__8_pixel_pixels__sub_grid_2x2__no_overlapping_pixels(
        self, five_pixels
    ):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
            [[0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]]
        ).T
        slim_index_for_sub_slim_index = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        )
        mapping_matrix = aa.util.mapper.mapping_matrix_from(
            pix_weights_for_sub_slim_index=np.ones((20, 1), dtype="int"),
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=np.ones(20, dtype="int"),
            pixels=8,
            total_mask_sub_pixels=5,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_fraction=0.25,
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

    def test__5_image_pixels__8_pixel_pixels__sub_grid_2x2__include_overlapping_pixels(
        self, five_pixels
    ):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
            [[0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]]
        ).T
        slim_index_for_sub_slim_index = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        )

        mapping_matrix = aa.util.mapper.mapping_matrix_from(
            pix_weights_for_sub_slim_index=np.ones((20, 1), dtype="int"),
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=np.ones(20, dtype="int"),
            pixels=8,
            total_mask_sub_pixels=5,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_fraction=0.25,
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

    def test__3_image_pixels__6_pixel_pixels__sub_grid_4x4(self, three_pixels):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
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
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=np.ones(48, dtype="int"),
            pixels=6,
            total_mask_sub_pixels=3,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            sub_fraction=1.0 / 16.0,
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


class TestDataToPixUnique:
    def test__data_to_pix_unique_from(self):

        image_pixels = 2
        sub_size = 2

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

        data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
            data_pixels=image_pixels,
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_indexes_for_sub_slim_sizes=pix_size_for_sub_slim_index,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
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

        data_to_pix_unique, data_weights, pix_lengths = aa.util.mapper.data_slim_to_pixelization_unique_from(
            data_pixels=image_pixels,
            pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
            pix_indexes_for_sub_slim_sizes=pix_size_for_sub_slim_index,
            pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
            sub_size=sub_size,
        )

        assert (
            data_to_pix_unique[0, :] == np.array([0, 1, 2, -1, -1, -1, -1, -1])
        ).all()
        assert (
            data_weights[0, :]
            == np.array([0.375, 0.5625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0])
        ).all()
        assert (
            data_to_pix_unique[1, :] == np.array([2, 1, 0, -1, -1, -1, -1, -1])
        ).all()
        assert (
            data_weights[1, :] == np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
        ).all()
        assert (pix_lengths == np.array([3, 3])).all()


class TestPixelizationIndexesVoronoi:
    def test__grid_to_pixel_pixels_via_nearest_neighbour(self):
        pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        grid = aa.Grid2D.manual_slim(
            [[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]],
            shape_native=(2, 2),
            pixel_scales=1.0,
        )

        sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

        assert sub_to_pix[0] == 0
        assert sub_to_pix[1] == 1
        assert sub_to_pix[2] == 2
        assert sub_to_pix[3] == 3

        pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        grid = aa.Grid2D.manual_slim(
            [
                [1.1, 1.1],
                [-1.1, 1.1],
                [-1.1, -1.1],
                [1.1, -1.1],
                [0.9, -0.9],
                [-0.9, -0.9],
                [-0.9, 0.9],
                [0.9, 0.9],
            ],
            shape_native=(3, 3),
            pixel_scales=0.1,
        )

        sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

        assert sub_to_pix[0] == 0
        assert sub_to_pix[1] == 1
        assert sub_to_pix[2] == 2
        assert sub_to_pix[3] == 3
        assert sub_to_pix[4] == 3
        assert sub_to_pix[5] == 2
        assert sub_to_pix[6] == 1
        assert sub_to_pix[7] == 0

        pixel_centers = np.array(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]]
        )
        grid = aa.Grid2D.manual_slim(
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

    def test__pixelization_index_for_voronoi_sub_slim_index_from__matches_nearest_neighbor_calculation(
        self, grid_2d_7x7
    ):
        pixelization_grid = aa.Grid2D.manual_slim(
            [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]],
            shape_native=(3, 2),
            pixel_scales=1.0,
        )

        sub_to_pix_nearest_neighbour = np.array(
            [grid_to_pixel_pixels_via_nearest_neighbour(grid_2d_7x7, pixelization_grid)]
        ).T

        nearest_pixelization_index_for_slim_index = np.array(
            [0, 0, 1, 0, 0, 1, 2, 2, 3]
        )

        pixelization_grid = aa.Grid2DVoronoi(
            grid=pixelization_grid,
            nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,
        )

        mapper = aa.Mapper(
            source_grid_slim=grid_2d_7x7, source_pixelization_grid=pixelization_grid
        )

        assert (
            mapper.pix_indexes_for_sub_slim_index.mappings
            == sub_to_pix_nearest_neighbour
        ).all()


class TestPixelSignals:
    def test__x3_image_pixels_signals_1s__pixel_scale_1__pixel_signals_all_1s(self):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([[0], [1], [2]])
        pixel_weights = np.ones((3, 1), dtype="int")
        pixel_sizes = np.ones(3, dtype="int")
        slim_index_for_sub_slim_index = np.array([0, 1, 2])
        galaxy_image = np.array([1.0, 1.0, 1.0])

        pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=3,
            signal_scale=1.0,
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=pixel_sizes,
            pixel_weights=pixel_weights,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    def test__x4_image_pixels_signals_1s__pixel_signals_still_all_1s(self):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([[0], [1], [2], [0]])
        pixel_weights = np.ones((4, 1), dtype="int")
        pixel_sizes = np.ones(4, dtype="int")
        slim_index_for_sub_slim_index = np.array([0, 1, 2, 0])
        galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

        pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=3,
            signal_scale=1.0,
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=pixel_sizes,
            pixel_weights=pixel_weights,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    def test__galaxy_flux_in_a_pixel_pixel_is_double_the_others__pixel_signal_is_1_others_a_half(
        self,
    ):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([[0], [1], [2]])
        pixel_weights = np.ones((3, 1), dtype="int")
        pixel_sizes = np.ones(3, dtype="int")
        slim_index_for_sub_slim_index = np.array([0, 1, 2])
        galaxy_image = np.array([2.0, 1.0, 1.0])

        pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=3,
            signal_scale=1.0,
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=pixel_sizes,
            pixel_weights=pixel_weights,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 0.5, 0.5])).all()

    def test__same_as_above_but_pixel_scale_2__scales_pixel_signals(self):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([[0], [1], [2]])
        pixel_weights = np.ones((3, 1), dtype="int")
        pixel_sizes = np.ones(3, dtype="int")
        slim_index_for_sub_slim_index = np.array([0, 1, 2])
        galaxy_image = np.array([2.0, 1.0, 1.0])

        pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=3,
            signal_scale=2.0,
            pix_indexes_for_sub_slim_index=pixelization_1d_index_for_sub_mask_1d_index,
            pix_size_for_sub_slim_index=pixel_sizes,
            pixel_weights=pixel_weights,
            slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 0.25, 0.25])).all()
