import numpy as np
import pytest

import autoarray as aa


def grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers):
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two grid (no square root for efficiency)"""
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


class TestRectangularMapper:
    def test__sub_to_pix__variuos_grids__1_coordinate_per_square_pixel__in_centre_of_pixels(
        self
    ):
        #   _ _ _
        #  I_I_I_I Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
        #  I_I_I_I Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
        #  I_I_I_I Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

        grid = aa.Grid.manual_1d(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 3),
        )

        pixelization_grid = aa.GridRectangular(
            grid=np.ones((2, 2)), shape_2d=(3, 3), pixel_scales=(1.0, 1.0)
        )

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ]

        #   _ _ _
        #  I_I_I_I Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
        #  I_I_I_I Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
        #  I_I_I_I Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

        grid = aa.Grid.manual_1d(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [-0.32, -1.0],
                [-0.32, 0.32],
                [0.0, 1.0],
                [-0.34, -0.34],
                [-0.34, 0.325],
                [-1.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 3),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ]

    def test__sub_to_pix__3x3_grid_of_pixel_grid__add_multiple_grid_to_1_pixel(self):
        #                  _ _ _
        # -1.0 to -(1/3)  I_I_I_I
        # -(1/3) to (1/3) I_I_I_I
        #  (1/3) to 1.0   I_I_I_I

        grid = aa.Grid.manual_1d(
            [
                [1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-1.0, -1.0],
                [0.0, 0.0],
                [-1.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 3),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [],
            [2],
            [],
            [1, 3, 4, 5, 7],
            [],
            [6],
            [],
            [8],
        ]

    def test__sub_to_pix__various_grids__1_coordinate_in_each_pixel(self):
        #   _ _ _
        #  I_I_I_I
        #  I_I_I_I
        #  I_I_I_I
        #  I_I_I_I

        # Boundaries for column pixel 0 -1.0 to -(1/3)
        # Boundaries for column pixel 1 -(1/3) to (1/3)
        # Boundaries for column pixel 2  (1/3) to 1.0

        # Bounadries for row pixel 0 -1.0 to -0.5
        # Bounadries for row pixel 1 -0.5 to 0.0
        # Bounadries for row pixel 2  0.0 to 0.5
        # Bounadries for row pixel 3  0.5 to 1.0

        grid = aa.Grid.manual_1d(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.5, -1.0],
                [-0.5, 1.0],
                [-1.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 2),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(4, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 3, 8, 11])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [3],
            [],
            [],
            [],
            [],
            [4],
            [],
            [],
            [5],
        ]

        #   _ _ _ _
        #  I_I_I_I_I
        #  I_I_I_I_I
        #  I_I_I_I_I

        # Boundaries for row pixel 0 -1.0 to -(1/3)
        # Boundaries for row pixel 1 -(1/3) to (1/3)
        # Boundaries for row pixel 2  (1/3) to 1.0

        # Bounadries for column pixel 0 -1.0 to -0.5
        # Bounadries for column pixel 1 -0.5 to 0.0
        # Bounadries for column pixel 2  0.0 to 0.5
        # Bounadries for column pixel 3  0.5 to 1.0

        grid = aa.Grid.manual_1d(
            [
                [1.0, -1.0],
                [1.0, -0.49],
                [1.0, 0.01],
                [0.32, 0.01],
                [-0.34, -0.01],
                [-1.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(2, 3),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 4), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 6, 9, 11])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [],
            [],
            [],
            [3],
            [],
            [],
            [4],
            [],
            [5],
        ]

    def test__sub_to_pix__3x3_grid__change_arcsecond_dimensions_size__grid_adapts_accordingly(
        self
    ):
        #   _ _ _
        #  I_I_I_I Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
        #  I_I_I_I Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
        #  I_I_I_I Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

        grid = aa.Grid.manual_1d(
            [[1.5, -1.5], [1.0, 0.0], [1.0, 0.6], [-1.4, 0.0], [-1.5, 1.5]],
            pixel_scales=1.0,
            shape_2d=(5, 1),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 7, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [],
            [],
            [],
            [],
            [3],
            [4],
        ]

    def test__sub_to_pix__various_grids__change_arcsecond_dimensions__not_symmetric(
        self
    ):
        #   _ _ _
        #  I_I_I_I Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
        #  I_I_I_I Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
        #  I_I_I_I Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

        grid = aa.Grid.manual_1d(
            [[1.0, -1.5], [1.0, -0.49], [0.32, -1.5], [0.32, 0.51], [-1.0, 1.5]],
            pixel_scales=1.0,
            shape_2d=(5, 1),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 3, 5, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [],
            [2],
            [],
            [3],
            [],
            [],
            [4],
        ]

        #   _ _ _
        #  I_I_I_I
        #  I_I_I_I
        #  I_I_I_I
        #  I_I_I_I

        grid = aa.Grid.manual_1d(
            [[1.0, -1.5], [1.0, -0.49], [0.49, -1.5], [-0.6, 0.0], [-1.0, 1.5]],
            pixel_scales=1.0,
            shape_2d=(5, 1),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(4, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 3, 10, 11])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [],
            [2],
            [],
            [],
            [],
            [],
            [],
            [],
            [3],
            [4],
        ]

        #   _ _ _ _
        #  I_I_I_I_I
        #  I_I_I_I_I
        #  I_I_I_I_I

        grid = aa.Grid.manual_1d(
            [[1.0, -1.5], [1.0, -0.49], [0.32, -1.5], [-0.34, 0.49], [-1.0, 1.5]],
            pixel_scales=1.0,
            shape_2d=(5, 1),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 4), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 4, 10, 11])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [],
            [],
            [2],
            [],
            [],
            [],
            [],
            [],
            [3],
            [4],
        ]

    def test__sub_to_pix__different_image_and_sub_grids(self):
        #                  _ _ _
        # -1.0 to -(1/3)  I_I_I_I
        # -(1/3) to (1/3) I_I_I_I
        #  (1/3) to 1.0   I_I_I_I

        grid = aa.Grid.manual_1d(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 3),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ]

    def test__sub_to_pix__3x3_grid_of_pixel_grid___shift_coordinates_to_new_centre__centre_adjusts_based_on_grid(
        self
    ):
        #   _ _ _
        #  I_I_I_I Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
        #  I_I_I_I Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
        #  I_I_I_I Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

        grid = aa.Grid.manual_1d(
            [
                [2.0, 0.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 3),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        ).all()

        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ]

    def test__sub_to_pix__4x3_grid__non_symmetric_centre_shift(self):
        #   _ _ _
        #  I_I_I_I
        #  I_I_I_I
        #  I_I_I_I
        #  I_I_I_I

        grid = aa.Grid.manual_1d(
            [[3.0, -0.5], [3.0, 0.51], [2.49, -0.5], [1.4, 1.0], [1.0, 2.5]],
            pixel_scales=1.0,
            shape_2d=(5, 1),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(4, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == np.array([0, 1, 3, 10, 11])
        ).all()
        assert mapper.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
            [0],
            [1],
            [],
            [2],
            [],
            [],
            [],
            [],
            [],
            [],
            [3],
            [4],
        ]

    def test__reconstructed_pixelization__3x3_pixelization__solution_vector_ascending(
        self
    ):
        grid = aa.Grid.manual_1d(
            [
                [2.0, -1.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-2.0, -1.0],
                [-2.0, 0.0],
                [-2.0, 1.0],
            ],
            pixel_scales=1.0,
            shape_2d=(3, 3),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        recon_pix = mapper.reconstructed_pixelization_from_solution_vector(
            solution_vector=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        )

        assert (
            recon_pix.in_2d
            == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        ).all()
        assert recon_pix.pixel_scales == pytest.approx((4.0 / 3.0, 2.0 / 3.0), 1e-2)
        assert recon_pix.origin == (0.0, 0.0)

    def test__reconstructed_pixelization__compare_to_imaging_util(self):

        grid = aa.Grid.manual_1d(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            pixel_scales=1.0,
            shape_2d=(2, 2),
        )

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(4, 3), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        solution = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
        )
        recon_pix = mapper.reconstructed_pixelization_from_solution_vector(
            solution_vector=solution
        )
        recon_pix_util = aa.util.array.sub_array_2d_from(
            sub_array_1d=solution,
            mask=np.full(fill_value=False, shape=(4, 3)),
            sub_size=1,
        )
        assert (recon_pix.in_2d == recon_pix_util).all()
        assert recon_pix.shape_2d == (4, 3)

        pixelization_grid = aa.GridRectangular.overlay_grid(shape_2d=(3, 4), grid=grid)

        mapper = aa.Mapper(grid=grid, pixelization_grid=pixelization_grid)

        solution = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
        )
        recon_pix = mapper.reconstructed_pixelization_from_solution_vector(
            solution_vector=solution
        )
        recon_pix_util = aa.util.array.sub_array_2d_from(
            sub_array_1d=solution,
            mask=np.full(fill_value=False, shape=(3, 4)),
            sub_size=1,
        )
        assert (recon_pix.in_2d == recon_pix_util).all()
        assert recon_pix.shape_2d == (3, 4)

    def test__pixel_signals__compare_to_mapper_util(self, grid_7x7, image_7x7):
        pixelization_grid = aa.GridRectangular.overlay_grid(
            shape_2d=(3, 3), grid=grid_7x7
        )

        mapper = aa.Mapper(
            grid=grid_7x7, pixelization_grid=pixelization_grid, hyper_image=image_7x7
        )

        pixel_signals = mapper.pixel_signals_from_signal_scale(signal_scale=2.0)

        pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=9,
            signal_scale=2.0,
            pixelization_1d_index_for_sub_mask_1d_index=mapper.pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=grid_7x7.regions._mask_1d_index_for_sub_mask_1d_index,
            hyper_image=image_7x7,
        )

        assert (pixel_signals == pixel_signals_util).all()

    def test__image_from_source__different_types_of_lists_input(self, sub_grid_7x7):

        rectangular_pixelization_grid = aa.GridRectangular.overlay_grid(
            grid=sub_grid_7x7, shape_2d=(3, 3)
        )
        rectangular_mapper = aa.Mapper(
            grid=sub_grid_7x7, pixelization_grid=rectangular_pixelization_grid
        )

        image_pixel_indexes = rectangular_mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=[0, 1]
        )

        assert image_pixel_indexes == [0, 1, 2, 3, 4, 5, 6, 7]

        image_pixel_indexes = rectangular_mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=[[0], [4]]
        )

        assert image_pixel_indexes == [[0, 1, 2, 3], [16, 17, 18, 19]]


class TestVoronoiMapper:
    def test__grid_to_pixel_pixels_via_nearest_neighbour__case1__correct_pairs(self):
        pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        grid = aa.Grid.manual_1d(
            [[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]],
            shape_2d=(2, 2),
            pixel_scales=1.0,
        )

        sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

        assert sub_to_pix[0] == 0
        assert sub_to_pix[1] == 1
        assert sub_to_pix[2] == 2
        assert sub_to_pix[3] == 3

    def test__grid_to_pixel_pixels_via_nearest_neighbour___case2__correct_pairs(self):
        pixel_centers = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        grid = aa.Grid.manual_1d(
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
            shape_2d=(3, 3),
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

    def test__grid_to_pixel_pixels_via_nearest_neighbour___case3__correct_pairs(self):
        pixel_centers = np.array(
            [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 0.0], [2.0, 2.0]]
        )
        grid = aa.Grid.manual_1d(
            [
                [0.1, 0.1],
                [-0.1, -0.1],
                [0.49, 0.49],
                [0.51, 0.51],
                [1.01, 1.01],
                [1.51, 1.51],
            ],
            shape_2d=(3, 2),
            pixel_scales=1.0,
        )

        sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

        assert sub_to_pix[0] == 4
        assert sub_to_pix[1] == 4
        assert sub_to_pix[2] == 4
        assert sub_to_pix[3] == 0
        assert sub_to_pix[4] == 0
        assert sub_to_pix[5] == 5

    def test__sub_to_pix_of_mapper_matches_nearest_neighbor_calculation(self, grid_7x7):
        pixelization_grid = aa.Grid.manual_1d(
            [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]],
            shape_2d=(3, 2),
            pixel_scales=1.0,
        )

        sub_to_pix_nearest_neighbour = grid_to_pixel_pixels_via_nearest_neighbour(
            grid_7x7, pixelization_grid
        )

        nearest_pixelization_1d_index_for_mask_1d_index = np.array(
            [0, 0, 1, 0, 0, 1, 2, 2, 3]
        )

        pixelization_grid = aa.GridVoronoi(
            grid=pixelization_grid,
            nearest_pixelization_1d_index_for_mask_1d_index=nearest_pixelization_1d_index_for_mask_1d_index,
        )

        mapper = aa.Mapper(grid=grid_7x7, pixelization_grid=pixelization_grid)

        assert (
            mapper.pixelization_1d_index_for_sub_mask_1d_index
            == sub_to_pix_nearest_neighbour
        ).all()

    def test__pixel_scales___for_voronoi_mapper(self, grid_7x7, image_7x7):
        pixelization_grid = aa.Grid.manual_1d(
            [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]],
            shape_2d=(3, 2),
            pixel_scales=1.0,
        )

        nearest_pixelization_1d_index_for_mask_1d_index = np.array(
            [0, 0, 1, 0, 0, 1, 2, 2, 3]
        )

        pixelization_grid = aa.GridVoronoi(
            grid=pixelization_grid,
            nearest_pixelization_1d_index_for_mask_1d_index=nearest_pixelization_1d_index_for_mask_1d_index,
        )

        mapper = aa.Mapper(
            grid=grid_7x7, pixelization_grid=pixelization_grid, hyper_image=image_7x7
        )

        pixel_signals = mapper.pixel_signals_from_signal_scale(signal_scale=2.0)

        pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=6,
            signal_scale=2.0,
            pixelization_1d_index_for_sub_mask_1d_index=mapper.pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=grid_7x7.regions._mask_1d_index_for_sub_mask_1d_index,
            hyper_image=image_7x7,
        )

        assert (pixel_signals == pixel_signals_util).all()
