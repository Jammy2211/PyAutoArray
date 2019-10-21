from autoarray.structures import grids
import autoarray as aa
import numpy as np
import pytest
import scipy.spatial


class TestRectangular:
    class TestGeometry:
        def test__3x3_grid__buffer_is_small__grid_give_min_minus_1_max_1__sets_up_geometry_correctly(
            self
        ):

            pix = aa.pix.Rectangular(shape=(3, 3))

            pixelization_grid = np.array(
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
                ]
            )

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.shape_2d == (3, 3)
            assert geometry.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)
            assert (geometry.pixel_neighbors[0] == [1, 3, -1, -1]).all()
            assert (geometry.pixel_neighbors[1] == [0, 2, 4, -1]).all()
            assert (geometry.pixel_neighbors[2] == [1, 5, -1, -1]).all()
            assert (geometry.pixel_neighbors[3] == [0, 4, 6, -1]).all()
            assert (geometry.pixel_neighbors[4] == [1, 3, 5, 7]).all()
            assert (geometry.pixel_neighbors[5] == [2, 4, 8, -1]).all()
            assert (geometry.pixel_neighbors[6] == [3, 7, -1, -1]).all()
            assert (geometry.pixel_neighbors[7] == [4, 6, 8, -1]).all()
            assert (geometry.pixel_neighbors[8] == [5, 7, -1, -1]).all()

            assert (geometry.pixel_neighbors_size == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()

        def test__3x3_grid__same_as_above_change_buffer(self):
            pix = aa.pix.Rectangular(shape=(3, 3))

            pixelization_grid = np.array(
                [
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ]
            )

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-4)

            assert geometry.shape_2d == (3, 3)
            assert geometry.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)

        def test__5x4_grid__buffer_is_small(self):

            pix = aa.pix.Rectangular(shape=(5, 4))

            pixelization_grid = np.array(
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
                ]
            )

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.shape_2d == (5, 4)
            assert geometry.pixel_scales == pytest.approx((2.0 / 5.0, 2.0 / 4.0), 1e-2)

        def test__3x3_grid__larger_range_of_grid(self):

            pix = aa.pix.Rectangular(shape=(3, 3))

            pixelization_grid = np.array(
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]]
            )

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.shape_2d == (3, 3)
            assert geometry.pixel_scales == pytest.approx((6.0 / 3.0, 6.0 / 3.0), 1e-2)

    class TestPixelCentres:
        def test__3x3_grid__pixel_centres(self):

            pix = aa.pix.Rectangular(shape=(3, 3))

            pixelization_grid = np.array(
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
                ]
            )

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.pixel_centres == pytest.approx(
                np.array(
                    [
                        [2.0 / 3.0, -2.0 / 3.0],
                        [2.0 / 3.0, 0.0],
                        [2.0 / 3.0, 2.0 / 3.0],
                        [0.0, -2.0 / 3.0],
                        [0.0, 0.0],
                        [0.0, 2.0 / 3.0],
                        [-2.0 / 3.0, -2.0 / 3.0],
                        [-2.0 / 3.0, 0.0],
                        [-2.0 / 3.0, 2.0 / 3.0],
                    ]
                )
            )

        def test__4x3_grid__pixel_centres(self):

            pix = aa.pix.Rectangular(shape=(4, 3))

            pixelization_grid = np.array(
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
                ]
            )

            geometry = pix.geometry_from_grid(pixelization_grid, buffer=1e-8)

            assert geometry.pixel_centres == pytest.approx(
                np.array(
                    [
                        [0.75, -2.0 / 3.0],
                        [0.75, 0.0],
                        [0.75, 2.0 / 3.0],
                        [0.25, -2.0 / 3.0],
                        [0.25, 0.0],
                        [0.25, 2.0 / 3.0],
                        [-0.25, -2.0 / 3.0],
                        [-0.25, 0.0],
                        [-0.25, 2.0 / 3.0],
                        [-0.75, -2.0 / 3.0],
                        [-0.75, 0.0],
                        [-0.75, 2.0 / 3.0],
                    ]
                )
            )

    class TestPixelNeighbors:
        def test__compare_to_pixelization_util(self):
            # |0 | 1| 2| 3|
            # |4 | 5| 6| 7|
            # |8 | 9|10|11|
            # |12|13|14|15|

            pix = aa.pix.Rectangular(shape=(7, 5))

            pixel_neighbors, pixel_neighbors_size = pix.pixel_neighbors
            pixel_neighbors_util, pixel_neighbors_size_util = aa.util.pixelization.rectangular_neighbors_from_shape(
                shape=(7, 5)
            )

            assert (pixel_neighbors == pixel_neighbors_util).all()
            assert (pixel_neighbors_size == pixel_neighbors_size_util).all()

    class TestPixelizationGrid:
        def test__pixelization_grid_returns_none_as_not_used(self, sub_grid_7x7):

            pix = aa.pix.Rectangular(shape=(3, 3))

            assert pix.pixelization_grid_from_grid(grid=sub_grid_7x7) == None


class TestVoronoi:
    class TestVoronoiGrid:
        def test__9_points___check_voronoi_swaps_axis_from_y_x__to_x_y(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array(
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
                ]
            )

            pix = aa.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            assert (
                voronoi.points
                == np.array(
                    [
                        [0.0, 2.0],
                        [1.0, 2.0],
                        [2.0, 2.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [2.0, 1.0],
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                    ]
                )
            ).all()

        def test__points_in_x_cross_shape__sets_up_diamond_voronoi_vertices(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array(
                [[-1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]]
            )

            pix = aa.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0, 1.0] in voronoi.vertices
            assert [-1.0, 0.0] in voronoi.vertices
            assert [1.0, 0.0] in voronoi.vertices
            assert [0.0, -1.0] in voronoi.vertices

        def test__9_points_in_square___sets_up_square_of_voronoi_vertices(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array(
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
                ]
            )

            pix = aa.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.vertices = list(map(lambda x: list(x), voronoi.vertices))

            assert [0.5, 1.5] in voronoi.vertices
            assert [1.5, 0.5] in voronoi.vertices
            assert [0.5, 0.5] in voronoi.vertices
            assert [1.5, 1.5] in voronoi.vertices

        def test__points_in_x_cross_shape__sets_up_pairs_of_voronoi_cells(self):
            # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

            points = np.array(
                [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
            )

            pix = aa.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 8

            assert [2, 0] in voronoi.ridge_points or [0, 2] in voronoi.ridge_points
            assert [2, 1] in voronoi.ridge_points or [1, 2] in voronoi.ridge_points
            assert [2, 3] in voronoi.ridge_points or [3, 2] in voronoi.ridge_points
            assert [2, 4] in voronoi.ridge_points or [4, 2] in voronoi.ridge_points
            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [0.3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 1] in voronoi.ridge_points or [1, 4] in voronoi.ridge_points

        def test__9_points_in_square___sets_up_pairs_of_voronoi_cells(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array(
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
                ]
            )

            pix = aa.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)

            # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
            # to look for each list

            voronoi.ridge_points = list(map(lambda x: list(x), voronoi.ridge_points))

            assert len(voronoi.ridge_points) == 12

            assert [0, 1] in voronoi.ridge_points or [1, 0] in voronoi.ridge_points
            assert [1, 2] in voronoi.ridge_points or [2, 1] in voronoi.ridge_points
            assert [3, 4] in voronoi.ridge_points or [4, 3] in voronoi.ridge_points
            assert [4, 5] in voronoi.ridge_points or [5, 4] in voronoi.ridge_points
            assert [6, 7] in voronoi.ridge_points or [7, 6] in voronoi.ridge_points
            assert [7, 8] in voronoi.ridge_points or [8, 7] in voronoi.ridge_points

            assert [0, 3] in voronoi.ridge_points or [3, 0] in voronoi.ridge_points
            assert [1, 4] in voronoi.ridge_points or [4, 1] in voronoi.ridge_points
            assert [4, 7] in voronoi.ridge_points or [7, 4] in voronoi.ridge_points
            assert [2, 5] in voronoi.ridge_points or [5, 2] in voronoi.ridge_points
            assert [5, 8] in voronoi.ridge_points or [8, 5] in voronoi.ridge_points
            assert [3, 6] in voronoi.ridge_points or [6, 3] in voronoi.ridge_points

    class TestNeighbors:
        def test__compare_to_pixelization_util(self):
            # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

            points = np.array(
                [
                    [3.0, 0.0],
                    [2.0, 1.0],
                    [2.0, 2.0],
                    [8.0, 3.0],
                    [1.0, 3.0],
                    [1.0, 9.0],
                    [6.0, 31.0],
                    [0.0, 2.0],
                    [3.0, 5.0],
                ]
            )

            pix = aa.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(points)
            pixel_neighbors, pixel_neighbors_size = pix.neighbors_from_pixels_and_ridge_points(
                pixels=9, ridge_points=voronoi.ridge_points
            )

            voronoi = scipy.spatial.Voronoi(
                np.asarray([points[:, 1], points[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
            )
            pixel_neighbors_util, pixel_neighbors_size_util = aa.util.pixelization.voronoi_neighbors_from_pixels_and_ridge_points(
                pixels=9, ridge_points=np.array(voronoi.ridge_points)
            )

            assert (pixel_neighbors == pixel_neighbors_util).all()
            assert (pixel_neighbors_size == pixel_neighbors_size_util).all()


class TestVoronoiMagnification:
    def test__number_of_pixels_setup_correct(self):

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))

        assert pix.shape == (3, 3)

    def test__pixelization_grid_returns_same_as_computed_from_grids_module(
        self, sub_grid_7x7
    ):

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))

        pixelization_grid = pix.pixelization_grid_from_grid(grid=sub_grid_7x7)

        pixelization_grid_manual = grids.PixelizationGrid.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(3, 3), grid=sub_grid_7x7
        )

        assert (pixelization_grid_manual == pixelization_grid).all()
        assert (
            pixelization_grid_manual.nearest_pixelization_1d_index_for_mask_1d_index
            == pixelization_grid.nearest_pixelization_1d_index_for_mask_1d_index
        ).all()


class TestVoronoiBrightness:
    def test__hyper_image_doesnt_use_min_and_max_weight_map_uses_floor_and_power(
        self
    ):

        hyper_image = np.array([0.0, 1.0, 0.0])

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=0.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.ones(3)).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.array([0.0, 1.0, 0.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.array([1.0, 2.0, 1.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=2.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.array([1.0, 4.0, 1.0])).all()

    def test__hyper_image_uses_min_and_max__weight_map_uses_floor_and_power(
        self
    ):

        hyper_image = np.array([-1.0, 1.0, 3.0])

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.array([0.0, 0.5, 1.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=2.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.array([0.0, 0.25, 1.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        assert (weight_map == np.array([3.0, 3.5, 4.0])).all()

    def test__pixelization_grid_returns_same_as_computed_from_grids_module(
        self, sub_grid_7x7,
    ):

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=6, weight_floor=0.1, weight_power=2.0
        )

        hyper_image = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        pixelization_grid = pix.pixelization_grid_from_grid(
            grid=sub_grid_7x7,
            hyper_image=hyper_image,
            seed=1,
        )

        weight_map = pix.weight_map_from_hyper_image(
            hyper_image=hyper_image
        )

        sparse_to_grid = grids.SparseToGrid.from_total_pixels_grid_and_weight_map(
            total_pixels=pix.pixels,
            grid=sub_grid_7x7,
            weight_map=weight_map,
            seed=1,
        )

        pixelization_grid_manual = grids.PixelizationGrid(
            grid_1d=sparse_to_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )

        assert (pixelization_grid_manual == pixelization_grid).all()
        assert (
            pixelization_grid_manual.nearest_pixelization_1d_index_for_mask_1d_index
            == pixelization_grid.nearest_pixelization_1d_index_for_mask_1d_index
        ).all()
