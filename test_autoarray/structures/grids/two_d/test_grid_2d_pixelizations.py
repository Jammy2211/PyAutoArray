import numpy as np
import pytest
import scipy.spatial

from autoarray import exc
import autoarray as aa


class TestGrid2DRectangular:
    def test__pixel_neighbors__compare_to_pixelization_util(self):
        # I0 I 1I 2I 3I
        # I4 I 5I 6I 7I
        # I8 I 9I10I11I
        # I12I13I14I15I

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(7, 5), grid=np.zeros((2, 2)), buffer=1e-8
        )

        (
            pixel_neighbors_util,
            pixel_neighbors_sizes_util,
        ) = aa.util.pixelization.rectangular_neighbors_from(shape_native=(7, 5))

        assert (pixelization_grid.pixel_neighbors == pixel_neighbors_util).all()
        assert (
            pixelization_grid.pixel_neighbors.sizes == pixel_neighbors_sizes_util
        ).all()

    def test__shape_native_and_pixel_scales(self):
        grid = np.array(
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

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(3, 3), grid=grid, buffer=1e-8
        )

        assert pixelization_grid.shape_native == (3, 3)
        assert pixelization_grid.pixel_scales == pytest.approx(
            (2.0 / 3.0, 2.0 / 3.0), 1e-2
        )

        grid = np.array(
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

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(5, 4), grid=grid, buffer=1e-8
        )

        assert pixelization_grid.shape_native == (5, 4)
        assert pixelization_grid.pixel_scales == pytest.approx(
            (2.0 / 5.0, 2.0 / 4.0), 1e-2
        )

        grid = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(3, 3), grid=grid, buffer=1e-8
        )

        assert pixelization_grid.shape_native == (3, 3)
        assert pixelization_grid.pixel_scales == pytest.approx(
            (6.0 / 3.0, 6.0 / 3.0), 1e-2
        )

    def test__pixel_centres__3x3_grid__pixel_centres(self):

        grid = np.array(
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

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(3, 3), grid=grid, buffer=1e-8
        )

        assert pixelization_grid == pytest.approx(
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

    def test__pixel_centres__4x3_grid__pixel_centres(self):
        grid = np.array(
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

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(4, 3), grid=grid, buffer=1e-8
        )

        assert pixelization_grid == pytest.approx(
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


class TestGrid2DVoronoi:
    def test__pixel_neighbors__compare_to_pixelization_util(self):

        # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

        grid = np.array(
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

        pix = aa.Grid2DVoronoi(grid=grid)

        voronoi = scipy.spatial.Voronoi(
            np.asarray([grid[:, 1], grid[:, 0]]).T, qhull_options="Qbb Qc Qx Qm"
        )

        (
            pixel_neighbors_util,
            pixel_neighbors_sizes_util,
        ) = aa.util.pixelization.voronoi_neighbors_from(
            pixels=9, ridge_points=np.array(voronoi.ridge_points)
        )

        assert (pix.pixel_neighbors == pixel_neighbors_util).all()
        assert (pix.pixel_neighbors.sizes == pixel_neighbors_sizes_util).all()

    def test__pixelization_areas(self):

        grid = np.array(
            [
                [-2.0, 0.0],
                [-np.sqrt(2), np.sqrt(2)],
                [0.0, 0.0],
                [0.0, 2.0],
                [np.sqrt(2), np.sqrt(2)],
                [2.0, 0.0],
                [np.sqrt(2), -np.sqrt(2)],
                [0.0, -2.0],
                [-np.sqrt(2), -np.sqrt(2)],
            ]
        )

        pix = aa.Grid2DVoronoi(grid=grid)

        assert pix.pixel_areas == pytest.approx(
            np.array(
                [
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                ]
            ),
            1e-6,
        )

    def test__pixelization_grid__attributes(self):

        pixelization_grid = aa.Grid2DVoronoi(
            grid=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 4.0]]),
            nearest_pixelization_index_for_slim_index=np.array([0, 1, 2, 3]),
        )

        assert type(pixelization_grid) == aa.Grid2DVoronoi
        assert (
            pixelization_grid
            == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 4.0]])
        ).all()
        assert (
            pixelization_grid.nearest_pixelization_index_for_slim_index
            == np.array([0, 1, 2, 3])
        ).all()

    def test__from_unmasked_sparse_shape_and_grid(self):

        mask = aa.Mask2D.manual(
            mask=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=(0.5, 0.5),
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        sparse_grid = aa.Grid2DSparse.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(10, 10), grid=grid
        )

        pixelization_grid = aa.Grid2DVoronoi(
            grid=sparse_grid,
            nearest_pixelization_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
        )

        assert (sparse_grid == pixelization_grid).all()
        assert (
            sparse_grid.sparse_index_for_slim_index
            == pixelization_grid.nearest_pixelization_index_for_slim_index
        ).all()

    def test__voronoi_grid__simple_shapes_make_voronoi_grid_correctly(self):

        # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

        grid = np.array(
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

        pix = aa.Grid2DVoronoi(grid=grid)

        assert (
            pix.voronoi.points
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

        # 5 points in the shape of the face of a 5 on a die - makes a diamond Voronoi diagram

        grid = np.array(
            [[-1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]]
        )

        pix = aa.Grid2DVoronoi(grid=grid)

        pix.voronoi.vertices = list(map(lambda x: list(x), pix.voronoi.vertices))

        assert [0, 1.0] in pix.voronoi.vertices
        assert [-1.0, 0.0] in pix.voronoi.vertices
        assert [1.0, 0.0] in pix.voronoi.vertices
        assert [0.0, -1.0] in pix.voronoi.vertices

        # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

        grid = np.array(
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

        pix = aa.Grid2DVoronoi(grid=grid)

        # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
        # to look for each list

        pix.voronoi.vertices = list(map(lambda x: list(x), pix.voronoi.vertices))

        assert [0.5, 1.5] in pix.voronoi.vertices
        assert [1.5, 0.5] in pix.voronoi.vertices
        assert [0.5, 0.5] in pix.voronoi.vertices
        assert [1.5, 1.5] in pix.voronoi.vertices

        # 9 points in a square - makes a square (this is the example int he scipy documentaiton page)

        grid = np.array(
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

        pix = aa.Grid2DVoronoi(grid=grid)

        # ridge points is a numpy array for speed, but convert to list for the comparisons below so we can use in
        # to look for each list

        pix.voronoi.ridge_grid = list(map(lambda x: list(x), pix.voronoi.ridge_points))

        assert len(pix.voronoi.ridge_points) == 12

        assert [0, 1] in pix.voronoi.ridge_points or [1, 0] in pix.voronoi.ridge_points
        assert [1, 2] in pix.voronoi.ridge_points or [2, 1] in pix.voronoi.ridge_points
        assert [3, 4] in pix.voronoi.ridge_points or [4, 3] in pix.voronoi.ridge_points
        assert [4, 5] in pix.voronoi.ridge_points or [5, 4] in pix.voronoi.ridge_points
        assert [6, 7] in pix.voronoi.ridge_points or [7, 6] in pix.voronoi.ridge_points
        assert [7, 8] in pix.voronoi.ridge_points or [8, 7] in pix.voronoi.ridge_points

        assert [0, 3] in pix.voronoi.ridge_points or [3, 0] in pix.voronoi.ridge_points
        assert [1, 4] in pix.voronoi.ridge_points or [4, 1] in pix.voronoi.ridge_points
        assert [4, 7] in pix.voronoi.ridge_points or [7, 4] in pix.voronoi.ridge_points
        assert [2, 5] in pix.voronoi.ridge_points or [5, 2] in pix.voronoi.ridge_points
        assert [5, 8] in pix.voronoi.ridge_points or [8, 5] in pix.voronoi.ridge_points
        assert [3, 6] in pix.voronoi.ridge_points or [6, 3] in pix.voronoi.ridge_points

    def test__qhull_error_is_caught(self):

        grid = np.array([[3.0, 0.0]])
        grid = aa.Grid2DVoronoi(grid=grid)

        with pytest.raises(exc.PixelizationException):
            grid.voronoi


class TestGrid2DDelaunay:
    def test__pixelization_areas(self):

        grid = np.array(
            [
                [-2.0, 0.0],
                [-np.sqrt(2), np.sqrt(2)],
                [0.0, 0.0],
                [0.0, 2.0],
                [np.sqrt(2), np.sqrt(2)],
                [2.0, 0.0],
                [np.sqrt(2), -np.sqrt(2)],
                [0.0, -2.0],
                [-np.sqrt(2), -np.sqrt(2)],
            ]
        )

        pix = aa.Grid2DDelaunay(grid=grid)

        assert pix.pixel_areas == pytest.approx(
            np.array(
                [
                    -0.1372583,
                    -0.1372583,
                    #1.0 * np.tan(22.5 / 180.0 * np.pi) * 0.5 * 16.0,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                    -0.1372583,
                ]
            ),
            1e-6,
        )
