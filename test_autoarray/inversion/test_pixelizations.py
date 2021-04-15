import autoarray as aa
import numpy as np


class TestSettingsPixelization:
    def test__settings_with_is_stochastic_true(self):

        settings = aa.SettingsPixelization(is_stochastic=False)
        settings = settings.settings_with_is_stochastic_true()
        assert settings.is_stochastic is True

        settings = aa.SettingsPixelization(is_stochastic=True)
        settings = settings.settings_with_is_stochastic_true()
        assert settings.is_stochastic is True


class TestRectangular:
    def test__pixelization_grid_returns_none_as_not_used(self, sub_grid_2d_7x7):

        pixelization = aa.pix.Rectangular(shape=(3, 3))

        assert pixelization.sparse_grid_from_grid(grid=sub_grid_2d_7x7) == None


class TestVoronoiMagnification:
    def test__number_of_pixels_setup_correct(self):

        pixelization = aa.pix.VoronoiMagnification(shape=(3, 3))

        assert pixelization.shape == (3, 3)

    def test__pixelization_grid_returns_same_as_computed_from_grids_module(
        self, sub_grid_2d_7x7
    ):

        pixelization = aa.pix.VoronoiMagnification(shape=(3, 3))

        sparse_grid = pixelization.sparse_grid_from_grid(grid=sub_grid_2d_7x7)

        pixelization_grid = aa.Grid2DVoronoi(
            grid=sparse_grid,
            nearest_pixelization_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
        )

        assert (pixelization_grid == sparse_grid).all()
        assert (
            pixelization_grid.nearest_pixelization_index_for_slim_index
            == sparse_grid.sparse_index_for_slim_index
        ).all()


class TestVoronoiBrightness:
    def test__hyper_image_doesnt_use_min_and_max_weight_map_uses_floor_and_power(self):

        hyper_image = np.array([0.0, 1.0, 0.0])

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=0.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.ones(3)).all()

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=1.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([0.0, 1.0, 0.0])).all()

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=1.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([1.0, 2.0, 1.0])).all()

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=2.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([1.0, 4.0, 1.0])).all()

    def test__hyper_image_uses_min_and_max__weight_map_uses_floor_and_power(self):

        hyper_image = np.array([-1.0, 1.0, 3.0])

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=1.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([0.0, 0.5, 1.0])).all()

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=2.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([0.0, 0.25, 1.0])).all()

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=1.0
        )

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([3.0, 3.5, 4.0])).all()

    def test__pixelization_grid_returns_same_as_computed_from_grids_module(
        self, sub_grid_2d_7x7
    ):

        pixelization = aa.pix.VoronoiBrightnessImage(
            pixels=6, weight_floor=0.1, weight_power=2.0
        )

        hyper_image = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        weight_map = pixelization.weight_map_from_hyper_image(hyper_image=hyper_image)

        sparse_grid = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
            total_pixels=pixelization.pixels,
            grid=sub_grid_2d_7x7,
            weight_map=weight_map,
            seed=1,
        )

        pixelization_grid = aa.Grid2DVoronoi(
            grid=sparse_grid,
            nearest_pixelization_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
        )

        assert (pixelization_grid == sparse_grid).all()
        assert (
            pixelization_grid.nearest_pixelization_index_for_slim_index
            == sparse_grid.sparse_index_for_slim_index
        ).all()


class TestRegression:
    def test__grid_is_relocated_via_border(self, sub_grid_2d_7x7):
        pixelization = aa.pix.VoronoiMagnification(shape=(3, 3))

        mask = aa.Mask2D.circular(
            shape_native=(60, 60),
            radius=1.0,
            pixel_scales=(0.1, 0.1),
            centre=(1.0, 1.0),
            sub_size=1,
        )

        grid = aa.Grid2D.from_mask(mask=mask)

        sparse_grid = pixelization.sparse_grid_from_grid(grid=grid)

        grid[8, 0] = 100.0

        mapper = pixelization.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=sparse_grid,
            settings=aa.SettingsPixelization(use_border=True),
        )

        assert grid[8, 0] != mapper.source_grid_slim[8, 0]
        assert mapper.source_grid_slim[8, 0] < 5.0

        grid[0, 0] = 0.0
        sparse_grid[0, 0] = 100.0

        mapper = pixelization.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=sparse_grid,
            settings=aa.SettingsPixelization(use_border=True),
        )

        assert sparse_grid[0, 0] != mapper.source_pixelization_grid[0, 0]
        assert mapper.source_pixelization_grid[0, 0] < 5.0
