from autoarray.structures import grids
import autoarray as aa
import numpy as np


class TestRectangular:
    class TestGridIrregular:
        def test__pixelization_grid_returns_none_as_not_used(self, sub_grid_7x7):

            pix = aa.pix.Rectangular(shape=(3, 3))

            assert pix.sparse_grid_from_grid(grid=sub_grid_7x7) == None


class TestVoronoiMagnification:
    def test__number_of_pixels_setup_correct(self):

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))

        assert pix.shape == (3, 3)

    def test__pixelization_grid_returns_same_as_computed_from_grids_module(
        self, sub_grid_7x7
    ):

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))

        pixelization_grid = pix.sparse_grid_from_grid(grid=sub_grid_7x7)

        pixelization_grid_manual = grids.GridIrregular.from_grid_and_unmasked_2d_grid_shape(
            unmasked_sparse_shape=(3, 3), grid=sub_grid_7x7
        )

        assert (pixelization_grid_manual == pixelization_grid).all()
        assert (
            pixelization_grid_manual.nearest_irregular_1d_index_for_mask_1d_index
            == pixelization_grid.nearest_irregular_1d_index_for_mask_1d_index
        ).all()


class TestVoronoiBrightness:
    def test__hyper_image_doesnt_use_min_and_max_weight_map_uses_floor_and_power(self):

        hyper_image = np.array([0.0, 1.0, 0.0])

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=0.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.ones(3)).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([0.0, 1.0, 0.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([1.0, 2.0, 1.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=2.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([1.0, 4.0, 1.0])).all()

    def test__hyper_image_uses_min_and_max__weight_map_uses_floor_and_power(self):

        hyper_image = np.array([-1.0, 1.0, 3.0])

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([0.0, 0.5, 1.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=0.0, weight_power=2.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([0.0, 0.25, 1.0])).all()

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=5, weight_floor=1.0, weight_power=1.0
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        assert (weight_map == np.array([3.0, 3.5, 4.0])).all()

    def test__pixelization_grid_returns_same_as_computed_from_grids_module(
        self, sub_grid_7x7
    ):

        pix = aa.pix.VoronoiBrightnessImage(
            pixels=6, weight_floor=0.1, weight_power=2.0
        )

        hyper_image = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        pixelization_grid = pix.sparse_grid_from_grid(
            grid=sub_grid_7x7, hyper_image=hyper_image, seed=1
        )

        weight_map = pix.weight_map_from_hyper_image(hyper_image=hyper_image)

        sparse_grid = grids.SparseGrid.from_total_pixels_grid_and_weight_map(
            total_pixels=pix.pixels, grid=sub_grid_7x7, weight_map=weight_map, seed=1
        )

        pixelization_grid_manual = grids.GridIrregular(
            grid=sparse_grid.sparse,
            nearest_irregular_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

        assert (pixelization_grid_manual == pixelization_grid).all()
        assert (
            pixelization_grid_manual.nearest_irregular_1d_index_for_mask_1d_index
            == pixelization_grid.nearest_irregular_1d_index_for_mask_1d_index
        ).all()
