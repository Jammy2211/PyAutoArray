import autoarray as aa


class TestRectangular:
    def test__sparse_grid_from__returns_none_as_not_used(self, sub_grid_2d_7x7):

        pixelization = aa.pix.Rectangular(shape=(3, 3))

        assert (
            pixelization.data_pixelization_grid_from(data_grid_slim=sub_grid_2d_7x7)
            == None
        )

    def test__preloads_used_for_relocated_grid(self, sub_grid_2d_7x7):

        pixelization = aa.pix.Rectangular(shape=(3, 3))

        relocated_grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

        mapper = pixelization.mapper_from(
            source_grid_slim=relocated_grid,
            source_pixelization_grid=None,
            settings=aa.SettingsPixelization(use_border=True),
            preloads=aa.Preloads(relocated_grid=relocated_grid),
        )

        assert (mapper.source_grid_slim == relocated_grid).all()
