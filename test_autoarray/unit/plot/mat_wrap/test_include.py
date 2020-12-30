import autoarray as aa
import autoarray.plot as aplt


class TestInclude2d:
    def test__loads_default_values_from_config_if_not_input(self):

        include = aplt.Include2D()

        assert include.origin == True
        assert include.mask == True
        assert include.border == True
        assert include.parallel_overscan == True
        assert include.serial_prescan == True
        assert include.serial_overscan == False

        include = aplt.Include2D(origin=False, border=False, serial_overscan=True)

        assert include.origin == False
        assert include.mask == True
        assert include.border == False
        assert include.parallel_overscan == True
        assert include.serial_prescan == True
        assert include.serial_overscan == True

    def test__visuals_from_array(self, array_7x7):

        include = aplt.Include2D(origin=True, mask=True, border=True)

        visuals = include.visuals_from_array(array=array_7x7)

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.mask == array_7x7.mask).all()
        assert (
            visuals.border == array_7x7.mask.geometry.border_grid_sub_1.in_1d_binned
        ).all()

        include = aplt.Include2D(origin=False, mask=False, border=False)

        visuals = include.visuals_from_array(array=array_7x7)

        assert visuals.origin == None
        assert visuals.mask == None
        assert visuals.border == None

    def test__visuals_from_frame(self, frame_7x7):

        include = aplt.Include2D(origin=True, mask=True, border=True)

        visuals = include.visuals_from_frame(frame=frame_7x7)

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.mask == frame_7x7.mask).all()
        assert (
            visuals.border == frame_7x7.mask.geometry.border_grid_sub_1.in_1d_binned
        ).all()

        include = aplt.Include2D(origin=False, mask=False, border=False)

        visuals = include.visuals_from_frame(frame=frame_7x7)

        assert visuals.origin == None
        assert visuals.mask == None
        assert visuals.border == None

    def test__visuals_from_grid(self, grid_7x7):

        include = aplt.Include2D(origin=True, mask=True, border=True)

        visuals = include.visuals_from_grid(grid=grid_7x7)

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.mask == grid_7x7.mask).all()
        assert (
            visuals.border == grid_7x7.mask.geometry.border_grid_sub_1.in_1d_binned
        ).all()

        include = aplt.Include2D(origin=False, mask=False, border=False)

        visuals = include.visuals_from_grid(grid=grid_7x7)

        assert visuals.origin == None
        assert visuals.mask == None
        assert visuals.border == None

    def test__visuals_for_data_from_rectangular_mapper(
        self, rectangular_mapper_7x7_3x3
    ):
        include = aplt.Include2D(
            origin=True, mask=True, mapper_data_pixelization_grid=True, border=True
        )

        visuals = include.visuals_of_data_from_mapper(mapper=rectangular_mapper_7x7_3x3)

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.mask == rectangular_mapper_7x7_3x3.source_full_grid.mask).all()
        assert visuals.grid == None
        #  assert visuals.border == (0, 2)

        include = aplt.Include2D(
            origin=False, mask=False, mapper_data_pixelization_grid=False, border=False
        )

        visuals = include.visuals_of_data_from_mapper(mapper=rectangular_mapper_7x7_3x3)

        assert visuals.origin == None
        assert visuals.mask == None
        assert visuals.grid == None
        assert visuals.border == None

    def test__visuals_for_data_from_voronoi_mapper(self, voronoi_mapper_9_3x3):

        include = aplt.Include2D(
            origin=True, mask=True, mapper_data_pixelization_grid=True, border=True
        )

        visuals = include.visuals_of_data_from_mapper(mapper=voronoi_mapper_9_3x3)

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.mask == voronoi_mapper_9_3x3.source_full_grid.mask).all()
        assert (
            visuals.pixelization_grid
            == aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=0.1)
        ).all()
        #      assert visuals.border.shape == (0, 2)

        include = aplt.Include2D(
            origin=False, mask=False, mapper_data_pixelization_grid=False, border=False
        )

        visuals = include.visuals_of_data_from_mapper(mapper=voronoi_mapper_9_3x3)

        assert visuals.origin == None
        assert visuals.mask == None
        assert visuals.grid == None
        assert visuals.pixelization_grid == None
        assert visuals.border == None

    def test__visuals_for_source_from_rectangular_mapper(
        self, rectangular_mapper_7x7_3x3
    ):

        include = aplt.Include2D(
            origin=True,
            mapper_source_full_grid=True,
            mapper_source_pixelization_grid=True,
            mapper_source_border=True,
        )

        visuals = include.visuals_of_source_from_mapper(
            mapper=rectangular_mapper_7x7_3x3
        )

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.grid == rectangular_mapper_7x7_3x3.source_full_grid).all()
        assert (
            visuals.pixelization_grid
            == rectangular_mapper_7x7_3x3.source_pixelization_grid
        ).all()
        #    assert visuals.border.shape == (0, 2)

        include = aplt.Include2D(
            origin=False,
            mapper_source_full_grid=False,
            mapper_source_pixelization_grid=False,
            mapper_source_border=False,
        )

        visuals = include.visuals_of_source_from_mapper(
            mapper=rectangular_mapper_7x7_3x3
        )

        assert visuals.origin == None
        assert visuals.grid == None
        assert visuals.pixelization_grid == None
        assert visuals.border == None

    def test__visuals_for_source_from_voronoi_mapper(self, voronoi_mapper_9_3x3):

        include = aplt.Include2D(
            origin=True,
            mapper_source_full_grid=True,
            mapper_source_pixelization_grid=True,
            mapper_source_border=True,
        )

        visuals = include.visuals_of_source_from_mapper(mapper=voronoi_mapper_9_3x3)

        assert visuals.origin.in_1d_list == [(0.0, 0.0)]
        assert (visuals.grid == voronoi_mapper_9_3x3.source_full_grid).all()
        assert (
            visuals.pixelization_grid == voronoi_mapper_9_3x3.source_pixelization_grid
        ).all()
        #      assert visuals.border.shape == (0, 2)

        include = aplt.Include2D(
            origin=False,
            mapper_source_pixelization_grid=False,
            mapper_source_border=False,
        )

        visuals = include.visuals_of_source_from_mapper(mapper=voronoi_mapper_9_3x3)

        assert visuals.origin == None
        assert visuals.grid == None
        assert visuals.border == None
