import autoarray as aa
import autoarray.plot as aplt
from os import path
import pytest
import numpy as np
import shutil

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "structures"
    )


class TestArrayPlotter:
    def test___visuals_in_constructor_use_array_and_include(self, array_7x7):

        visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

        include = aplt.Include2D(origin=True, mask=True, border=True)

        array_plotter = aplt.ArrayPlotter(
            array=array_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert array_plotter.visuals_2d.origin == (1.0, 1.0)
        assert array_plotter.visuals_with_include_2d.origin == (1.0, 1.0)

        assert array_plotter.visuals_2d.mask == None
        assert (array_plotter.visuals_with_include_2d.mask == array_7x7.mask).all()

        assert array_plotter.visuals_2d.border == None
        assert (
            array_plotter.visuals_with_include_2d.border
            == array_7x7.mask.border_grid_sub_1.in_1d_binned
        ).all()

        assert array_plotter.visuals_2d.vector_field == 2
        assert array_plotter.visuals_with_include_2d.vector_field == 2

        include = aplt.Include2D(origin=False, mask=False, border=False)

        array_plotter = aplt.ArrayPlotter(
            array=array_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert array_plotter.visuals_with_include_2d.origin == (1.0, 1.0)
        assert array_plotter.visuals_with_include_2d.mask == None
        assert array_plotter.visuals_with_include_2d.border == None
        assert array_plotter.visuals_with_include_2d.vector_field == 2

    def test__works_with_all_extras_included(
        self,
        array_7x7,
        mask_7x7,
        grid_7x7,
        grid_irregular_grouped_7x7,
        plot_path,
        plot_patch,
    ):

        array_plotter = aplt.ArrayPlotter(
            array=array_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array1", format="png")
            ),
        )

        array_plotter.figure()

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        array_plotter = aplt.ArrayPlotter(
            array=array_7x7,
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array2", format="png")
            ),
        )

        array_plotter.figure()

        assert path.join(plot_path, "array2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_irregular_grouped_7x7,
            mask=mask_7x7,
            border=mask_7x7.border_grid_sub_1.in_1d_binned,
            grid=grid_7x7,
            positions=grid_irregular_grouped_7x7,
            lines=grid_irregular_grouped_7x7,
            array_overlay=array_7x7,
        )

        array_plotter = aplt.ArrayPlotter(
            array=array_7x7,
            visuals_2d=visuals_2d,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        array_plotter.figure()

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, array_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        array_plotter = aplt.ArrayPlotter(
            array=array_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array", format="fits")
            ),
        )

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        array_plotter.figure()

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == array_7x7.in_2d).all()


class TestFramePlotter:
    def test___visuals_in_constructor_use_frame_and_include(self, frame_7x7):

        visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

        include = aplt.Include2D(
            origin=True, mask=True, border=True, parallel_overscan=True
        )

        frame_plotter = aplt.FramePlotter(
            frame=frame_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert frame_plotter.visuals_2d.origin == (1.0, 1.0)
        assert frame_plotter.visuals_with_include_2d.origin == (1.0, 1.0)

        assert frame_plotter.visuals_2d.mask == None
        assert (frame_plotter.visuals_with_include_2d.mask == frame_7x7.mask).all()

        assert frame_plotter.visuals_2d.border == None
        assert (
            frame_plotter.visuals_with_include_2d.border
            == frame_7x7.mask.border_grid_sub_1.in_1d_binned
        ).all()

        assert frame_plotter.visuals_2d.parallel_overscan == None
        assert (
            frame_plotter.visuals_with_include_2d.parallel_overscan
            == frame_7x7.scans.parallel_overscan
        )

        include = aplt.Include2D(
            origin=False, mask=False, border=False, parallel_overscan=False
        )

        frame_plotter = aplt.FramePlotter(
            frame=frame_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert frame_plotter.visuals_with_include_2d.origin == (1.0, 1.0)
        assert frame_plotter.visuals_with_include_2d.mask == None
        assert frame_plotter.visuals_with_include_2d.border == None
        assert frame_plotter.visuals_with_include_2d.vector_field == 2
        assert frame_plotter.visuals_with_include_2d.parallel_overscan == None

    def test__works_with_all_extras_included(
        self,
        frame_7x7,
        mask_7x7,
        grid_7x7,
        grid_irregular_grouped_7x7,
        scans_7x7,
        plot_path,
        plot_patch,
    ):

        frame_plotter = aplt.FramePlotter(
            frame=frame_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame1", format="png")
            ),
        )

        frame_plotter.figure()

        assert path.join(plot_path, "frame1.png") in plot_patch.paths

        frame_plotter = aplt.FramePlotter(
            frame=frame_7x7,
            include_2d=aplt.Include2D(
                origin=True,
                mask=True,
                border=True,
                parallel_overscan=True,
                serial_prescan=True,
                serial_overscan=True,
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame2", format="png")
            ),
        )

        frame_plotter.figure()

        assert path.join(plot_path, "frame2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_irregular_grouped_7x7,
            mask=mask_7x7,
            border=mask_7x7.border_grid_sub_1.in_1d_binned,
            grid=grid_7x7,
            positions=grid_irregular_grouped_7x7,
            lines=grid_irregular_grouped_7x7,
            array_overlay=frame_7x7,
            parallel_overscan=scans_7x7.parallel_overscan,
            serial_prescan=scans_7x7.serial_prescan,
            serial_overscan=scans_7x7.serial_overscan,
        )

        frame_plotter = aplt.FramePlotter(
            frame=frame_7x7,
            visuals_2d=visuals_2d,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame3", format="png")
            ),
        )

        frame_plotter.figure()

        assert path.join(plot_path, "frame3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, frame_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        frame_plotter = aplt.FramePlotter(
            frame=frame_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame", format="fits")
            ),
        )

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        frame_plotter.figure()

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert (frame == frame_7x7.in_2d).all()


class TestGridPlotter:
    def test___visuals_in_constructor_use_grid_and_include(self, grid_7x7):

        visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

        include = aplt.Include2D(origin=True)

        grid_plotter = aplt.GridPlotter(
            grid=grid_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert grid_plotter.visuals_2d.origin == (1.0, 1.0)
        assert grid_plotter.visuals_with_include_2d.origin == (1.0, 1.0)

        include = aplt.Include2D(origin=False)

        grid_plotter = aplt.GridPlotter(
            grid=grid_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert grid_plotter.visuals_with_include_2d.origin == (1.0, 1.0)
        assert grid_plotter.visuals_with_include_2d.vector_field == 2

    def test__works_with_all_extras_included(
        self,
        array_7x7,
        grid_7x7,
        mask_7x7,
        grid_irregular_grouped_7x7,
        plot_path,
        plot_patch,
    ):

        grid_plotter = aplt.GridPlotter(
            grid=grid_7x7,
            visuals_2d=aplt.Visuals2D(indexes=[0, 1, 2]),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid1", format="png")
            ),
        )

        color_array = np.linspace(start=0.0, stop=1.0, num=grid_7x7.shape_1d)

        grid_plotter.figure(color_array=color_array)

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        grid_plotter = aplt.GridPlotter(
            grid=grid_7x7,
            visuals_2d=aplt.Visuals2D(indexes=[0, 1, 2]),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid2", format="png")
            ),
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        )

        grid_plotter.figure(color_array=color_array)

        assert path.join(plot_path, "grid2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_irregular_grouped_7x7,
            mask=mask_7x7,
            border=mask_7x7.border_grid_sub_1.in_1d_binned,
            grid=grid_7x7,
            positions=grid_irregular_grouped_7x7,
            lines=grid_irregular_grouped_7x7,
            array_overlay=array_7x7,
            indexes=[0, 1, 2],
        )

        grid_plotter = aplt.GridPlotter(
            grid=grid_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
            visuals_2d=visuals_2d,
        )

        grid_plotter.figure(color_array=color_array)

        assert path.join(plot_path, "grid3.png") in plot_patch.paths


class TestMapperPlotter:
    def test__visuals_for_data_from_rectangular_mapper(
        self, rectangular_mapper_7x7_3x3
    ):
        include = aplt.Include2D(
            origin=True, mask=True, mapper_data_pixelization_grid=True, border=True
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_data_with_include_2d.origin.in_1d_list == [
            (0.0, 0.0)
        ]
        assert (
            mapper_plotter.visuals_data_with_include_2d.mask
            == rectangular_mapper_7x7_3x3.source_slim_grid.mask
        ).all()
        assert mapper_plotter.visuals_data_with_include_2d.grid == None
        #  assert visuals.border == (0, 2)

        include = aplt.Include2D(
            origin=False, mask=False, mapper_data_pixelization_grid=False, border=False
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_data_with_include_2d.origin == None
        assert mapper_plotter.visuals_data_with_include_2d.mask == None
        assert mapper_plotter.visuals_data_with_include_2d.grid == None
        assert mapper_plotter.visuals_data_with_include_2d.border == None

    def test__visuals_for_data_from_voronoi_mapper(self, voronoi_mapper_9_3x3):

        include = aplt.Include2D(
            origin=True, mask=True, mapper_data_pixelization_grid=True, border=True
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_data_with_include_2d.origin.in_1d_list == [
            (0.0, 0.0)
        ]
        assert (
            mapper_plotter.visuals_data_with_include_2d.mask
            == voronoi_mapper_9_3x3.source_slim_grid.mask
        ).all()
        assert (
            mapper_plotter.visuals_data_with_include_2d.pixelization_grid
            == aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=0.1)
        ).all()
        #      assert visuals.border.shape == (0, 2)

        include = aplt.Include2D(
            origin=False, mask=False, mapper_data_pixelization_grid=False, border=False
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_data_with_include_2d.origin == None
        assert mapper_plotter.visuals_data_with_include_2d.mask == None
        assert mapper_plotter.visuals_data_with_include_2d.grid == None
        assert mapper_plotter.visuals_data_with_include_2d.pixelization_grid == None
        assert mapper_plotter.visuals_data_with_include_2d.border == None

    def test__visuals_for_source_from_rectangular_mapper(
        self, rectangular_mapper_7x7_3x3
    ):

        include = aplt.Include2D(
            origin=True,
            mapper_source_slim_grid=True,
            mapper_source_pixelization_grid=True,
            border=True,
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_2d.origin == None
        assert mapper_plotter.visuals_source_with_include_2d.origin.in_1d_list == [
            (0.0, 0.0)
        ]
        assert (
            mapper_plotter.visuals_source_with_include_2d.grid
            == rectangular_mapper_7x7_3x3.source_slim_grid
        ).all()
        assert (
            mapper_plotter.visuals_source_with_include_2d.pixelization_grid
            == rectangular_mapper_7x7_3x3.source_pixelization_grid
        ).all()
        assert (
            mapper_plotter.visuals_source_with_include_2d.border
            == rectangular_mapper_7x7_3x3.source_slim_grid.sub_border_grid
        ).all()

        include = aplt.Include2D(
            origin=False,
            border=False,
            mapper_source_slim_grid=False,
            mapper_source_pixelization_grid=False,
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_source_with_include_2d.origin == None
        assert mapper_plotter.visuals_source_with_include_2d.grid == None
        assert mapper_plotter.visuals_source_with_include_2d.pixelization_grid == None
        assert mapper_plotter.visuals_source_with_include_2d.border == None

    def test__visuals_for_source_from_voronoi_mapper(self, voronoi_mapper_9_3x3):

        include = aplt.Include2D(
            origin=True,
            border=True,
            mapper_source_slim_grid=True,
            mapper_source_pixelization_grid=True,
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_2d.origin == None
        assert mapper_plotter.visuals_source_with_include_2d.origin.in_1d_list == [
            (0.0, 0.0)
        ]
        assert (
            mapper_plotter.visuals_source_with_include_2d.grid
            == voronoi_mapper_9_3x3.source_slim_grid
        ).all()
        assert (
            mapper_plotter.visuals_source_with_include_2d.pixelization_grid
            == voronoi_mapper_9_3x3.source_pixelization_grid
        ).all()
        assert (
            mapper_plotter.visuals_source_with_include_2d.border
            == voronoi_mapper_9_3x3.source_slim_grid.sub_border_grid
        ).all()

        include = aplt.Include2D(
            origin=False, border=False, mapper_source_pixelization_grid=False
        )

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3, include_2d=include
        )

        assert mapper_plotter.visuals_source_with_include_2d.origin == None
        assert mapper_plotter.visuals_source_with_include_2d.grid == None
        assert mapper_plotter.visuals_source_with_include_2d.border == None

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            ),
        )

        mapper_plotter.figure()

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            ),
        )

        mapper_plotter.figure()

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
            include_2d=aplt.Include2D(
                origin=True, mapper_source_pixelization_grid=True
            ),
        )

        mapper_plotter.figure()

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        mapper_plotter = aplt.MapperPlotter(
            mapper=voronoi_mapper_9_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            ),
        )

        mapper_plotter.figure()

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mapper=voronoi_mapper_9_3x3,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            ),
        )

        mapper_plotter.figure()

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mapper=voronoi_mapper_9_3x3,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        mapper_plotter.figure()

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__image_and_mapper_subplot_is_output_for_all_mappers(
        self,
        imaging_7x7,
        rectangular_mapper_7x7_3x3,
        voronoi_mapper_9_3x3,
        plot_path,
        plot_patch,
    ):

        mapper_plotter = aplt.MapperPlotter(
            mapper=rectangular_mapper_7x7_3x3,
            visuals_2d=aplt.Visuals2D(
                indexes=[[(0, 0), (0, 1)], [(1, 2)]], pixelization_indexes=[[0, 1], [2]]
            ),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, format="png")
            ),
            include_2d=aplt.Include2D(mapper_source_pixelization_grid=True),
        )

        mapper_plotter.subplot_image_and_mapper(image=imaging_7x7.image)

        assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

        mapper_plotter.subplot_image_and_mapper(image=imaging_7x7.image)

        assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths


class TestLinePlotter:
    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        line_plotter = aplt.LinePlotter(
            mat_plot_1d=aplt.MatPlot1D(
                output=aplt.Output(path=plot_path, filename="line1", format="png")
            )
        )

        line_plotter.figure(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line1.png") in plot_patch.paths

        line_plotter = aplt.LinePlotter(
            mat_plot_1d=aplt.MatPlot1D(
                output=aplt.Output(path=plot_path, filename="line2", format="png")
            )
        )

        line_plotter.figure(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line2.png") in plot_patch.paths

        line_plotter = aplt.LinePlotter(
            mat_plot_1d=aplt.MatPlot1D(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            )
        )

        line_plotter.figure(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line3.png") in plot_patch.paths
