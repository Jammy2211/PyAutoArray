import autoarray as aa
import autoarray.plot as aplt
from os import path
import matplotlib.pyplot as plt
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
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array1", format="png")
            )
        )

        array_plotter.array(array=array_7x7)

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        array_plotter = aplt.ArrayPlotter(
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array2", format="png")
            ),
        )

        array_plotter.array(array=array_7x7)

        assert path.join(plot_path, "array2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_irregular_grouped_7x7,
            mask=mask_7x7,
            border=mask_7x7.geometry.border_grid_sub_1.in_1d_binned,
            grid=grid_7x7,
            positions=grid_irregular_grouped_7x7,
            lines=grid_irregular_grouped_7x7,
            array_overlay=array_7x7,
        )

        array_plotter = aplt.ArrayPlotter(
            visuals_2d=visuals_2d,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        array_plotter.array(array=array_7x7)

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, array_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        array_plotter = aplt.ArrayPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array", format="fits")
            )
        )

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        array_plotter.array(array=array_7x7)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == array_7x7.in_2d).all()


class TestFramePlotter:
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
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame1", format="png")
            )
        )

        frame_plotter.frame(frame=frame_7x7)

        assert path.join(plot_path, "frame1.png") in plot_patch.paths

        frame_plotter = aplt.FramePlotter(
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

        frame_plotter.frame(frame=frame_7x7)

        assert path.join(plot_path, "frame2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_irregular_grouped_7x7,
            mask=mask_7x7,
            border=mask_7x7.geometry.border_grid_sub_1.in_1d_binned,
            grid=grid_7x7,
            positions=grid_irregular_grouped_7x7,
            lines=grid_irregular_grouped_7x7,
            array_overlay=frame_7x7,
            parallel_overscan=scans_7x7.parallel_overscan,
            serial_prescan=scans_7x7.serial_prescan,
            serial_overscan=scans_7x7.serial_overscan,
        )

        frame_plotter = aplt.FramePlotter(
            visuals_2d=visuals_2d,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame3", format="png")
            ),
        )

        frame_plotter.frame(frame=frame_7x7)

        assert path.join(plot_path, "frame3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, frame_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        frame_plotter = aplt.FramePlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="frame", format="fits")
            )
        )

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        frame_plotter.frame(frame=frame_7x7)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert (frame == frame_7x7.in_2d).all()


class TestGridPlotter:
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
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid1", format="png")
            )
        )

        color_array = np.linspace(start=0.0, stop=1.0, num=grid_7x7.shape_1d)

        grid_plotter.grid(
            grid=grid_7x7,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        grid_plotter = aplt.GridPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid2", format="png")
            ),
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        )

        grid_plotter.grid(
            grid=grid_7x7,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert path.join(plot_path, "grid2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_irregular_grouped_7x7,
            mask=mask_7x7,
            border=mask_7x7.geometry.border_grid_sub_1.in_1d_binned,
            grid=grid_7x7,
            positions=grid_irregular_grouped_7x7,
            lines=grid_irregular_grouped_7x7,
            array_overlay=array_7x7,
        )

        grid_plotter = aplt.GridPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
            visuals_2d=visuals_2d,
        )

        grid_plotter.grid(
            grid=grid_7x7,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert path.join(plot_path, "grid3.png") in plot_patch.paths


class TestMapperPlotter:
    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        mapper_plotter = aplt.MapperPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            )
        )

        mapper_plotter.mapper(
            mapper=rectangular_mapper_7x7_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            )
        )

        mapper_plotter.mapper(
            mapper=rectangular_mapper_7x7_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
            include_2d=aplt.Include2D(
                origin=True,
                mapper_source_pixelization_grid=True,
                mapper_source_border=True,
            ),
        )

        mapper_plotter.mapper(
            mapper=rectangular_mapper_7x7_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        mapper_plotter = aplt.MapperPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            )
        )

        mapper_plotter.mapper(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            )
        )

        mapper_plotter.mapper(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        mapper_plotter = aplt.MapperPlotter(
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            )
        )

        mapper_plotter.mapper(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

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
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, format="png")
            ),
            include_2d=aplt.Include2D(
                mapper_source_pixelization_grid=True, mapper_source_border=True
            ),
        )

        mapper_plotter.subplot_image_and_mapper(
            image=imaging_7x7.image,
            mapper=rectangular_mapper_7x7_3x3,
            full_indexes=[[0, 1, 2], [3]],
            pixelization_indexes=[[1, 2], [0]],
        )

        assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

        mapper_plotter.subplot_image_and_mapper(
            image=imaging_7x7.image,
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[0, 1, 2], [3]],
            pixelization_indexes=[[1, 2], [0]],
        )

        assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths


class TestLinePlotter:
    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        line_plotter = aplt.LinePlotter(
            mat_plot_1d=aplt.MatPlot1D(
                output=aplt.Output(path=plot_path, filename="line1", format="png")
            )
        )

        line_plotter.line(
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

        line_plotter.line(
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

        line_plotter.line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line3.png") in plot_patch.paths
