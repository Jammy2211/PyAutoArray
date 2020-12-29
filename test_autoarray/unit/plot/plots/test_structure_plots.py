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


class TestPlotArray:
    def test__works_with_all_extras_included(
        self,
        array_7x7,
        mask_7x7,
        grid_7x7,
        grid_irregular_grouped_7x7,
        plot_path,
        plot_patch,
    ):

        aplt.Array(
            array=array_7x7,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="array1", format="png")
            ),
        )

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        aplt.Array(
            array=array_7x7,
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="array2", format="png")
            ),
        )

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

        aplt.Array(
            array=array_7x7,
            visuals_2d=visuals_2d,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, array_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        plotter_2d = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="array", format="fits")
        )

        aplt.Array(array=array_7x7, plotter_2d=plotter_2d)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == array_7x7.in_2d).all()


class TestPlotFrame:
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

        aplt.Frame(
            frame=frame_7x7,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="frame1", format="png")
            ),
        )

        assert path.join(plot_path, "frame1.png") in plot_patch.paths

        aplt.Frame(
            frame=frame_7x7,
            include_2d=aplt.Include2D(
                origin=True,
                mask=True,
                border=True,
                parallel_overscan=True,
                serial_prescan=True,
                serial_overscan=True,
            ),
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="frame2", format="png")
            ),
        )

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

        aplt.Frame(
            frame=frame_7x7,
            visuals_2d=visuals_2d,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="frame3", format="png")
            ),
        )

        assert path.join(plot_path, "frame3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, frame_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        plotter_2d = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="frame", format="fits")
        )

        aplt.Frame(frame=frame_7x7, plotter_2d=plotter_2d)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert (frame == frame_7x7.in_2d).all()


class TestPlotGrid:
    def test__works_with_all_extras_included(
        self,
        array_7x7,
        grid_7x7,
        mask_7x7,
        grid_irregular_grouped_7x7,
        plot_path,
        plot_patch,
    ):

        color_array = np.linspace(start=0.0, stop=1.0, num=grid_7x7.shape_1d)

        aplt.Grid(
            grid=grid_7x7,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="grid1", format="png")
            ),
        )

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        aplt.Grid(
            grid=grid_7x7,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            indexes=[0, 1, 2, 14],
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
            symmetric_around_centre=True,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="grid2", format="png")
            ),
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

        aplt.Grid(
            grid=grid_7x7,
            visuals_2d=visuals_2d,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
        )

        assert path.join(plot_path, "grid3.png") in plot_patch.paths


class TestPlotMapper:
    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            ),
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            ),
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        include_2d = aplt.Include2D(
            origin=True, mapper_source_grid=True, mapper_source_border=True
        )

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            include_2d=include_2d,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper1", format="png")
            ),
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper2", format="png")
            ),
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter_2d=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths


class TestPlotLine:
    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter_1d=aplt.Plotter1D(
                output=aplt.Output(path=plot_path, filename="line1", format="png")
            ),
        )

        assert path.join(plot_path, "line1.png") in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter_1d=aplt.Plotter1D(
                output=aplt.Output(path=plot_path, filename="line2", format="png")
            ),
        )

        assert path.join(plot_path, "line2.png") in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter_1d=aplt.Plotter1D(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert path.join(plot_path, "line3.png") in plot_patch.paths
