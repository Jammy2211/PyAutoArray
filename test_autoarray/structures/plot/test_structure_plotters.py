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


class TestYX1DPlotter:
    def test__plot_yx_line__works_with_all_extras_included(self, plot_path, plot_patch):

        visuals_1d = aplt.Visuals1D(vertical_line=1.0)

        mat_plot_1d = aplt.MatPlot1D(
            yx_plot=aplt.YXPlot(plot_axis_type="loglog", c="k"),
            vertical_line_axvline=aplt.AXVLine(c="k"),
            output=aplt.Output(path=plot_path, filename="yx_1", format="png"),
        )

        yx_1d_plotter = aplt.YX1DPlotter(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

        yx_1d_plotter.figure_1d()

        assert path.join(plot_path, "yx_1.png") in plot_patch.paths


class TestArray2DPlotter:
    def test___visuals_in_constructor_use_array_and_include(self, array_2d_7x7):

        visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

        include = aplt.Include2D(origin=True, mask=True, border=True)

        array_plotter = aplt.Array2DPlotter(
            array=array_2d_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert array_plotter.visuals_2d.origin == (1.0, 1.0)
        assert array_plotter.visuals_with_include_2d.origin == (1.0, 1.0)

        assert array_plotter.visuals_2d.mask == None
        assert (array_plotter.visuals_with_include_2d.mask == array_2d_7x7.mask).all()

        assert array_plotter.visuals_2d.border == None
        assert (
            array_plotter.visuals_with_include_2d.border
            == array_2d_7x7.mask.border_grid_sub_1.binned
        ).all()

        assert array_plotter.visuals_2d.vector_field == 2
        assert array_plotter.visuals_with_include_2d.vector_field == 2

        include = aplt.Include2D(origin=False, mask=False, border=False)

        array_plotter = aplt.Array2DPlotter(
            array=array_2d_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert array_plotter.get_2d.via_mask_from(mask=array_2d_7x7.mask).origin == (
            1.0,
            1.0,
        )
        assert array_plotter.get_2d.via_mask_from(mask=array_2d_7x7.mask).mask == None
        assert array_plotter.get_2d.via_mask_from(mask=array_2d_7x7.mask).border == None
        assert (
            array_plotter.get_2d.via_mask_from(mask=array_2d_7x7.mask).vector_field == 2
        )

    def test__works_with_all_extras_included(
        self,
        array_2d_7x7,
        mask_2d_7x7,
        grid_2d_7x7,
        grid_2d_irregular_7x7_list,
        plot_path,
        plot_patch,
    ):

        array_plotter = aplt.Array2DPlotter(
            array=array_2d_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array1", format="png")
            ),
        )

        array_plotter.figure_2d()

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        array_plotter = aplt.Array2DPlotter(
            array=array_2d_7x7,
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array2", format="png")
            ),
        )

        array_plotter.figure_2d()

        assert path.join(plot_path, "array2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_2d_irregular_7x7_list,
            mask=mask_2d_7x7,
            border=mask_2d_7x7.border_grid_sub_1.binned,
            grid=grid_2d_7x7,
            positions=grid_2d_irregular_7x7_list,
            #       lines=grid_2d_irregular_7x7_list,
            array_overlay=array_2d_7x7,
        )

        array_plotter = aplt.Array2DPlotter(
            array=array_2d_7x7,
            visuals_2d=visuals_2d,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        array_plotter.figure_2d()

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__fits_files_output_correctly(self, array_2d_7x7, plot_path):

        plot_path = path.join(plot_path, "fits")

        array_plotter = aplt.Array2DPlotter(
            array=array_2d_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="array", format="fits")
            ),
        )

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        array_plotter.figure_2d()

        arr = aa.util.array_2d.numpy_array_2d_via_fits_from(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == array_2d_7x7.native).all()


class TestGrid2DPlotter:
    def test___visuals_in_constructor_use_grid_and_include(self, grid_2d_7x7):

        visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

        include = aplt.Include2D(origin=True)

        grid_plotter = aplt.Grid2DPlotter(
            grid=grid_2d_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert grid_plotter.visuals_2d.origin == (1.0, 1.0)
        assert grid_plotter.get_2d.via_grid_from(grid=grid_2d_7x7).origin == (1.0, 1.0)

        include = aplt.Include2D(origin=False)

        grid_plotter = aplt.Grid2DPlotter(
            grid=grid_2d_7x7, visuals_2d=visuals_2d, include_2d=include
        )

        assert grid_plotter.get_2d.via_grid_from(grid=grid_2d_7x7).origin == (1.0, 1.0)
        assert grid_plotter.get_2d.via_grid_from(grid=grid_2d_7x7).vector_field == 2

    def test__works_with_all_extras_included(
        self,
        array_2d_7x7,
        grid_2d_7x7,
        mask_2d_7x7,
        grid_2d_irregular_7x7_list,
        plot_path,
        plot_patch,
    ):

        grid_plotter = aplt.Grid2DPlotter(
            grid=grid_2d_7x7,
            visuals_2d=aplt.Visuals2D(indexes=[0, 1, 2]),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid1", format="png")
            ),
        )

        color_array = np.linspace(start=0.0, stop=1.0, num=grid_2d_7x7.shape_slim)

        grid_plotter.figure_2d(color_array=color_array)

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        grid_plotter = aplt.Grid2DPlotter(
            grid=grid_2d_7x7,
            visuals_2d=aplt.Visuals2D(indexes=[0, 1, 2]),
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid2", format="png")
            ),
            include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        )

        grid_plotter.figure_2d(color_array=color_array)

        assert path.join(plot_path, "grid2.png") in plot_patch.paths

        visuals_2d = aplt.Visuals2D(
            origin=grid_2d_irregular_7x7_list,
            mask=mask_2d_7x7,
            border=mask_2d_7x7.border_grid_sub_1.binned,
            grid=grid_2d_7x7,
            positions=grid_2d_irregular_7x7_list,
            lines=grid_2d_irregular_7x7_list,
            array_overlay=array_2d_7x7,
            indexes=[0, 1, 2],
        )

        grid_plotter = aplt.Grid2DPlotter(
            grid=grid_2d_7x7,
            mat_plot_2d=aplt.MatPlot2D(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
            visuals_2d=visuals_2d,
        )

        grid_plotter.figure_2d(color_array=color_array)

        assert path.join(plot_path, "grid3.png") in plot_patch.paths
