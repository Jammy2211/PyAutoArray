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


def test__plot_yx_line(plot_path, plot_patch):
    yx_1d_plotter = aplt.YX1DPlotter(
        y=aa.Array1D.no_mask([1.0, 2.0, 3.0], pixel_scales=1.0),
        x=aa.Array1D.no_mask([0.5, 1.0, 1.5], pixel_scales=0.5),
        output=aplt.Output(path=plot_path, filename="yx_1", format="png"),
        vertical_line=1.0,
        plot_axis_type="loglog",
    )

    yx_1d_plotter.figure_1d()

    assert path.join(plot_path, "yx_1.png") in plot_patch.paths


def test__array(
    array_2d_7x7,
    mask_2d_7x7,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    array_plotter = aplt.Array2DPlotter(
        array=array_2d_7x7,
        output=aplt.Output(path=plot_path, filename="array1", format="png"),
    )

    array_plotter.figure_2d()

    assert path.join(plot_path, "array1.png") in plot_patch.paths

    array_plotter = aplt.Array2DPlotter(
        array=array_2d_7x7,
        output=aplt.Output(path=plot_path, filename="array2", format="png"),
    )

    array_plotter.figure_2d()

    assert path.join(plot_path, "array2.png") in plot_patch.paths

    array_plotter = aplt.Array2DPlotter(
        array=array_2d_7x7,
        origin=grid_2d_irregular_7x7_list,
        border=mask_2d_7x7.derive_grid.border,
        grid=grid_2d_7x7,
        positions=grid_2d_irregular_7x7_list,
        array_overlay=array_2d_7x7,
        output=aplt.Output(path=plot_path, filename="array3", format="png"),
    )

    array_plotter.figure_2d()

    assert path.join(plot_path, "array3.png") in plot_patch.paths


def test__array__fits_files_output_correctly(array_2d_7x7, plot_path):
    plot_path = path.join(plot_path, "fits")

    array_plotter = aplt.Array2DPlotter(
        array=array_2d_7x7,
        output=aplt.Output(path=plot_path, filename="array", format="fits"),
    )

    if path.exists(plot_path):
        shutil.rmtree(plot_path)

    array_plotter.figure_2d()

    arr = aa.ndarray_via_fits_from(file_path=path.join(plot_path, "array.fits"), hdu=0)

    assert (arr == array_2d_7x7.native).all()


def test__grid(
    array_2d_7x7,
    grid_2d_7x7,
    mask_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    grid_2d_plotter = aplt.Grid2DPlotter(
        grid=grid_2d_7x7,
        indexes=[0, 1, 2],
        output=aplt.Output(path=plot_path, filename="grid1", format="png"),
    )

    color_array = np.linspace(start=0.0, stop=1.0, num=grid_2d_7x7.shape_slim)

    grid_2d_plotter.figure_2d(color_array=color_array)

    assert path.join(plot_path, "grid1.png") in plot_patch.paths

    grid_2d_plotter = aplt.Grid2DPlotter(
        grid=grid_2d_7x7,
        indexes=[0, 1, 2],
        output=aplt.Output(path=plot_path, filename="grid2", format="png"),
    )

    grid_2d_plotter.figure_2d(color_array=color_array)

    assert path.join(plot_path, "grid2.png") in plot_patch.paths

    grid_2d_plotter = aplt.Grid2DPlotter(
        grid=grid_2d_7x7,
        lines=grid_2d_irregular_7x7_list,
        positions=grid_2d_irregular_7x7_list,
        indexes=[0, 1, 2],
        output=aplt.Output(path=plot_path, filename="grid3", format="png"),
    )

    grid_2d_plotter.figure_2d(color_array=color_array)

    assert path.join(plot_path, "grid3.png") in plot_patch.paths


def test__array_rgb(
    array_2d_rgb_7x7,
    plot_path,
    plot_patch,
):
    array_plotter = aplt.Array2DPlotter(
        array=array_2d_rgb_7x7,
        output=aplt.Output(path=plot_path, filename="array_rgb", format="png"),
    )

    array_plotter.figure_2d()

    assert path.join(plot_path, "array_rgb.png") in plot_patch.paths
