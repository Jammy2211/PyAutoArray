from os import path
import pytest

import autoarray as aa
import autoarray.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "structures"
    )


def test__figure_2d(
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
    plot_path,
    plot_patch,
):
    visuals_2d = aplt.Visuals2D(
        indexes=[[(0, 0), (0, 1)], [(1, 2)]],
    )

    mat_plot_2d = aplt.MatPlot2D(
        output=aplt.Output(path=plot_path, filename="mapper1", format="png")
    )

    mapper_plotter = aplt.MapperPlotter(
        mapper=rectangular_mapper_7x7_3x3,
        visuals_2d=visuals_2d,
        mat_plot_2d=mat_plot_2d,
    )

    mapper_plotter.figure_2d()

    assert path.join(plot_path, "mapper1.png") in plot_patch.paths

    plot_patch.paths = []

    mapper_plotter = aplt.MapperPlotter(
        mapper=delaunay_mapper_9_3x3,
        visuals_2d=visuals_2d,
        mat_plot_2d=mat_plot_2d,
    )

    mapper_plotter.figure_2d(interpolate_to_uniform=True)

    assert path.join(plot_path, "mapper1.png") in plot_patch.paths

    pytest.importorskip(
        "autoarray.util.nn.nn_py",
        reason="Voronoi C library not installed, see util.nn README.md",
    )

    plot_patch.paths = []

    mapper_plotter = aplt.MapperPlotter(
        mapper=voronoi_mapper_9_3x3,
        visuals_2d=visuals_2d,
        mat_plot_2d=mat_plot_2d,
    )

    mapper_plotter.figure_2d(interpolate_to_uniform=True)

    assert path.join(plot_path, "mapper1.png") in plot_patch.paths


def test__subplot_image_and_mapper(
    imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    voronoi_mapper_9_3x3,
    plot_path,
    plot_patch,
):
    visuals_2d = aplt.Visuals2D(indexes=[0, 1, 2])

    mapper_plotter = aplt.MapperPlotter(
        mapper=rectangular_mapper_7x7_3x3,
        visuals_2d=visuals_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    mapper_plotter.subplot_image_and_mapper(
        image=imaging_7x7.data, interpolate_to_uniform=True
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

    plot_patch.paths = []

    mapper_plotter = aplt.MapperPlotter(
        mapper=delaunay_mapper_9_3x3,
        visuals_2d=visuals_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    mapper_plotter.subplot_image_and_mapper(
        image=imaging_7x7.data, interpolate_to_uniform=True
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

    pytest.importorskip(
        "autoarray.util.nn.nn_py",
        reason="Voronoi C library not installed, see util.nn README.md",
    )

    plot_patch.paths = []

    mapper_plotter = aplt.MapperPlotter(
        mapper=voronoi_mapper_9_3x3,
        visuals_2d=visuals_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    mapper_plotter.subplot_image_and_mapper(
        image=imaging_7x7.data, interpolate_to_uniform=True
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths
