from os import path
import pytest
import autoarray as aa
import autoarray.plot as aplt

import numpy as np

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__multi_plotter__subplot_of_plotter_list_figure(
    imaging_7x7, plot_path, plot_patch
):
    mat_plot_2d = aplt.MatPlot2D(output=aplt.Output(plot_path, format="png"))

    plotter_0 = aplt.ImagingPlotter(dataset=imaging_7x7, mat_plot_2d=mat_plot_2d)
    plotter_1 = aplt.ImagingPlotter(dataset=imaging_7x7, mat_plot_2d=mat_plot_2d)

    plotter_list = [plotter_0, plotter_1]

    multi_plotter = aplt.MultiFigurePlotter(plotter_list=plotter_list)
    multi_plotter.subplot_of_figure(func_name="figures_2d", figure_name="data")

    assert path.join(plot_path, "subplot_data.png") in plot_patch.paths

    plot_patch.paths = []

    multi_plotter = aplt.MultiFigurePlotter(plotter_list=plotter_list)
    multi_plotter.subplot_of_figure(
        func_name="figures_2d", figure_name="data", noise_map=True
    )

    assert path.join(plot_path, "subplot_data.png") in plot_patch.paths


class MockYX1DPlotter(aplt.YX1DPlotter):
    def __init__(
        self,
        y,
        x,
        mat_plot_1d: aplt.MatPlot1D = None,
        visuals_1d: aplt.Visuals1D = None,
    ):
        super().__init__(
            y=y,
            x=x,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
        )

    def figures_1d(self, figure_name=False):
        if figure_name:
            self.figure_1d()


def test__multi_yx_plotter__subplot_of_plotter_list_figure(
    imaging_7x7, plot_path, plot_patch
):
    mat_plot_1d = aplt.MatPlot1D(output=aplt.Output(plot_path, format="png"))

    plotter_0 = MockYX1DPlotter(
        y=aa.Array1D.no_mask([1.0, 2.0, 3.0], pixel_scales=1.0),
        x=aa.Array1D.no_mask([0.5, 1.0, 1.5], pixel_scales=0.5),
        mat_plot_1d=mat_plot_1d,
    )

    plotter_1 = MockYX1DPlotter(
        y=aa.Array1D.no_mask([1.0, 2.0, 4.0], pixel_scales=1.0),
        x=aa.Array1D.no_mask([0.5, 1.0, 1.5], pixel_scales=0.5),
        mat_plot_1d=mat_plot_1d,
    )

    multi_plotter = aplt.MultiYX1DPlotter(plotter_list=[plotter_0, plotter_1])
    multi_plotter.figure_1d(func_name="figures_1d", figure_name="figure_name")

    assert path.join(plot_path, "multi_figure_name.png") in plot_patch.paths


def test__multi_yx_plotter__xticks_span_all_plotter_ranges():
    plotter_0 = MockYX1DPlotter(
        y=aa.Array1D.no_mask([1.0, 2.0, 3.0], pixel_scales=1.0),
        x=aa.Array1D.no_mask([0.5, 1.0, 1.5], pixel_scales=0.5),
    )

    plotter_1 = MockYX1DPlotter(
        y=aa.Array1D.no_mask([1.0, 2.0, 4.0], pixel_scales=1.0),
        x=aa.Array1D.no_mask([0.25, 1.0, 1.5], pixel_scales=0.5),
    )

    multi_plotter = aplt.MultiYX1DPlotter(plotter_list=[plotter_0, plotter_1])

    assert multi_plotter.xticks.manual_min_max_value == (0.25, 1.5)
    assert multi_plotter.yticks.manual_min_max_value == (1.0, 4.0)
