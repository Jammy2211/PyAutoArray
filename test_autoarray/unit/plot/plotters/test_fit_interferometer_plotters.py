import autoarray.plot as aplt
import pytest

from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "fit_interferometer",
    )


def test__fit_quantities_are_output(fit_interferometer_7, plot_path, plot_patch):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_interferometer_plotter.figure_visibilities()
    assert path.join(plot_path, "visibilities.png") in plot_patch.paths

    fit_interferometer_plotter.figure_noise_map()
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    fit_interferometer_plotter.figure_signal_to_noise_map()
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    fit_interferometer_plotter.figure_model_visibilities()
    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths

    fit_interferometer_plotter.figure_residual_map_vs_uv_distances()
    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

    fit_interferometer_plotter.figure_residual_map_vs_uv_distances(plot_real=False)
    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )

    fit_interferometer_plotter.figure_normalized_residual_map_vs_uv_distances()
    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

    fit_interferometer_plotter.figure_normalized_residual_map_vs_uv_distances(
        plot_real=False
    )
    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )

    fit_interferometer_plotter.figure_chi_squared_map_vs_uv_distances()
    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )

    fit_interferometer_plotter.figure_chi_squared_map_vs_uv_distances(plot_real=False)
    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )


def test__fit_sub_plot(fit_interferometer_7, plot_path, plot_patch):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_interferometer_plotter.subplot_fit_interferometer()

    assert path.join(plot_path, "subplot_fit_interferometer.png") in plot_patch.paths


def test__fit_individuals__dependent_on_input(
    fit_interferometer_7, plot_path, plot_patch
):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_interferometer_plotter.individuals(
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_chi_squared_map=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_real.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "residual_map_vs_uv_distances_imag.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_real.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "normalized_residual_map_vs_uv_distances_imag.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_real.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "chi_squared_map_vs_uv_distances_imag.png")
        in plot_patch.paths
    )
