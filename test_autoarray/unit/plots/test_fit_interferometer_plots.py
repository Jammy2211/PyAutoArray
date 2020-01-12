import autoarray as aa
import pytest
import os

from os import path
from autoarray import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="fit_interferometer_path")
def make_fit_interferometer_path_setup():
    return "{}/../../test_files/plotting/fit_interferometer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


def test__fit_sub_plot(fit_interferometer_7, fit_interferometer_path, plot_patch):

    aa.plot.fit_interferometer.subplot_fit_interferometer(
        fit=fit_interferometer_7,
        sub_plotter=aa.plotter.SubPlotter(
            output=aa.plotter.Output(fit_interferometer_path, format="png")
        ),
    )

    assert fit_interferometer_path + "subplot_fit_interferometer.png" in plot_patch.paths


def test__fit_individuals__dependent_on_input(
    fit_interferometer_7, fit_interferometer_path, plot_patch
):

    aa.plot.fit_interferometer.individuals(
        fit=fit_interferometer_7,
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_chi_squared_map=True,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(fit_interferometer_path, format="png")
        ),
    )

    assert fit_interferometer_path + "visibilities.png" in plot_patch.paths

    assert fit_interferometer_path + "noise_map.png" not in plot_patch.paths

    assert fit_interferometer_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert fit_interferometer_path + "model_visibilities.png" in plot_patch.paths

    assert (
        fit_interferometer_path + "residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_path + "residual_map_vs_uv_distances_imag.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_path + "normalized_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_path + "normalized_residual_map_vs_uv_distances_imag.png"
        not in plot_patch.paths
    )

    assert (
        fit_interferometer_path + "chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    assert (
        fit_interferometer_path + "chi_squared_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )


def test__fit_quantities_are_output(
    fit_interferometer_7, fit_interferometer_path, plot_patch
):

    aa.plot.fit_interferometer.visibilities(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert fit_interferometer_path + "visibilities.png" in plot_patch.paths

    aa.plot.fit_interferometer.noise_map(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert fit_interferometer_path + "noise_map.png" in plot_patch.paths

    aa.plot.fit_interferometer.signal_to_noise_map(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert fit_interferometer_path + "signal_to_noise_map.png" in plot_patch.paths

    aa.plot.fit_interferometer.model_visibilities(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert fit_interferometer_path + "model_visibilities.png" in plot_patch.paths

    aa.plot.fit_interferometer.residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert (
        fit_interferometer_path + "residual_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plot_real=False,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert (
        fit_interferometer_path + "residual_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.normalized_residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert (
        fit_interferometer_path + "normalized_residual_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.normalized_residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plot_real=False,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert (
        fit_interferometer_path + "normalized_residual_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.chi_squared_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert (
        fit_interferometer_path + "chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.chi_squared_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plot_real=False,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=fit_interferometer_path, format="png")
        ),
    )

    assert (
        fit_interferometer_path + "chi_squared_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )
