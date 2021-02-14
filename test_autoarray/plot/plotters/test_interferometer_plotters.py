from os import path

import pytest
import autoarray.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "interferometer",
    )


def test__individual_attributes_are_output(interferometer_7, plot_path, plot_patch):

    interferometer_plotter = aplt.InterferometerPlotter(
        interferometer=interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    interferometer_plotter.figures(
        visibilities=True,
        noise_map=True,
        u_wavelengths=True,
        v_wavelengths=True,
        uv_wavelengths=True,
        amplitudes_vs_uv_distances=True,
        phases_vs_uv_distances=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "u_wavelengths.png") in plot_patch.paths
    assert path.join(plot_path, "v_wavelengths.png") in plot_patch.paths
    assert path.join(plot_path, "uv_wavelengths.png") in plot_patch.paths
    assert path.join(plot_path, "amplitudes_vs_uv_distances.png") in plot_patch.paths
    assert path.join(plot_path, "phases_vs_uv_distances.png") in plot_patch.paths

    plot_patch.paths = []

    interferometer_plotter.figures(
        visibilities=True,
        u_wavelengths=False,
        v_wavelengths=True,
        amplitudes_vs_uv_distances=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert not path.join(plot_path, "u_wavelengths.png") in plot_patch.paths
    assert path.join(plot_path, "v_wavelengths.png") in plot_patch.paths
    assert path.join(plot_path, "amplitudes_vs_uv_distances.png") in plot_patch.paths
    assert path.join(plot_path, "phases_vs_uv_distances.png") not in plot_patch.paths


def test__subplot_is_output(interferometer_7, plot_path, plot_patch):

    interferometer_plotter = aplt.InterferometerPlotter(
        interferometer=interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    interferometer_plotter.subplot_interferometer()

    assert path.join(plot_path, "subplot_interferometer.png") in plot_patch.paths
