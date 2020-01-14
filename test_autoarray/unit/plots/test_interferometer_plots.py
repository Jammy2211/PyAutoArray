from os import path
from autoarray import conf
import os

import pytest

import autoarray as aa

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="interferometer_plotter_path")
def make_interferometer_plotter_setup():
    interferometer_plotter_path = "{}/../../test_files/plotting/interferometer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return interferometer_plotter_path


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


def test__individual_attributes_are_output(
    interferometer_7, interferometer_plotter_path, plot_patch
):

    aa.plot.interferometer.visibilities(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "visibilities.png" in plot_patch.paths

    aa.plot.interferometer.noise_map(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "noise_map.png" in plot_patch.paths

    aa.plot.interferometer.u_wavelengths(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "u_wavelengths.png" in plot_patch.paths

    aa.plot.interferometer.v_wavelengths(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "v_wavelengths.png" in plot_patch.paths

    aa.plot.interferometer.uv_wavelengths(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "uv_wavelengths.png" in plot_patch.paths

    aa.plot.interferometer.amplitudes_vs_uv_distances(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert (
        interferometer_plotter_path + "amplitudes_vs_uv_distances.png"
        in plot_patch.paths
    )

    aa.plot.interferometer.phases_vs_uv_distances(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert (
        interferometer_plotter_path + "phases_vs_uv_distances.png" in plot_patch.paths
    )

    aa.plot.interferometer.primary_beam(
        interferometer=interferometer_7,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "primary_beam.png" in plot_patch.paths


def test__subplot_is_output(interferometer_7, interferometer_plotter_path, plot_patch):

    aa.plot.interferometer.subplot_interferometer(
        interferometer=interferometer_7,
        sub_plotter=aa.plotter.SubPlotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert (
        interferometer_plotter_path + "subplot_interferometer.png" in plot_patch.paths
    )


def test__individuals__output_dependent_on_input(
    interferometer_7, interferometer_plotter_path, plot_patch
):
    aa.plot.interferometer.individual(
        interferometer=interferometer_7,
        plot_visibilities=True,
        plot_u_wavelengths=False,
        plot_v_wavelengths=True,
        plot_primary_beam=True,
        plot_amplitudes_vs_uv_distances=True,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(path=interferometer_plotter_path, format="png")
        ),
    )

    assert interferometer_plotter_path + "visibilities.png" in plot_patch.paths

    assert not interferometer_plotter_path + "u_wavelengths.png" in plot_patch.paths

    assert interferometer_plotter_path + "v_wavelengths.png" in plot_patch.paths

    assert (
        interferometer_plotter_path + "amplitudes_vs_uv_distances.png"
        in plot_patch.paths
    )

    assert (
        interferometer_plotter_path + "phases_vs_uv_distances.png"
        not in plot_patch.paths
    )

    assert interferometer_plotter_path + "primary_beam.png" in plot_patch.paths
