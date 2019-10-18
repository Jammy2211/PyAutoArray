import os

import pytest

import autoarray as aa


@pytest.fixture(name="interferometer_plotter_path")
def make_interferometer_plotter_setup():
    interferometer_plotter_path = "{}/../../test_files/plotting/interferometer/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return interferometer_plotter_path


def test__individual_attributes_are_output(
    interferometer_7, interferometer_plotter_path, plot_patch
):
    aa.plot.interferometer.visibilities(
        interferometer=interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_visibilities.png" in plot_patch.paths

    aa.plot.interferometer.u_wavelengths(
        interferometer=interferometer_7,
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_u_wavelengths.png" in plot_patch.paths

    aa.plot.interferometer.v_wavelengths(
        interferometer=interferometer_7,
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_v_wavelengths.png" in plot_patch.paths

    aa.plot.interferometer.primary_beam(
        interferometer=interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_primary_beam.png" in plot_patch.paths

    aa.plot.interferometer.subplot(
        interferometer=interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer.png" in plot_patch.paths


def test__individuals__output_dependent_on_input(
    interferometer_7, interferometer_plotter_path, plot_patch
):
    aa.plot.interferometer.individual(
        interferometer=interferometer_7,
        should_plot_visibilities=True,
        should_plot_u_wavelengths=False,
        should_plot_v_wavelengths=True,
        should_plot_primary_beam=True,
        output_path=interferometer_plotter_path,
        output_format="png",
    )

    assert interferometer_plotter_path + "interferometer_visibilities.png" in plot_patch.paths

    assert not interferometer_plotter_path + "interferometer_u_wavelengths.png" in plot_patch.paths

    assert interferometer_plotter_path + "interferometer_v_wavelengths.png" in plot_patch.paths

    assert interferometer_plotter_path + "interferometer_primary_beam.png" in plot_patch.paths
