import os

import pytest

from autoarray.data.plotters import imaging_plotters

@pytest.fixture(name="imaging_plotter_path")
def make_imaging_plotter_setup():
    imaging_plotter_path = "{}/../../test_files/plotting/imaging/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return imaging_plotter_path


def test__individual_attributes_are_output(
    imaging_data_7x7, positions_7x7, mask_7x7, imaging_plotter_path, plot_patch
):
    imaging_plotters.plot_image(
        imaging_data=imaging_data_7x7,
        positions=positions_7x7,
        mask_overlay=mask_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_plotter_path,
        output_format="png",
    )

    assert imaging_plotter_path + "imaging_image.png" in plot_patch.paths

    imaging_plotters.plot_noise_map(
        imaging_data=imaging_data_7x7,
        mask_overlay=mask_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_plotter_path,
        output_format="png",
    )

    assert imaging_plotter_path + "imaging_noise_map.png" in plot_patch.paths

    imaging_plotters.plot_psf(
        imaging_data=imaging_data_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_plotter_path,
        output_format="png",
    )

    assert imaging_plotter_path + "imaging_psf.png" in plot_patch.paths

    imaging_plotters.plot_signal_to_noise_map(
        imaging_data=imaging_data_7x7,
        mask_overlay=mask_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_plotter_path,
        output_format="png",
    )

    assert imaging_plotter_path + "imaging_signal_to_noise_map.png" in plot_patch.paths

    imaging_plotters.plot_imaging_subplot(
        imaging_data=imaging_data_7x7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=imaging_plotter_path,
        output_format="png",
    )

    assert imaging_plotter_path + "imaging_data.png" in plot_patch.paths


def test__imaging_individuals__output_dependent_on_input(
    imaging_data_7x7, imaging_plotter_path, plot_patch
):
    imaging_plotters.plot_imaging_individual(
        imaging_data=imaging_data_7x7,
        should_plot_image=True,
        should_plot_psf=True,
        should_plot_absolute_signal_to_noise_map=True,
        output_path=imaging_plotter_path,
        output_format="png",
    )

    assert imaging_plotter_path + "imaging_image.png" in plot_patch.paths

    assert not imaging_plotter_path + "imaging_noise_map.png" in plot_patch.paths

    assert imaging_plotter_path + "imaging_psf.png" in plot_patch.paths

    assert (
        not imaging_plotter_path + "imaging_signal_to_noise_map.png" in plot_patch.paths
    )

    assert (
        imaging_plotter_path + "imaging_absolute_signal_to_noise_map.png"
        in plot_patch.paths
    )

    assert (
        not imaging_plotter_path + "imaging_potential_chi_squared_map.png"
        in plot_patch.paths
    )
