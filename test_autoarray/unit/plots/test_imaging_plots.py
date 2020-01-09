from os import path
from autoarray import conf
import os

import pytest

import autoarray as aa


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="imaging_plotter_path")
def make_imaging_plotter_setup():
    imaging_plotter_path = "{}/../../test_files/plotting/imaging/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    return imaging_plotter_path


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


def test__individual_attributes_are_output(
    imaging_7x7, positions_7x7, mask_7x7, imaging_plotter_path, plot_patch
):

    aa.plot.imaging.image(
        imaging=imaging_7x7,
        positions=positions_7x7,
        mask=mask_7x7,
        include=aa.plotter.Include(mask=True),
        array_plotter=aa.plotter.array(
            output=aa.plotter.Output(imaging_plotter_path, format="png")
        ),
    )

    assert imaging_plotter_path + "image.png" in plot_patch.paths

    aa.plot.imaging.noise_map(
        imaging=imaging_7x7,
        mask=mask_7x7,
        array_plotter=aa.plotter.array(
            output=aa.plotter.Output(imaging_plotter_path, format="png")
        ),
    )

    assert imaging_plotter_path + "noise_map.png" in plot_patch.paths

    aa.plot.imaging.psf(
        imaging=imaging_7x7,
        array_plotter=aa.plotter.array(
            output=aa.plotter.Output(imaging_plotter_path, format="png")
        ),
    )

    assert imaging_plotter_path + "psf.png" in plot_patch.paths

    aa.plot.imaging.signal_to_noise_map(
        imaging=imaging_7x7,
        mask=mask_7x7,
        array_plotter=aa.plotter.array(
            output=aa.plotter.Output(imaging_plotter_path, format="png")
        ),
    )

    assert imaging_plotter_path + "signal_to_noise_map.png" in plot_patch.paths

    aa.plot.imaging.subplot(
        imaging=imaging_7x7,
        array_plotter=aa.plotter.array(
            output=aa.plotter.Output(imaging_plotter_path, format="png")
        ),
    )

    assert imaging_plotter_path + "imaging.png" in plot_patch.paths


def test__imaging_individuals__output_dependent_on_input(
    imaging_7x7, imaging_plotter_path, plot_patch
):
    aa.plot.imaging.individual(
        imaging=imaging_7x7,
        plot_image=True,
        plot_psf=True,
        plot_absolute_signal_to_noise_map=True,
        array_plotter=aa.plotter.array(
            output=aa.plotter.Output(imaging_plotter_path, format="png")
        ),
    )

    assert imaging_plotter_path + "image.png" in plot_patch.paths

    assert not imaging_plotter_path + "noise_map.png" in plot_patch.paths

    assert imaging_plotter_path + "psf.png" in plot_patch.paths

    assert not imaging_plotter_path + "signal_to_noise_map.png" in plot_patch.paths

    assert imaging_plotter_path + "absolute_signal_to_noise_map.png" in plot_patch.paths

    assert (
        not imaging_plotter_path + "potential_chi_squared_map.png" in plot_patch.paths
    )
