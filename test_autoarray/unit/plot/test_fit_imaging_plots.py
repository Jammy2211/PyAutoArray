import autoarray as aa
import autoarray.plot as aplt
import pytest
import os
from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="fit_imaging_path")
def make_fit_imaging_path_setup():
    return "{}/../../test_files/plotting/fit_imaging/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__fit_quantities_are_output(fit_imaging_7x7, fit_imaging_path, plot_patch):

    aa.plot.fit_imaging.image(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "image.png" in plot_patch.paths

    aa.plot.fit_imaging.noise_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "noise_map.png" in plot_patch.paths

    aa.plot.fit_imaging.signal_to_noise_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "signal_to_noise_map.png" in plot_patch.paths

    aa.plot.fit_imaging.model_image(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "model_image.png" in plot_patch.paths

    aa.plot.fit_imaging.residual_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "residual_map.png" in plot_patch.paths

    aa.plot.fit_imaging.normalized_residual_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "normalized_residual_map.png" in plot_patch.paths

    aa.plot.fit_imaging.chi_squared_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "chi_squared_map.png" in plot_patch.paths


def test__fit_sub_plot(fit_imaging_7x7, fit_imaging_path, plot_patch):

    aa.plot.fit_imaging.subplot_fit_imaging(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=fit_imaging_path, format="png")
        ),
    )

    assert fit_imaging_path + "subplot_fit_imaging.png" in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(
    fit_imaging_7x7, fit_imaging_path, plot_patch
):

    aa.plot.fit_imaging.individuals(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "image.png" in plot_patch.paths

    assert fit_imaging_path + "noise_map.png" not in plot_patch.paths

    assert fit_imaging_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert fit_imaging_path + "model_image.png" in plot_patch.paths

    assert fit_imaging_path + "residual_map.png" not in plot_patch.paths

    assert fit_imaging_path + "normalized_residual_map.png" not in plot_patch.paths

    assert fit_imaging_path + "chi_squared_map.png" in plot_patch.paths

    aa.plot.fit_imaging.individuals(
        fit=fit_imaging_7x7,
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plotter=aplt.Plotter(output=aplt.Output(path=fit_imaging_path, format="png")),
    )

    assert fit_imaging_path + "image.png" in plot_patch.paths

    assert fit_imaging_path + "noise_map.png" not in plot_patch.paths

    assert fit_imaging_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert fit_imaging_path + "model_image.png" in plot_patch.paths

    assert fit_imaging_path + "residual_map.png" not in plot_patch.paths

    assert fit_imaging_path + "normalized_residual_map.png" not in plot_patch.paths

    assert fit_imaging_path + "chi_squared_map.png" in plot_patch.paths
