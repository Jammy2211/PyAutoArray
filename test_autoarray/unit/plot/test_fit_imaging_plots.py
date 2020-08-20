from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt
import pytest
import os
from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return "{}/files/plots/fit_imaging/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


def test__fit_quantities_are_output(fit_imaging_7x7, plot_path, plot_patch):

    aplt.FitImaging.image(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "image.png" in plot_patch.paths

    aplt.FitImaging.noise_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "noise_map.png" in plot_patch.paths

    aplt.FitImaging.signal_to_noise_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "signal_to_noise_map.png" in plot_patch.paths

    aplt.FitImaging.model_image(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "model_image.png" in plot_patch.paths

    aplt.FitImaging.residual_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "residual_map.png" in plot_patch.paths

    aplt.FitImaging.normalized_residual_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "normalized_residual_map.png" in plot_patch.paths

    aplt.FitImaging.chi_squared_map(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "chi_squared_map.png" in plot_patch.paths


def test__fit_sub_plot(fit_imaging_7x7, plot_path, plot_patch):

    aplt.FitImaging.subplot_fit_imaging(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "subplot_fit_imaging.png" in plot_patch.paths


def test__fit_individuals__source_and_lens__depedent_on_input(
    fit_imaging_7x7, plot_path, plot_patch
):

    aplt.FitImaging.individuals(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "image.png" in plot_patch.paths

    assert plot_path + "noise_map.png" not in plot_patch.paths

    assert plot_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert plot_path + "model_image.png" in plot_patch.paths

    assert plot_path + "residual_map.png" not in plot_patch.paths

    assert plot_path + "normalized_residual_map.png" not in plot_patch.paths

    assert plot_path + "chi_squared_map.png" in plot_patch.paths

    aplt.FitImaging.individuals(
        fit=fit_imaging_7x7,
        plot_image=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_image=True,
        plot_chi_squared_map=True,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="png")),
    )

    assert plot_path + "image.png" in plot_patch.paths

    assert plot_path + "noise_map.png" not in plot_patch.paths

    assert plot_path + "signal_to_noise_map.png" not in plot_patch.paths

    assert plot_path + "model_image.png" in plot_patch.paths

    assert plot_path + "residual_map.png" not in plot_patch.paths

    assert plot_path + "normalized_residual_map.png" not in plot_patch.paths

    assert plot_path + "chi_squared_map.png" in plot_patch.paths


def test__output_as_fits__correct_output_format(
    fit_imaging_7x7, positions_7x7, mask_7x7, plot_path, plot_patch
):

    aplt.FitImaging.individuals(
        fit=fit_imaging_7x7,
        include=aplt.Include(mask=True),
        plot_image=True,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="fits")),
    )

    image_from_plot = aa.util.array.numpy_array_2d_from_fits(
        file_path=plot_path + "image.fits", hdu=0
    )

    assert image_from_plot.shape == (5, 5)
