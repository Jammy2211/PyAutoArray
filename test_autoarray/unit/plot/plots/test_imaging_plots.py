from os import path
import pytest

from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files", "plotter"), path.join(directory, "output")
    )


def test__individual_attributes_are_output(
    imaging_7x7, positions_7x7, mask_7x7, plot_path, plot_patch
):

    aplt.Imaging.image(
        imaging=imaging_7x7,
        positions=positions_7x7,
        mask=mask_7x7,
        include=aplt.Include(mask=True),
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths

    aplt.Imaging.noise_map(
        imaging=imaging_7x7,
        mask=mask_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    aplt.Imaging.psf(
        imaging=imaging_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "psf.png") in plot_patch.paths

    aplt.Imaging.inverse_noise_map(
        imaging=imaging_7x7,
        mask=mask_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths

    aplt.Imaging.signal_to_noise_map(
        imaging=imaging_7x7,
        mask=mask_7x7,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths


def test__subplot_is_output(
    imaging_7x7, positions_7x7, mask_7x7, plot_path, plot_patch
):

    aplt.Imaging.subplot_imaging(
        imaging=imaging_7x7,
        sub_plotter=aplt.SubPlotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_imaging.png") in plot_patch.paths


def test__imaging_individuals__output_dependent_on_input(
    imaging_7x7, plot_path, plot_patch
):
    aplt.Imaging.individual(
        imaging=imaging_7x7,
        plot_image=True,
        plot_psf=True,
        plot_inverse_noise_map=True,
        plot_absolute_signal_to_noise_map=True,
        plotter=aplt.Plotter(output=aplt.Output(plot_path, format="png")),
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths

    assert not path.join(plot_path, "noise_map.png") in plot_patch.paths

    assert path.join(plot_path, "psf.png") in plot_patch.paths

    assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths

    assert not path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    assert path.join(plot_path, "absolute_signal_to_noise_map.png") in plot_patch.paths

    assert not path.join(plot_path, "potential_chi_squared_map.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    imaging_7x7, positions_7x7, mask_7x7, plot_path, plot_patch
):

    aplt.Imaging.individual(
        imaging=imaging_7x7,
        plot_image=True,
        plot_psf=True,
        plot_absolute_signal_to_noise_map=True,
        plotter=aplt.Plotter(output=aplt.Output(path=plot_path, format="fits")),
    )

    image_from_plot = aa.util.array.numpy_array_2d_from_fits(
        file_path=path.join(plot_path, "image.fits"), hdu=0
    )

    assert image_from_plot.shape == (7, 7)
