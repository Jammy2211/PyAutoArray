from os import path
import pytest
import autoarray as aa
import autoarray.plot as aplt


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__individual_attributes_are_output(
    imaging_7x7, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    visuals_2d = aplt.Visuals2D(mask=mask_7x7, positions=grid_irregular_grouped_7x7)

    imaging_plot = aplt.ImagingPlotter(
        visuals_2d=visuals_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    imaging_plot.image(imaging=imaging_7x7)
    assert path.join(plot_path, "image.png") in plot_patch.paths

    imaging_plot.noise_map(imaging=imaging_7x7)
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    imaging_plot.psf(imaging=imaging_7x7)
    assert path.join(plot_path, "psf.png") in plot_patch.paths

    imaging_plot.inverse_noise_map(imaging=imaging_7x7)
    assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths

    imaging_plot.signal_to_noise_map(imaging=imaging_7x7)
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths


def test__subplot_is_output(
    imaging_7x7, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    imaging_plot = aplt.ImagingPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png"))
    )

    imaging_plot.subplot_imaging(imaging=imaging_7x7)

    assert path.join(plot_path, "subplot_imaging.png") in plot_patch.paths


def test__imaging_individuals__output_dependent_on_input(
    imaging_7x7, plot_path, plot_patch
):

    imaging_plot = aplt.ImagingPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png"))
    )

    imaging_plot.individual(
        imaging=imaging_7x7,
        plot_image=True,
        plot_psf=True,
        plot_inverse_noise_map=True,
        plot_absolute_signal_to_noise_map=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert not path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "psf.png") in plot_patch.paths
    assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths
    assert not path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "absolute_signal_to_noise_map.png") in plot_patch.paths
    assert not path.join(plot_path, "potential_chi_squared_map.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    imaging_7x7, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    imaging_plot = aplt.ImagingPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="fits"))
    )

    imaging_plot.individual(
        imaging=imaging_7x7,
        plot_image=True,
        plot_psf=True,
        plot_absolute_signal_to_noise_map=True,
    )

    image_from_plot = aa.util.array.numpy_array_2d_from_fits(
        file_path=path.join(plot_path, "image.fits"), hdu=0
    )

    assert image_from_plot.shape == (7, 7)
