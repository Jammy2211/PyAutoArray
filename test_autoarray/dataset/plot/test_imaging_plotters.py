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
    imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    visuals = aplt.Visuals2D(mask=mask_2d_7x7, positions=grid_2d_irregular_7x7_list)

    dataset_plotter = aplt.ImagingPlotter(
        dataset=imaging_7x7,
        visuals_2d=visuals,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_plotter.figures_2d(
        data=True,
        noise_map=True,
        psf=True,
        signal_to_noise_map=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "psf.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    plot_patch.paths = []

    dataset_plotter.figures_2d(
        data=True,
        psf=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert not path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "psf.png") in plot_patch.paths
    assert not path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths


def test__subplot_is_output(
    imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    dataset_plotter = aplt.ImagingPlotter(
        dataset=imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    dataset_plotter.subplot_dataset()

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    dataset_plotter = aplt.ImagingPlotter(
        dataset=imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="fits")),
    )

    dataset_plotter.figures_2d(data=True, psf=True)

    image_from_plot = aa.util.array_2d.numpy_array_2d_via_fits_from(
        file_path=path.join(plot_path, "data.fits"), hdu=0
    )

    assert image_from_plot.shape == (7, 7)
