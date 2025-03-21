import autoarray as aa
import autoarray.plot as aplt
import pytest
from os import path

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "fit_dataset",
    )


def test__fit_quantities_are_output(fit_imaging_7x7, plot_path, plot_patch):
    fit_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7,
        include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_plotter.figures_2d(
        data=True,
        noise_map=True,
        signal_to_noise_map=True,
        model_image=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    plot_patch.paths = []

    fit_plotter.figures_2d(
        data=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths


def test__fit_sub_plot(fit_imaging_7x7, plot_path, plot_patch):
    fit_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7,
        include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_plotter.subplot_fit()

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    fit_imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    fit_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7,
        include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="fits")),
    )

    fit_plotter.figures_2d(data=True)

    image_from_plot = aa.ndarray_via_fits_from(
        file_path=path.join(plot_path, "data.fits"), hdu=0
    )

    assert image_from_plot.shape == (7, 7)
