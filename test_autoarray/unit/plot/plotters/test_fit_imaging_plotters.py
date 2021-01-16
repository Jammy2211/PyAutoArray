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
        "fit_imaging",
    )


def test__visuals_in_constructor_use_imaging_and_include(fit_imaging_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

    include = aplt.Include2D(origin=True, mask=True, border=True)

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert fit_imaging_plotter.visuals_2d.origin == (1.0, 1.0)
    assert fit_imaging_plotter.visuals_with_include_2d.origin == (1.0, 1.0)

    assert (
        fit_imaging_plotter.visuals_with_include_2d.mask == fit_imaging_7x7.image.mask
    ).all()
    assert (
        fit_imaging_plotter.visuals_with_include_2d.border
        == fit_imaging_7x7.image.mask.geometry.border_grid_sub_1.in_1d_binned
    ).all()
    assert fit_imaging_plotter.visuals_with_include_2d.vector_field == 2

    include = aplt.Include2D(origin=False, mask=False, border=False)

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert fit_imaging_plotter.visuals_with_include_2d.origin == (1.0, 1.0)
    assert fit_imaging_plotter.visuals_with_include_2d.mask == None
    assert fit_imaging_plotter.visuals_with_include_2d.border == None
    assert fit_imaging_plotter.visuals_with_include_2d.vector_field == 2


def test__fit_quantities_are_output(fit_imaging_7x7, plot_path, plot_patch):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7,
        include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_imaging_plotter.figures(
        image=True,
        noise_map=True,
        signal_to_noise_map=True,
        model_image=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    plot_patch.paths = []

    fit_imaging_plotter.figures(
        image=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths


def test__fit_sub_plot(fit_imaging_7x7, plot_path, plot_patch):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7,
        include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_imaging_plotter.subplot_fit_imaging()

    assert path.join(plot_path, "subplot_fit_imaging.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    fit_imaging_7x7, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    fit_imaging_plotter = aplt.FitImagingPlotter(
        fit=fit_imaging_7x7,
        include_2d=aplt.Include2D(origin=True, mask=True, border=True),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_imaging_plotter.figures(image=True)

    image_from_plot = aa.util.array.numpy_array_2d_from_fits(
        file_path=path.join(plot_path, "image.fits"), hdu=0
    )

    assert image_from_plot.shape == (5, 5)
