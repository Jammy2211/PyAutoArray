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


def test__visuals_in_constructor_use_imaging_and_include(imaging_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)

    include = aplt.Include2D(origin=True, mask=True, border=True)

    imaging_plotter = aplt.ImagingPlotter(
        imaging=imaging_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert imaging_plotter.visuals_2d.origin == (1.0, 1.0)
    assert imaging_plotter.visuals_with_include_2d.origin == (1.0, 1.0)

    assert imaging_plotter.visuals_2d.mask == None
    assert (
        imaging_plotter.visuals_with_include_2d.mask == imaging_7x7.image.mask
    ).all()

    assert imaging_plotter.visuals_2d.border == None
    assert (
        imaging_plotter.visuals_with_include_2d.border
        == imaging_7x7.image.mask.geometry.border_grid_sub_1.in_1d_binned
    ).all()

    assert imaging_plotter.visuals_2d.vector_field == 2
    assert imaging_plotter.visuals_with_include_2d.vector_field == 2

    include = aplt.Include2D(origin=False, mask=False, border=False)

    imaging_plotter = aplt.ImagingPlotter(
        imaging=imaging_7x7, visuals_2d=visuals_2d, include_2d=include
    )

    assert imaging_plotter.visuals_with_include_2d.origin == (1.0, 1.0)
    assert imaging_plotter.visuals_with_include_2d.mask == None
    assert imaging_plotter.visuals_with_include_2d.border == None
    assert imaging_plotter.visuals_with_include_2d.vector_field == 2


def test__individual_attributes_are_output(
    imaging_7x7, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    visuals_2d = aplt.Visuals2D(mask=mask_7x7, positions=grid_irregular_grouped_7x7)

    imaging_plot = aplt.ImagingPlotter(
        imaging=imaging_7x7,
        visuals_2d=visuals_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    imaging_plot.figure_image()
    assert path.join(plot_path, "image.png") in plot_patch.paths

    imaging_plot.figure_noise_map()
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    imaging_plot.figure_psf()
    assert path.join(plot_path, "psf.png") in plot_patch.paths

    imaging_plot.figure_inverse_noise_map()
    assert path.join(plot_path, "inverse_noise_map.png") in plot_patch.paths

    imaging_plot.figure_signal_to_noise_map()
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths


def test__subplot_is_output(
    imaging_7x7, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    imaging_plot = aplt.ImagingPlotter(
        imaging=imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    imaging_plot.subplot_imaging()

    assert path.join(plot_path, "subplot_imaging.png") in plot_patch.paths


def test__imaging_individuals__output_dependent_on_input(
    imaging_7x7, plot_path, plot_patch
):

    imaging_plot = aplt.ImagingPlotter(
        imaging=imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    imaging_plot.figure_individuals(
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
        imaging=imaging_7x7,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="fits")),
    )

    imaging_plot.figure_individuals(
        plot_image=True, plot_psf=True, plot_absolute_signal_to_noise_map=True
    )

    image_from_plot = aa.util.array.numpy_array_2d_from_fits(
        file_path=path.join(plot_path, "image.fits"), hdu=0
    )

    assert image_from_plot.shape == (7, 7)
