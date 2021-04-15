from os import path
import autoarray.plot as aplt

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "inversion",
    )


def test__individual_attributes_are_output_for_all_mappers(
    rectangular_inversion_7x7_3x3,
    voronoi_inversion_9_3x3,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):

    inversion_plotter = aplt.InversionPlotter(
        inversion=rectangular_inversion_7x7_3x3,
        visuals_2d=aplt.Visuals2D(indexes=[0], pixelization_indexes=[1]),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    inversion_plotter.figures_2d(
        reconstructed_image=True,
        reconstruction=True,
        errors=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
        regularization_weights=True,
        interpolated_reconstruction=True,
        interpolated_errors=True,
    )

    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths
    assert path.join(plot_path, "reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "errors.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
    assert path.join(plot_path, "regularization_weights.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_errors.png") in plot_patch.paths

    plot_patch.paths = []

    inversion_plotter = aplt.InversionPlotter(
        inversion=voronoi_inversion_9_3x3,
        visuals_2d=aplt.Visuals2D(indexes=[0], pixelization_indexes=[1]),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    inversion_plotter.figures_2d(
        reconstructed_image=True,
        reconstruction=True,
        errors=True,
        residual_map=True,
        normalized_residual_map=True,
        chi_squared_map=True,
        regularization_weights=True,
        interpolated_reconstruction=True,
        interpolated_errors=True,
    )

    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths
    assert path.join(plot_path, "reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "errors.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
    assert path.join(plot_path, "regularization_weights.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_errors.png") in plot_patch.paths

    plot_patch.paths = []

    inversion_plotter.figures_2d(
        reconstructed_image=True,
        errors=True,
        chi_squared_map=True,
        interpolated_reconstruction=True,
    )

    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths
    assert path.join(plot_path, "reconstruction.png") not in plot_patch.paths
    assert path.join(plot_path, "errors.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_errors.png") not in plot_patch.paths


def test__inversion_subplot_is_output_for_all_inversions(
    imaging_7x7,
    rectangular_inversion_7x7_3x3,
    voronoi_inversion_9_3x3,
    plot_path,
    plot_patch,
):

    inversion_plotter = aplt.InversionPlotter(
        inversion=rectangular_inversion_7x7_3x3,
        visuals_2d=aplt.Visuals2D(indexes=[0], pixelization_indexes=[1]),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    inversion_plotter.subplot_inversion()
    assert path.join(plot_path, "subplot_inversion.png") in plot_patch.paths
