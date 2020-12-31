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


def test__individual_attributes_are_output_for_rectangular_inversion(
    rectangular_inversion_7x7_3x3, grid_irregular_grouped_7x7, plot_path, plot_patch
):

    inversion_plotter = aplt.InversionPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    )

    inversion_plotter.reconstructed_image(inversion=rectangular_inversion_7x7_3x3)

    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths

    inversion_plotter.reconstruction(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[0],
        pixelization_indexes=[1],
    )

    assert path.join(plot_path, "reconstruction.png") in plot_patch.paths

    inversion_plotter.errors(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[0],
        pixelization_indexes=[1],
    )

    assert path.join(plot_path, "errors.png") in plot_patch.paths

    inversion_plotter.residual_map(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[0],
        pixelization_indexes=[1],
    )

    assert path.join(plot_path, "residual_map.png") in plot_patch.paths

    inversion_plotter.normalized_residual_map(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[0],
        pixelization_indexes=[1],
    )

    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths

    inversion_plotter.chi_squared_map(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[0],
        pixelization_indexes=[1],
    )

    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    inversion_plotter.regularization_weights(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[0],
        pixelization_indexes=[1],
    )

    assert path.join(plot_path, "regularization_weights.png") in plot_patch.paths

    inversion_plotter.interpolated_reconstruction(
        inversion=rectangular_inversion_7x7_3x3
    )

    assert path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths

    inversion_plotter.interpolated_errors(inversion=rectangular_inversion_7x7_3x3)

    assert path.join(plot_path, "interpolated_errors.png") in plot_patch.paths


def test__individual_attributes_are_output_for_voronoi_inversion(
    voronoi_inversion_9_3x3, grid_irregular_grouped_7x7, mask_7x7, plot_path, plot_patch
):

    inversion_plotter = aplt.InversionPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    )

    inversion_plotter.reconstructed_image(inversion=voronoi_inversion_9_3x3)

    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths

    inversion_plotter.reconstruction(
        inversion=voronoi_inversion_9_3x3, full_indexes=[0], pixelization_indexes=[1]
    )

    assert path.join(plot_path, "reconstruction.png") in plot_patch.paths

    inversion_plotter.errors(
        inversion=voronoi_inversion_9_3x3, full_indexes=[0], pixelization_indexes=[1]
    )

    assert path.join(plot_path, "errors.png") in plot_patch.paths

    inversion_plotter.residual_map(
        inversion=voronoi_inversion_9_3x3, full_indexes=[0], pixelization_indexes=[1]
    )

    assert path.join(plot_path, "residual_map.png") in plot_patch.paths

    inversion_plotter.normalized_residual_map(
        inversion=voronoi_inversion_9_3x3, full_indexes=[0], pixelization_indexes=[1]
    )

    assert path.join(plot_path, "normalized_residual_map.png") in plot_patch.paths

    inversion_plotter.chi_squared_map(
        inversion=voronoi_inversion_9_3x3, full_indexes=[0], pixelization_indexes=[1]
    )

    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    inversion_plotter.regularization_weights(
        inversion=voronoi_inversion_9_3x3, full_indexes=[0], pixelization_indexes=[1]
    )

    assert path.join(plot_path, "regularization_weights.png") in plot_patch.paths

    inversion_plotter.interpolated_reconstruction(inversion=voronoi_inversion_9_3x3)

    assert path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths

    inversion_plotter.interpolated_errors(inversion=voronoi_inversion_9_3x3)

    assert path.join(plot_path, "interpolated_errors.png") in plot_patch.paths


def test__inversion_subplot_is_output_for_all_inversions(
    imaging_7x7,
    rectangular_inversion_7x7_3x3,
    voronoi_inversion_9_3x3,
    plot_path,
    plot_patch,
):

    inversion_plotter = aplt.InversionPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    )

    inversion_plotter.subplot_inversion(
        inversion=rectangular_inversion_7x7_3x3,
        full_indexes=[[0, 1, 2], [3]],
        pixelization_indexes=[[1, 2], [0]],
    )
    assert path.join(plot_path, "subplot_inversion.png") in plot_patch.paths

    inversion_plotter.subplot_inversion(
        inversion=voronoi_inversion_9_3x3,
        full_indexes=[[0, 1, 2], [3]],
        pixelization_indexes=[[1, 2], [0]],
    )
    assert path.join(plot_path, "subplot_inversion.png") in plot_patch.paths


def test__inversion_individuals__output_dependent_on_input(
    rectangular_inversion_7x7_3x3,
    grid_irregular_grouped_7x7,
    mask_7x7,
    plot_path,
    plot_patch,
):

    inversion_plotter = aplt.InversionPlotter(
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png"))
    )

    inversion_plotter.individuals(
        inversion=rectangular_inversion_7x7_3x3,
        plot_reconstructed_image=True,
        plot_errors=True,
        plot_chi_squared_map=True,
        plot_interpolated_reconstruction=True,
    )

    assert path.join(plot_path, "reconstructed_image.png") in plot_patch.paths
    assert path.join(plot_path, "reconstruction.png") not in plot_patch.paths
    assert path.join(plot_path, "errors.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_reconstruction.png") in plot_patch.paths
    assert path.join(plot_path, "interpolated_errors.png") not in plot_patch.paths
