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
        "fit_interferometer",
    )


def test__fit_quantities_are_output(fit_interferometer_7, plot_path, plot_patch):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_interferometer_plotter.figures_2d(
        data=True,
        noise_map=True,
        signal_to_noise_map=True,
        model_visibilities=True,
        residual_map_real=True,
        residual_map_imag=True,
        normalized_residual_map_real=True,
        normalized_residual_map_imag=True,
        chi_squared_map_real=True,
        chi_squared_map_imag=True,
        dirty_image=True,
        dirty_noise_map=True,
        dirty_signal_to_noise_map=True,
        dirty_model_image=True,
        dirty_residual_map=True,
        dirty_normalized_residual_map=True,
        dirty_chi_squared_map=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths
    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_normalized_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_normalized_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert path.join(plot_path, "dirty_image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "dirty_noise_map_2d.png") in plot_patch.paths
    assert path.join(plot_path, "dirty_signal_to_noise_map_2d.png") in plot_patch.paths
    assert path.join(plot_path, "dirty_model_image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "dirty_residual_map_2d.png") in plot_patch.paths
    assert (
        path.join(plot_path, "dirty_normalized_residual_map_2d.png") in plot_patch.paths
    )
    assert path.join(plot_path, "dirty_chi_squared_map_2d.png") in plot_patch.paths

    plot_patch.paths = []

    fit_interferometer_plotter.figures_2d(
        data=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_visibilities=True,
        chi_squared_map_real=True,
        chi_squared_map_imag=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_visibilities.png") in plot_patch.paths
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_normalized_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_normalized_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )


def test__fit_sub_plots(fit_interferometer_7, plot_path, plot_patch):

    fit_interferometer_plotter = aplt.FitInterferometerPlotter(
        fit=fit_interferometer_7,
        mat_plot_1d=aplt.MatPlot1D(output=aplt.Output(path=plot_path, format="png")),
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_interferometer_plotter.subplot_fit_interferometer()

    assert path.join(plot_path, "subplot_fit_interferometer.png") in plot_patch.paths

    fit_interferometer_plotter.subplot_fit_dirty_images()

    assert path.join(plot_path, "subplot_fit_dirty_images.png") in plot_patch.paths
