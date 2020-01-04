import autoarray as aa
import pytest
import os


@pytest.fixture(name="fit_plotter_util_path")
def make_fit_plotter_util_path_setup():
    return "{}/../../test_files/plotting/fit_plotter_util/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__fit_sub_plot(fit_interferometer_7, fit_plotter_util_path, plot_patch):

    aa.plot.fit_interferometer.subplot(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit.png" in plot_patch.paths


def test__fit_individuals__depedent_on_input(
    fit_interferometer_7, fit_plotter_util_path, plot_patch
):

    aa.plot.fit_interferometer.individuals(
        fit=fit_interferometer_7,
        plot_visibilities=True,
        plot_noise_map=False,
        plot_signal_to_noise_map=False,
        plot_model_visibilities=True,
        plot_chi_squared_map=True,
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_visibilities.png" in plot_patch.paths

    assert fit_plotter_util_path + "fit_noise_map.png" not in plot_patch.paths

    assert fit_plotter_util_path + "fit_signal_to_noise_map.png" not in plot_patch.paths

    assert fit_plotter_util_path + "fit_model_visibilities.png" in plot_patch.paths

    assert (
        fit_plotter_util_path + "fit_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        fit_plotter_util_path + "fit_residual_map_vs_uv_distances_imag.png"
        not in plot_patch.paths
    )

    assert (
        fit_plotter_util_path + "fit_normalized_residual_map_vs_uv_distances_real.png"
        not in plot_patch.paths
    )

    assert (
        fit_plotter_util_path + "fit_normalized_residual_map_vs_uv_distances_imag.png"
        not in plot_patch.paths
    )

    assert (
        fit_plotter_util_path + "fit_chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    assert (
        fit_plotter_util_path + "fit_chi_squared_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )


def test__fit_quantities_are_output(
    fit_interferometer_7, fit_plotter_util_path, plot_patch
):

    aa.plot.fit_interferometer.visibilities(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_visibilities.png" in plot_patch.paths

    aa.plot.fit_interferometer.noise_map(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_noise_map.png" in plot_patch.paths

    aa.plot.fit_interferometer.signal_to_noise_map(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_signal_to_noise_map.png" in plot_patch.paths

    aa.plot.fit_interferometer.model_visibilities(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_model_visibilities.png" in plot_patch.paths

    aa.plot.fit_interferometer.residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_residual_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plot_real=False,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_residual_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.normalized_residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_normalized_residual_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.normalized_residual_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plot_real=False,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_normalized_residual_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.chi_squared_map_vs_uv_distances(
        fit=fit_interferometer_7,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_chi_squared_map_vs_uv_distances_real.png"
        in plot_patch.paths
    )

    aa.plot.fit_interferometer.chi_squared_map_vs_uv_distances(
        fit=fit_interferometer_7,
        plot_real=False,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_chi_squared_map_vs_uv_distances_imag.png"
        in plot_patch.paths
    )
