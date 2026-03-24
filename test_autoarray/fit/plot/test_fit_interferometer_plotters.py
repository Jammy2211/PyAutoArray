import autoarray.plot as aplt
import numpy as np
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


def test__fit_quantities_are_output(fit_interferometer_7, plot_path, plot_patch):
    uv = fit_interferometer_7.dataset.uv_distances / 10**3.0

    aplt.plot_grid_2d(
        grid=fit_interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert path.join(plot_path, "data.png") in plot_patch.paths

    aplt.plot_yx_1d(
        y=np.real(fit_interferometer_7.residual_map),
        x=uv,
        output_path=plot_path,
        output_filename="real_residual_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        in plot_patch.paths
    )

    aplt.plot_yx_1d(
        y=np.real(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="real_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    assert (
        path.join(plot_path, "real_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )

    aplt.plot_yx_1d(
        y=np.imag(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="imag_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    assert (
        path.join(plot_path, "imag_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )

    aplt.plot_array_2d(
        array=fit_interferometer_7.dirty_image,
        output_path=plot_path,
        output_filename="dirty_image",
        output_format="png",
    )
    assert path.join(plot_path, "dirty_image.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_grid_2d(
        grid=fit_interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    aplt.plot_yx_1d(
        y=np.real(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="real_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )
    aplt.plot_yx_1d(
        y=np.imag(fit_interferometer_7.chi_squared_map),
        x=uv,
        output_path=plot_path,
        output_filename="imag_chi_squared_map_vs_uv_distances",
        output_format="png",
        plot_axis_type="scatter",
    )

    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert (
        path.join(plot_path, "real_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "imag_chi_squared_map_vs_uv_distances.png")
        in plot_patch.paths
    )
    assert (
        path.join(plot_path, "real_residual_map_vs_uv_distances.png")
        not in plot_patch.paths
    )


def test__fit_sub_plots(fit_interferometer_7, plot_path, plot_patch):
    aplt.subplot_fit_interferometer(
        fit=fit_interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_fit.png") in plot_patch.paths

    aplt.subplot_fit_interferometer_dirty_images(
        fit=fit_interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_fit_dirty_images.png") in plot_patch.paths
