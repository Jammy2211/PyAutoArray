import autoarray as aa
import pytest
import os


@pytest.fixture(name="fit_plotter_util_path")
def make_fit_plotter_util_path_setup():
    return "{}/../../test_files/plotting/fit_plotter_util/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


def test__fit_quantities_are_output(
    fit_7x7, fit_plotter_util_path, plot_patch
):

    aa.plot.fit.image(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_image.png" in plot_patch.paths

    aa.plot.fit.noise_map(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_noise_map.png" in plot_patch.paths

    aa.plot.fit.signal_to_noise_map(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_signal_to_noise_map.png" in plot_patch.paths

    aa.plot.fit.model_image(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_model_image.png" in plot_patch.paths

    aa.plot.fit.residual_map(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_residual_map.png" in plot_patch.paths

    aa.plot.fit.normalized_residual_map(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert (
        fit_plotter_util_path + "fit_normalized_residual_map.png" in plot_patch.paths
    )

    aa.plot.fit.chi_squared_map(
        fit=fit_7x7,
        mask_overlay=fit_7x7.mask,
        cb_tick_values=[1.0],
        cb_tick_labels=["1.0"],
        output_path=fit_plotter_util_path,
        output_format="png",
    )

    assert fit_plotter_util_path + "fit_chi_squared_map.png" in plot_patch.paths