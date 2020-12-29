from os import path

import pytest
import autoarray.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "interferometer",
    )


def test__individual_attributes_are_output(interferometer_7, plot_path, plot_patch):

    aplt.Interferometer.visibilities(
        interferometer=interferometer_7,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths

    aplt.Interferometer.noise_map(
        interferometer=interferometer_7,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    aplt.Interferometer.u_wavelengths(
        interferometer=interferometer_7,
        plotter_1d=aplt.Plotter1D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "u_wavelengths.png") in plot_patch.paths

    aplt.Interferometer.v_wavelengths(
        interferometer=interferometer_7,
        plotter_1d=aplt.Plotter1D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "v_wavelengths.png") in plot_patch.paths

    aplt.Interferometer.uv_wavelengths(
        interferometer=interferometer_7,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "uv_wavelengths.png") in plot_patch.paths

    aplt.Interferometer.amplitudes_vs_uv_distances(
        interferometer=interferometer_7,
        plotter_1d=aplt.Plotter1D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "amplitudes_vs_uv_distances.png") in plot_patch.paths

    aplt.Interferometer.phases_vs_uv_distances(
        interferometer=interferometer_7,
        plotter_1d=aplt.Plotter1D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "phases_vs_uv_distances.png") in plot_patch.paths


def test__subplot_is_output(interferometer_7, plot_path, plot_patch):

    print(plot_patch.paths)

    aplt.Interferometer.subplot_interferometer(
        interferometer=interferometer_7,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
    )

    assert path.join(plot_path, "subplot_interferometer.png") in plot_patch.paths


def test__individuals__output_dependent_on_input(
    interferometer_7, plot_path, plot_patch
):
    aplt.Interferometer.individual(
        interferometer=interferometer_7,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
        plotter_1d=aplt.Plotter1D(output=aplt.Output(path=plot_path, format="png")),
        plot_visibilities=True,
        plot_u_wavelengths=False,
        plot_v_wavelengths=True,
        plot_amplitudes_vs_uv_distances=True,
    )

    assert path.join(plot_path, "visibilities.png") in plot_patch.paths

    assert not path.join(plot_path, "u_wavelengths.png") in plot_patch.paths

    assert path.join(plot_path, "v_wavelengths.png") in plot_patch.paths

    assert path.join(plot_path, "amplitudes_vs_uv_distances.png") in plot_patch.paths

    assert path.join(plot_path, "phases_vs_uv_distances.png") not in plot_patch.paths
