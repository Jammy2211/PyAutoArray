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
    aplt.plot_grid_2d(
        grid=interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert path.join(plot_path, "data.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=interferometer_7.dirty_image,
        output_path=plot_path,
        output_filename="dirty_image",
        output_format="png",
    )
    assert path.join(plot_path, "dirty_image.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=interferometer_7.dirty_noise_map,
        output_path=plot_path,
        output_filename="dirty_noise_map",
        output_format="png",
    )
    assert path.join(plot_path, "dirty_noise_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=interferometer_7.dirty_signal_to_noise_map,
        output_path=plot_path,
        output_filename="dirty_signal_to_noise_map",
        output_format="png",
    )
    assert path.join(plot_path, "dirty_signal_to_noise_map.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_grid_2d(
        grid=interferometer_7.data.in_grid,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert not path.join(plot_path, "dirty_image.png") in plot_patch.paths


def test__subplots_are_output(interferometer_7, plot_path, plot_patch):
    aplt.subplot_interferometer_dataset(
        dataset=interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_dataset.png") in plot_patch.paths

    aplt.subplot_interferometer_dirty_images(
        dataset=interferometer_7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "subplot_dirty_images.png") in plot_patch.paths
