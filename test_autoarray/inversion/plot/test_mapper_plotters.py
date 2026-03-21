from os import path
import pytest

import autoarray as aa
import autoarray.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "structures"
    )


def test__plot_mapper(
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    plot_path,
    plot_patch,
):
    aplt.plot_mapper(
        mapper=rectangular_mapper_7x7_3x3,
        output_path=plot_path,
        output_filename="mapper1",
        output_format="png",
    )

    assert path.join(plot_path, "mapper1.png") in plot_patch.paths


def test__subplot_image_and_mapper(
    imaging_7x7,
    rectangular_mapper_7x7_3x3,
    delaunay_mapper_9_3x3,
    plot_path,
    plot_patch,
):
    aplt.subplot_image_and_mapper(
        mapper=rectangular_mapper_7x7_3x3,
        image=imaging_7x7.data,
        output_path=plot_path,
        output_format="png",
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths
