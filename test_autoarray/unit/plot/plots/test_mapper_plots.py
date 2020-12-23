from os import path

from autoconf import conf
import autoarray.plot as aplt

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "mapper"
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files", "plotter"), path.join(directory, "output")
    )


def test__image_and_mapper_subplot_is_output_for_all_mappers(
    imaging_7x7, rectangular_mapper_7x7_3x3, voronoi_mapper_9_3x3, plot_path, plot_patch
):
    aplt.Mapper.subplot_image_and_mapper(
        image=imaging_7x7.image,
        mapper=rectangular_mapper_7x7_3x3,
        include=aplt.Include(
            inversion_pixelization_grid=True, inversion_grid=True, inversion_border=True
        ),
        image_pixel_indexes=[[0, 1, 2], [3]],
        source_pixel_indexes=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

    aplt.Mapper.subplot_image_and_mapper(
        image=imaging_7x7.image,
        mapper=voronoi_mapper_9_3x3,
        include=aplt.Include(
            inversion_pixelization_grid=True, inversion_grid=True, inversion_border=True
        ),
        image_pixel_indexes=[[0, 1, 2], [3]],
        source_pixel_indexes=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(output=aplt.Output(path=plot_path, format="png")),
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths
