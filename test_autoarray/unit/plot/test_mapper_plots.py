from os import path

import autoarray as aa
import autoarray.plot as aplt
import os

import numpy as np
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plotter_path")
def make_plotter_setup():
    return "{}/../../../test_files/plotting/mapper/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


def test__image_and_mapper_subplot_is_output_for_all_mappers(
    imaging_7x7, rectangular_mapper_7x7_3x3, voronoi_mapper_9_3x3, plotter_path, plot_patch
):
    aa.plot.mapper.subplot_image_and_mapper(
        imaging=imaging_7x7,
        mapper=rectangular_mapper_7x7_3x3,
        include=aplt.Include(
            inversion_pixelization_grid=True,
            inversion_grid=True,
            inversion_border=True,
        ),
        image_pixel_indexes=[[0, 1, 2], [3]],
        source_pixel_indexes=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=plotter_path, format="png")
        ),
    )
    assert plotter_path + "subplot_image_and_mapper.png" in plot_patch.paths

    aa.plot.mapper.subplot_image_and_mapper(
        imaging=imaging_7x7,
        mapper=voronoi_mapper_9_3x3,
        include=aplt.Include(
            inversion_pixelization_grid=True,
            inversion_grid=True,
            inversion_border=True,
        ),
        image_pixel_indexes=[[0, 1, 2], [3]],
        source_pixel_indexes=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=plotter_path, format="png")
        ),
    )
    assert plotter_path + "subplot_image_and_mapper.png" in plot_patch.paths
