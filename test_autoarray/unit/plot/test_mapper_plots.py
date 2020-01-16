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


@pytest.fixture(name="image")
def make_image():
    image = aa.array.ones(shape_2d=(20, 20), pixel_scales=1.0)
    noise_map = aa.array.full(fill_value=2.0, shape_2d=(20, 20), pixel_scales=1.0)
    psf = aa.kernel.manual_2d(array=3.0 * np.ones((20, 20)), pixel_scales=1.0)

    return aa.imaging.manual(image=image, noise_map=noise_map, psf=psf)


@pytest.fixture(name="mask")
def make_mask():
    return aa.mask.circular(shape_2d=((20, 20)), pixel_scales=0.1, radius=0.1)


@pytest.fixture(name="grid")
def make_grid():
    return aa.grid.uniform(shape_2d=(20, 20), pixel_scales=0.1, sub_size=2)


@pytest.fixture(name="rectangular_pixelization")
def make_rectangular_pixelization():
    return aa.pix.Rectangular(shape=(25, 25))


@pytest.fixture(name="rectangular_mapper")
def make_rectangular_mapper(rectangular_pixelization, grid):
    return rectangular_pixelization.mapper_from_grid_and_sparse_grid(
        grid=grid, sparse_grid=None, inversion_uses_border=False
    )


def test__image_and_rectangular_mapper_is_output(
    image, rectangular_mapper, plotter_path, plot_patch
):
    aa.plot.mapper.subplot_image_and_mapper(
        imaging=image,
        mapper=rectangular_mapper,
        include=aplt.Include(
            inversion_centres=True, inversion_grid=True, inversion_border=False
        ),
        image_pixels=[[0, 1, 2], [3]],
        source_pixels=[[1, 2], [0]],
        sub_plotter=aplt.SubPlotter(
            output=aplt.Output(path=plotter_path, format="png")
        ),
    )
    assert plotter_path + "subplot_image_and_mapper.png" in plot_patch.paths


def test__rectangular_mapper_is_output(rectangular_mapper, plotter_path, plot_patch):
    aa.plot.mapper.mapper_grid(
        mapper=rectangular_mapper,
        include=aplt.Include(
            inversion_centres=True, inversion_grid=True, inversion_border=False
        ),
        image_pixels=[[0, 1, 2], [3]],
        source_pixels=[[1, 2], [0]],
        plotter=aplt.Plotter(
            output=aplt.Output(path=plotter_path, format="png")
        ),
    )

    assert plotter_path + "mapper_grid.png" in plot_patch.paths
