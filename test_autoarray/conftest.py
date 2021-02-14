from matplotlib import pyplot

from os import path

import pytest

from autoarray.mock import fixtures
from autoconf import conf


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path(request):

    # if dirname(realpath(__file__)) in str(request.module):

    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    return fixtures.make_mask_7x7()


@pytest.fixture(name="sub_mask_7x7")
def make_sub_mask_7x7():
    return fixtures.make_sub_mask_7x7()


@pytest.fixture(name="mask_7x7_1_pix")
def make_mask_7x7_1_pix():
    return fixtures.make_mask_7x7_1_pix()


@pytest.fixture(name="array_7x7")
def make_array_7x7():
    return fixtures.make_array_7x7()


@pytest.fixture(name="scans_7x7")
def make_scans_7x7():
    return fixtures.make_scans_7x7()


@pytest.fixture(name="frame_7x7")
def make_frame_7x7():
    return fixtures.make_frame_7x7()


@pytest.fixture(name="grid_7x7")
def make_grid_7x7():
    return fixtures.make_grid_7x7()


@pytest.fixture(name="sub_grid_7x7")
def make_sub_grid_7x7():
    return fixtures.make_sub_grid_7x7()


@pytest.fixture(name="grid_iterate_7x7")
def make_grid_iterate_7x7():
    return fixtures.make_grid_iterate_7x7()


@pytest.fixture(name="grid_irregular_7x7")
def make_grid_irregular_7x7():
    return fixtures.make_grid_irregular_7x7()


@pytest.fixture(name="grid_irregular_7x7_list")
def make_grid_irregular_7x7_list():
    return fixtures.make_grid_irregular_7x7_list()


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7():
    return fixtures.make_blurring_grid_7x7()


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return fixtures.make_image_7x7()


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return fixtures.make_noise_map_7x7()


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7():
    return fixtures.make_imaging_7x7()


@pytest.fixture(name="fit_imaging_7x7")
def make_masked_imaging_fit_x1_plane_7x7():
    return fixtures.make_masked_imaging_fit_x1_plane_7x7()


@pytest.fixture(name="visibilities_mask_7")
def make_visibilities_mask_7():
    return fixtures.make_visibilities_mask_7()


@pytest.fixture(name="visibilities_7")
def make_visibilities_7():
    return fixtures.make_visibilities_7()


@pytest.fixture(name="visibilities_noise_map_7")
def make_noise_map_7():
    return fixtures.make_visibilities_noise_map_7()


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7():
    return fixtures.make_uv_wavelengths_7()


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7():
    return fixtures.make_transformer_7x7_7()


@pytest.fixture(name="interferometer_7")
def make_interferometer_7():
    return fixtures.make_interferometer_7()


@pytest.fixture(name="fit_interferometer_7")
def make_masked_interferometer_fit_x1_plane_7():
    return fixtures.make_masked_interferometer_fit_x1_plane_7()


@pytest.fixture(name="rectangular_mapper_7x7_3x3")
def make_rectangular_mapper_7x7_3x3():
    return fixtures.make_rectangular_mapper_7x7_3x3()


@pytest.fixture(name="voronoi_mapper_9_3x3")
def make_voronoi_mapper_9_3x3():
    return fixtures.make_voronoi_mapper_9_3x3()


@pytest.fixture(name="rectangular_inversion_7x7_3x3")
def make_rectangular_inversion_7x7_3x3():
    return fixtures.make_rectangular_inversion_7x7_3x3()


@pytest.fixture(name="voronoi_inversion_9_3x3")
def make_voronoi_inversion_9_3x3():
    return fixtures.make_voronoi_inversion_9_3x3()


@pytest.fixture(name="euclid_data")
def make_euclid_data():
    return fixtures.make_euclid_data()


@pytest.fixture(name="acs_ccd")
def make_acs_ccd():
    return fixtures.make_acs_ccd()


@pytest.fixture(name="acs_quadrant")
def make_acs_quadrant():
    return fixtures.make_acs_quadrant()
