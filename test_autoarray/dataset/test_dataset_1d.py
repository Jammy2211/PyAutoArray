import pytest

from autoarray import Array1D
from autocti import Dataset1D


@pytest.fixture(name="pixel_line_dict")
def make_pixel_line_dict():
    return {
        "location": [
            2,
            4,
        ],
        "date": 2453963.778275463,
        "background": 31.30540652532858,
        "flux": 1234.,
        "data": [
            5.0,
            3.0,
            2.0,
            1.0,
        ],
        "noise": [
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    }


@pytest.fixture(name="size")
def make_size():
    return 10


@pytest.fixture(name="dataset_1d")
def make_dataset_1d(pixel_line_dict, size):
    return Dataset1D.from_pixel_line_dict(
        pixel_line_dict,
        size=size,
    )


def test_parse_data(dataset_1d):
    assert (dataset_1d.data == Array1D.manual_native(
        [0., 0., 5., 3., 2., 1., 0., 0., 0., 0.],
        pixel_scales=0.1
    )).all()


def test_parse_noise(dataset_1d):
    assert (dataset_1d.noise_map == Array1D.manual_native(
        [0., 0., 1., 1., 1., 1., 0., 0., 0., 0.],
        pixel_scales=0.1
    )).all()


def test_pre_cti(dataset_1d):
    assert (dataset_1d.pre_cti_data == Array1D.manual_native(
        [0., 0., 1234., 0., 0., 0., 0., 0., 0., 0.],
        pixel_scales=0.1
    )).all()
