import os

import pytest
from pathlib import Path

import autoarray as aa


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "type": "autoarray.dataset.imaging.settings.SettingsImaging",
        "grid_class": {
            "type": "type",
            "class_path": "autoarray.structures.grids.uniform_2d.Grid2D",
        },
        "grid_pixelization_class": None,
        "sub_size": 1,
        "sub_size_pixelization": 4,
        "fractional_accuracy": 0.9999,
        "relative_accuracy": None,
        "sub_steps": None,
        "use_normalized_psf": True,
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(aa.SettingsImaging.from_dict(settings_dict), aa.SettingsImaging)


def test_file():
    filename = Path("/tmp/temp.json")

    aa.SettingsImaging(grid_class=aa.Grid2D).output_to_json(filename)

    try:
        assert isinstance(aa.SettingsImaging.from_json(filename), aa.SettingsImaging)
    finally:
        os.remove(filename)
