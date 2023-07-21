import os

import pytest
from pathlib import Path

import autoarray as aa


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "type": "autoarray.inversion.pixelization.settings.SettingsPixelization",
        "use_border": True,
        "is_stochastic": False,
        "kmeans_seed": 0,
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(
        aa.SettingsPixelization.from_dict(settings_dict), aa.SettingsPixelization
    )


def test_file():
    filename = Path("/tmp/temp.json")

    aa.SettingsPixelization().output_to_json(filename)

    try:
        assert isinstance(
            aa.SettingsPixelization().from_json(filename), aa.SettingsPixelization
        )
    finally:
        os.remove(filename)
