import os

import pytest
from pathlib import Path

import autoarray as aa
from autoconf.dictable import from_dict, output_to_json, from_json


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "type": "instance",
        "class_path": "autoarray.inversion.pixelization.settings.SettingsPixelization",
        "arguments": {
            "use_border": True,
        },
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(from_dict(settings_dict), aa.SettingsPixelization)


def test_file():
    filename = Path("/tmp/temp.json")

    output_to_json(aa.SettingsPixelization(), filename)

    try:
        assert isinstance(from_json(filename), aa.SettingsPixelization)
    finally:
        os.remove(filename)
