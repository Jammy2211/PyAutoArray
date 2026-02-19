import os

import pytest
from pathlib import Path

import autoarray as aa
from autoconf.dictable import from_dict, output_to_json, from_json


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "class_path": "autoarray.settings.Settings",
        "type": "instance",
        "arguments": {
            "use_positive_only_solver": False,
            "positive_only_uses_p_initial": False,
            "no_regularization_add_to_curvature_diag_value": 1e-08,
            "tolerance": 1e-08,
            "maxiter": 250,
        },
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(from_dict(settings_dict), aa.Settings)


def test_file():
    filename = Path("/tmp/temp.json")

    output_to_json(aa.Settings(), filename)

    try:
        assert isinstance(from_json(filename), aa.Settings)
    finally:
        os.remove(filename)
