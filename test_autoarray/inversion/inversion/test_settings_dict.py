import os

import pytest
from pathlib import Path

import autoarray as aa


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "class_path": "autoarray.inversion.inversion.settings.SettingsInversion",
        "type": "instance",
        "arguments": {
            "use_w_tilde": True,
            "use_positive_only_solver": False,
            "positive_only_uses_p_initial": False,
            "force_edge_pixels_to_zeros": True,
            "force_edge_image_pixels_to_zeros": False,
            "image_pixels_source_zero": None,
            "no_regularization_add_to_curvature_diag_value": 1e-08,
            "use_w_tilde_numpy": False,
            "use_source_loop": False,
            "use_linear_operators": False,
            "tolerance": 1e-08,
            "maxiter": 250,
        },
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(
        aa.SettingsInversion.from_dict(settings_dict), aa.SettingsInversion
    )


def test_file():
    filename = Path("/tmp/temp.json")

    aa.SettingsInversion().output_to_json(filename)

    try:
        assert isinstance(
            aa.SettingsInversion.from_json(filename), aa.SettingsInversion
        )
    finally:
        os.remove(filename)
