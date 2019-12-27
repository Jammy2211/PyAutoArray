from os import path
import autoarray as aa
from autoarray import conf

import pytest

directory = path.dirname(path.realpath(__file__))

@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/settings"), path.join(directory, "output")
    )

class TestSettings:

    def test__individual_attributes_are_output(self):

        settings = aa.plot.settings()

        assert settings.include_origin == True
        assert settings.include_mask == True
        assert settings.include_border == False
        assert settings.include_positions == True

        assert settings.figsize == (7, 7)
        assert settings.aspect == "square"
        assert settings.cmap == "jet"
        assert settings.norm == "linear"
        assert settings.norm_min == None
        assert settings.norm_max == None
        assert settings.linthresh == 0.05
        assert settings.linscale == 0.1
        assert settings.cb_ticksize == 10
        assert settings.cb_fraction == 0.047
        assert settings.cb_pad == 0.01
        assert settings.cb_tick_values == None
        assert settings.cb_tick_labels == None
        assert settings.title == None
        assert settings.titlesize == 16
        assert settings.xlabelsize == 16
        assert settings.ylabelsize == 16
        assert settings.xyticksize == 16
        assert settings.mask_pointsize == 10
        assert settings.position_pointsize == 30
        assert settings.grid_pointsize == 1
        assert settings.output_path == None
        assert settings.output_format == "show"
        assert settings.output_filename == None

        settings = aa.plot.settings(include_origin=False, include_mask=False, include_border=True, include_positions=False,
                                    figsize=(6,6), aspect="auto", cmap="cold", norm="log", norm_min=0.1, norm_max=1.0,
                                    linthresh=1.5, linscale=2.0, cb_ticksize=20, cb_fraction=0.001, cb_pad=10.0, cb_tick_values=[1.0, 2.0],
                                    cb_tick_labels=[3.0, 4.0], title="OMG", titlesize=20, xlabelsize=21, ylabelsize=22,
                                    xyticksize=23, mask_pointsize=24, position_pointsize=25, grid_pointsize=26,
                                    output_path="Path", output_format="png", output_filename="file")

        assert settings.include_origin == False
        assert settings.include_mask == False
        assert settings.include_border == True
        assert settings.include_positions == False

        assert settings.figsize == (6, 6)
        assert settings.aspect == "auto"
        assert settings.cmap == "cold"
        assert settings.norm == "log"
        assert settings.norm_min == 0.1
        assert settings.norm_max == 1.0
        assert settings.linthresh == 1.5
        assert settings.linscale == 2.0
        assert settings.cb_ticksize == 20
        assert settings.cb_fraction == 0.001
        assert settings.cb_pad == 10.0
        assert settings.cb_tick_values == [1.0, 2.0]
        assert settings.cb_tick_labels == [3.0, 4.0]
        assert settings.title == "OMG"
        assert settings.titlesize == 20
        assert settings.xlabelsize == 21
        assert settings.ylabelsize == 22
        assert settings.xyticksize == 23
        assert settings.mask_pointsize == 24
        assert settings.position_pointsize == 25
        assert settings.grid_pointsize == 26
        assert settings.output_path == "Path"
        assert settings.output_format == "png"
        assert settings.output_filename == "file"