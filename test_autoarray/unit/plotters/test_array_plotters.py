from os import path
import autoarray as aa
from autoarray import conf

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


class TestArrayPlotter:
    def test__plotter_inputs_inherit_from_abstract_class(self):

        plotter = aa.plotter.array()

        assert plotter.is_sub_plotter == False
        assert plotter.figsize == (7, 7)
        assert plotter.aspect == "auto"
        assert plotter.cmap == "jet"
        assert plotter.norm == "linear"
        assert plotter.norm_min == None
        assert plotter.norm_max == None
        assert plotter.linthresh == 1.0
        assert plotter.linscale == 2.0
        assert plotter.cb_ticksize == 1
        assert plotter.cb_fraction == 3.0
        assert plotter.cb_pad == 4.0
        assert plotter.cb_tick_values == None
        assert plotter.cb_tick_labels == None
        assert plotter.mask_pointsize == 2
        assert plotter.border_pointsize == 3
        assert plotter.point_pointsize == 4
        assert plotter.grid_pointsize == 5

        assert plotter.ticks.y_manual == None
        assert plotter.ticks.x_manual == None
        #      assert plotter.ticks.ysize == 14
        #      assert plotter.ticks.xsize == 14

        assert plotter.labels.title == None
        assert plotter.labels._yunits == None
        assert plotter.labels._xunits == None
        #      assert plotter.labels.titlesize == 11
        #      assert plotter.labels.ysize == 12
        #      assert plotter.labels.xsize == 13

        assert plotter.output.path == None
        assert plotter.output.format == "show"
        assert plotter.output.filename == None

        plotter = aa.plotter.array(
            figsize=(6, 6),
            aspect="auto",
            cmap="cold",
            norm="log",
            norm_min=0.1,
            norm_max=1.0,
            linthresh=1.5,
            linscale=2.0,
            cb_ticksize=20,
            cb_fraction=0.001,
            cb_pad=10.0,
            mask_pointsize=24,
            border_pointsize=25,
            point_pointsize=26,
            grid_pointsize=27,
            ticks=aa.plotter.Ticks(
                ysize=23, xsize=24, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            ),
            labels=aa.plotter.Labels(
                title="OMG",
                yunits="hi",
                xunits="hi2",
                titlesize=1,
                ysize=2,
                xsize=3,
                use_scaled_units=True,
            ),
            cb_tick_values=[5.0, 6.0],
            cb_tick_labels=[7.0, 8.0],
            output=aa.plotter.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.figsize == (6, 6)
        assert plotter.aspect == "auto"
        assert plotter.cmap == "cold"
        assert plotter.norm == "log"
        assert plotter.norm_min == 0.1
        assert plotter.norm_max == 1.0
        assert plotter.linthresh == 1.5
        assert plotter.linscale == 2.0
        assert plotter.cb_ticksize == 20
        assert plotter.cb_fraction == 0.001
        assert plotter.cb_pad == 10.0
        assert plotter.cb_tick_values == [5.0, 6.0]
        assert plotter.cb_tick_labels == [7.0, 8.0]

        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27

        assert plotter.ticks.ysize == 23
        assert plotter.ticks.xsize == 24
        assert plotter.ticks.y_manual == [1.0, 2.0]
        assert plotter.ticks.x_manual == [3.0, 4.0]

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3

        assert plotter.output.path == "Path"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"
