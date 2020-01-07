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
        assert plotter.mask_pointsize == 2
        assert plotter.border_pointsize == 3
        assert plotter.point_pointsize == 4
        assert plotter.grid_pointsize == 5
        assert plotter.titlesize == 11
        assert plotter.xlabelsize == 12
        assert plotter.ylabelsize == 13
        assert plotter.xyticksize == 14

        assert plotter.label_title == None
        assert plotter.label_yunits == None
        assert plotter.label_xunits == None
        assert plotter.label_yticks == None
        assert plotter.label_xticks == None
        assert plotter.cb_tick_values == None
        assert plotter.cb_tick_labels == None

        assert plotter.output_path == None
        assert plotter.output_format == "show"
        assert plotter.output_filename == None

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
            titlesize=20,
            xlabelsize=21,
            ylabelsize=22,
            xyticksize=23,
            mask_pointsize=24,
            border_pointsize=25,
            point_pointsize=26,
            grid_pointsize=27,
            label_title="OMG",
            label_yunits="hi",
            label_xunits="hi2",
            label_yticks=[1.0, 2.0],
            label_xticks=[3.0, 4.0],
            cb_tick_values=[5.0, 6.0],
            cb_tick_labels=[7.0, 8.0],
            output_path="Path",
            output_format="png",
            output_filename="file",
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
        assert plotter.titlesize == 20
        assert plotter.xlabelsize == 21
        assert plotter.ylabelsize == 22
        assert plotter.xyticksize == 23
        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27

        assert plotter.label_title == "OMG"
        assert plotter.label_yunits == "hi"
        assert plotter.label_xunits == "hi2"
        assert plotter.label_yticks == [1.0, 2.0]
        assert plotter.label_xticks == [3.0, 4.0]
        assert plotter.cb_tick_values == [5.0, 6.0]
        assert plotter.cb_tick_labels == [7.0, 8.0]

        assert plotter.output_path == "Path"
        assert plotter.output_format == "png"
        assert plotter.output_filename == "file"

    def test__new_plotter_with_labels_and_filename(self):

        plotter = aa.plotter.array(
            is_sub_plotter=False,
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
            titlesize=20,
            xlabelsize=21,
            ylabelsize=22,
            xyticksize=23,
            mask_pointsize=24,
            border_pointsize=25,
            point_pointsize=26,
            grid_pointsize=27,
            label_title="OMG",
            label_yunits="hi",
            label_xunits="hi2",
            label_yticks=[1.0, 2.0],
            label_xticks=[3.0, 4.0],
            cb_tick_values=[5.0, 6.0],
            cb_tick_labels=[7.0, 8.0],
            output_path="Path",
            output_format="png",
            output_filename="file",
        )

        plotter = plotter.plotter_with_new_labels_and_filename(
            label_title="a", label_yunits="b", label_xunits="c", output_filename="d"
        )

        assert plotter.label_title == "a"
        assert plotter.label_yunits == "b"
        assert plotter.label_xunits == "c"
        assert plotter.output_filename == "d"

        assert plotter.is_sub_plotter == False
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
        assert plotter.titlesize == 20
        assert plotter.xlabelsize == 21
        assert plotter.ylabelsize == 22
        assert plotter.xyticksize == 23
        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27

        assert plotter.label_yticks == [1.0, 2.0]
        assert plotter.label_xticks == [3.0, 4.0]
        assert plotter.cb_tick_values == [5.0, 6.0]
        assert plotter.cb_tick_labels == [7.0, 8.0]

        assert plotter.output_path == "Path"
        assert plotter.output_format == "png"
