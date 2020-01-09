from os import path
from autoarray import conf
from autoarray.plotters import plotters

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


class TestTicks:
    def test__tick_settings_setup_correctly_from_config(self):

        ticks = plotters.Ticks(is_sub_plotter=False)

        assert ticks.ysize == 14
        assert ticks.xsize == 15

        ticks = plotters.Ticks(is_sub_plotter=True)

        assert ticks.ysize == 24
        assert ticks.xsize == 25

        ticks = plotters.Ticks(ysize=34, xsize=35, is_sub_plotter=False)

        assert ticks.ysize == 34
        assert ticks.xsize == 35

    def test_y_and_x_manual_setup_correctly(self):

        ticks = plotters.Ticks(y_manual=[1.0, 2.0], x_manual=[3.0, 4.0])

        assert ticks.y_manual == [1.0, 2.0]
        assert ticks.x_manual == [3.0, 4.0]


class TestLabels:
    def test__title_setup_correctly_from_config(self):

        labels = plotters.Labels()

        assert labels.title == None

        labels = plotters.Labels(title="OMG")

        assert labels.title == "OMG"

    def test__yx_units_are_setup_correctly_from_config(self):

        labels = plotters.Labels(use_scaled_units=False)

        assert labels.use_scaled_units == False
        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits == "pixels"
        assert labels.xunits == "pixels"

        labels = plotters.Labels(use_scaled_units=True)

        assert labels.use_scaled_units == True
        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits == "scaled"
        assert labels.xunits == "scaled"

        labels = plotters.Labels(
            title="OMG", yunits="hi", xunits="hi2", use_scaled_units=True
        )

        assert labels.use_scaled_units == True
        assert labels.title == "OMG"
        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits == "hi"
        assert labels.xunits == "hi2"

    def test__title_yx_sizes_are_setup_correctly_from_config(self):

        labels = plotters.Labels(is_sub_plotter=False)

        assert labels.titlesize == 11
        assert labels.ysize == 12
        assert labels.xsize == 13

        labels = plotters.Labels(is_sub_plotter=True)

        assert labels.titlesize == 15
        assert labels.ysize == 22
        assert labels.xsize == 23

        labels = plotters.Labels(is_sub_plotter=False, titlesize=30, ysize=31, xsize=32)

        assert labels.titlesize == 30
        assert labels.ysize == 31
        assert labels.xsize == 32

        labels = plotters.Labels(is_sub_plotter=True, titlesize=33, ysize=34, xsize=35)

        assert labels.titlesize == 33
        assert labels.ysize == 34
        assert labels.xsize == 35

    def test__title_from_func__uses_func_name_if_title_is_none(self):
        def toy_func():
            pass

        labels = plotters.Labels(title=None)

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Toy_func"

        labels = plotters.Labels(title="Hi")

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Hi"

    def test__yx_units_from_func__uses_function_inputs_if_available(self):
        def toy_func():
            pass

        labels = plotters.Labels(yunits=None, xunits=None)

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == None
        assert xunits_from_func == None

        def toy_func(label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = plotters.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = plotters.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"

        def toy_func(argument, label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = plotters.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = plotters.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"


class TestOutput:
    def test__filename_from_func__returns_function_name_if_no_filename(self):
        def toy_func():
            pass

        output = plotters.Output(filename=None)

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "toy_func"

        output = plotters.Output(filename="Hi")

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "Hi"


class TestPlotter:
    def test__plotter_settings_use_figure_config_if_not_manually_input(self):

        plotter = plotters.Plotter()

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

        plotter = plotters.Plotter(
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
        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27

    def test__plotter_settings_use_subplot_config_if_not_manually_input(self):

        plotter = plotters.Plotter(is_sub_plotter=True)

        assert plotter.figsize == None
        assert plotter.aspect == "square"
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

        plotter = plotters.Plotter(
            is_sub_plotter=True,
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
        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27

    def test__ticks_are_setup_correctly(self):

        plotter = plotters.Plotter(is_sub_plotter=False)

        #    assert plotter.ticks.ysize == 14
        #    assert plotter.ticks.xsize == 15
        assert plotter.ticks.y_manual == None
        assert plotter.ticks.x_manual == None

        plotter = plotters.Plotter(
            ticks=plotters.Ticks(
                ysize=24, xsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            )
        )

        assert plotter.ticks.ysize == 24
        assert plotter.ticks.xsize == 25
        assert plotter.ticks.y_manual == [1.0, 2.0]
        assert plotter.ticks.x_manual == [3.0, 4.0]

    def test__labels_are_setup_correctly(self):

        # plotter = plotters.Plotter()
        #
        # assert plotter.labels.title == None
        # assert plotter.labels._yunits == None
        # assert plotter.labels._xunits == None
        # assert plotter.labels.titlesize == 11
        # assert plotter.labels.ysize == 12
        # assert plotter.labels.xsize == 13
        # assert plotter.use_scaled_units == False

        plotter = plotters.Plotter(
            labels=plotters.Labels(
                title="OMG",
                yunits="hi",
                xunits="hi2",
                titlesize=1,
                ysize=2,
                xsize=3,
                use_scaled_units=True,
            )
        )

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3
        assert plotter.use_scaled_units == True

    def test__plotter_outputs_are_setup_correctly(self):

        plotter = plotters.Plotter()

        assert plotter.output.path == None
        assert plotter.output.format == "show"
        assert plotter.output.filename == None

        plotter = plotters.Plotter(
            output=plotters.Output(path="Path", format="png", filename="file")
        )

        assert plotter.output.path == "Path"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

    def test__plotter_number_of_subplots(self):

        plotter = plotters.Plotter()

        rows, columns, figsize = plotter.get_subplot_rows_columns_figsize(
            number_subplots=1
        )

        assert rows == 1
        assert columns == 2
        assert figsize == (18, 8)

        rows, columns, figsize = plotter.get_subplot_rows_columns_figsize(
            number_subplots=4
        )

        assert rows == 2
        assert columns == 2
        assert figsize == (13, 10)
