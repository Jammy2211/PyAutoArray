from os import path
from autoarray import conf
from autoarray.plotters import plotters, mat_objs
import matplotlib.pyplot as plt

import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )

class TestAbstractPlotter:

    def test__plotter_settings_plot_in_kpc__use_config_if_available(self):

        plotter = plotters.Plotter()

        assert plotter.plot_in_kpc == False

        plotter = plotters.Plotter(plot_in_kpc=True)

        assert plotter.plot_in_kpc == True

    def test__ticks_are_setup_correctly(self):

        plotter = plotters.Plotter()

        assert plotter.ticks.ysize == 14
        assert plotter.ticks.xsize == 15
        assert plotter.ticks.y_manual == None
        assert plotter.ticks.x_manual == None

        plotter = plotters.Plotter(
            ticks=mat_objs.Ticks(
                ysize=24, xsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            )
        )

        assert plotter.ticks.ysize == 24
        assert plotter.ticks.xsize == 25
        assert plotter.ticks.y_manual == [1.0, 2.0]
        assert plotter.ticks.x_manual == [3.0, 4.0]

    def test__labels_are_setup_correctly(self):

        plotter = plotters.Plotter()

        assert plotter.labels.title == None
        assert plotter.labels._yunits == None
        assert plotter.labels._xunits == None
        assert plotter.labels.titlesize == 11
        assert plotter.labels.ysize == 12
        assert plotter.labels.xsize == 13
        assert plotter.use_scaled_units == True

        plotter = plotters.Plotter(
            use_scaled_units=False,
            labels=mat_objs.Labels(
                title="OMG",
                yunits="hi",
                xunits="hi2",
                titlesize=1,
                ysize=2,
                xsize=3,
            )
        )

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3
        assert plotter.use_scaled_units == False

    def test__plotter_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = plotters.Plotter(
            use_scaled_units=False,
            labels=mat_objs.Labels(
                title="OMG",
                yunits="hi",
                xunits="hi2",
                titlesize=1,
                ysize=2,
                xsize=3,
            )
        )

        plotter = plotter.plotter_with_new_labels()

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3
        assert plotter.use_scaled_units == False

        plotter = plotter.plotter_with_new_labels(
            labels=mat_objs.Labels(
                title="OMG0",
                yunits="hi0",
                xunits="hi20",
                titlesize=10,
                ysize=20,
                xsize=30,
        ))

        assert plotter.labels.title == "OMG0"
        assert plotter.labels._yunits == "hi0"
        assert plotter.labels._xunits == "hi20"
        assert plotter.labels.titlesize == 10
        assert plotter.labels.ysize == 20
        assert plotter.labels.xsize == 30
        assert plotter.use_scaled_units == False

    def test__plotter_outputs_are_setup_correctly(self):

        plotter = plotters.Plotter()

        assert plotter.output.path == None
        assert plotter.output._format == None
        assert plotter.output.format == "show"
        assert plotter.output.filename == None

        plotter = plotters.Plotter(
            output=mat_objs.Output(path="Path", format="png", filename="file")
        )

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

    def test__plotter_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = plotters.Plotter(
            output=mat_objs.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.plotter_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        plotter = plotter.plotter_with_new_output(output=mat_objs.Output(path="Path0", filename="file0"))

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        plotter = plotter.plotter_with_new_output(output=mat_objs.Output(path="Path1", filename="file1", format="fits"))

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

    def test__plotter_with_new_unit_conversion_factor__new_outputs_are_setup_correctly_if_input(self):

        plotter = plotters.Plotter(
            unit_conversion_factor=1.0
        )

        assert plotter.unit_conversion_factor == 1.0

        plotter = plotter.plotter_with_new_unit_conversion_factor()

        assert plotter.unit_conversion_factor == 1.0

        plotter = plotter.plotter_with_new_unit_conversion_factor(unit_conversion_factor=2.0)

        assert plotter.unit_conversion_factor == 2.0

    def test__plotter_array_property_inherits_plotter_attributes(self):

        plotter = plotters.Plotter()

        assert plotter.array.figsize == (7, 7)
        assert plotter.array.aspect == "auto"
        assert plotter.array.cmap == "jet"
        assert plotter.array.norm == "linear"
        assert plotter.array.norm_min == None
        assert plotter.array.norm_max == None
        assert plotter.array.linthresh == 1.0
        assert plotter.array.linscale == 2.0
        assert plotter.array.cb_ticksize == 1
        assert plotter.array.cb_fraction == 3.0
        assert plotter.array.cb_pad == 4.0
        assert plotter.array.cb_tick_values == None
        assert plotter.array.cb_tick_labels == None
        assert plotter.array.mask_pointsize == 2
        assert plotter.array.border_pointsize == 3
        assert plotter.array.point_pointsize == 4
        assert plotter.array.grid_pointsize == 5

        assert plotter.array.ticks.y_manual == None
        assert plotter.array.ticks.x_manual == None
        #      assert plotter.array.ticks.ysize == 14
        #      assert plotter.array.ticks.xsize == 14

        assert plotter.array.labels.title == None
        assert plotter.array.labels._yunits == None
        assert plotter.array.labels._xunits == None
        #      assert plotter.array.labels.titlesize == 11
        #      assert plotter.array.labels.ysize == 12
        #      assert plotter.array.labels.xsize == 13

        assert plotter.array.output.path == None
        assert plotter.array.output.format == "show"
        assert plotter.array.output.filename == None

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
            ticks=mat_objs.Ticks(
                ysize=23, xsize=24, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            ),
            labels=mat_objs.Labels(
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.array.figsize == (6, 6)
        assert plotter.array.aspect == "auto"
        assert plotter.array.cmap == "cold"
        assert plotter.array.norm == "log"
        assert plotter.array.norm_min == 0.1
        assert plotter.array.norm_max == 1.0
        assert plotter.array.linthresh == 1.5
        assert plotter.array.linscale == 2.0
        assert plotter.array.cb_ticksize == 20
        assert plotter.array.cb_fraction == 0.001
        assert plotter.array.cb_pad == 10.0
        assert plotter.array.cb_tick_values == [5.0, 6.0]
        assert plotter.array.cb_tick_labels == [7.0, 8.0]

        assert plotter.array.mask_pointsize == 24
        assert plotter.array.border_pointsize == 25
        assert plotter.array.point_pointsize == 26
        assert plotter.array.grid_pointsize == 27

        assert plotter.array.ticks.ysize == 23
        assert plotter.array.ticks.xsize == 24
        assert plotter.array.ticks.y_manual == [1.0, 2.0]
        assert plotter.array.ticks.x_manual == [3.0, 4.0]

        assert plotter.array.labels.title == "OMG"
        assert plotter.array.labels._yunits == "hi"
        assert plotter.array.labels._xunits == "hi2"
        assert plotter.array.labels.titlesize == 1
        assert plotter.array.labels.ysize == 2
        assert plotter.array.labels.xsize == 3

        assert plotter.array.output.path == "Path"
        assert plotter.array.output.format == "png"
        assert plotter.array.output.filename == "file"

    def test__plotter_grid_property_inherits_plotter_attributes(self):

        plotter = plotters.Plotter()

        assert plotter.grid.figsize == (7, 7)
        assert plotter.grid.aspect == "auto"
        assert plotter.grid.cmap == "jet"
        assert plotter.grid.norm == "linear"
        assert plotter.grid.norm_min == None
        assert plotter.grid.norm_max == None
        assert plotter.grid.linthresh == 1.0
        assert plotter.grid.linscale == 2.0
        assert plotter.grid.cb_ticksize == 1
        assert plotter.grid.cb_fraction == 3.0
        assert plotter.grid.cb_pad == 4.0
        assert plotter.grid.cb_tick_values == None
        assert plotter.grid.cb_tick_labels == None
        assert plotter.grid.grid_pointsize == 5

        assert plotter.grid.ticks.y_manual == None
        assert plotter.grid.ticks.x_manual == None
        assert plotter.grid.ticks.ysize == 14
        assert plotter.grid.ticks.xsize == 15

        assert plotter.grid.labels.title == None
        assert plotter.grid.labels._yunits == None
        assert plotter.grid.labels._xunits == None
        #      assert plotter.grid.labels.titlesize == 11
        #      assert plotter.grid.labels.ysize == 12
        #      assert plotter.grid.labels.xsize == 13

        assert plotter.grid.output.path == None
        assert plotter.grid.output.format == "show"
        assert plotter.grid.output.filename == None

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
            grid_pointsize=27,
            ticks=mat_objs.Ticks(
                ysize=23, xsize=24, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            ),
            labels=mat_objs.Labels(
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.grid.figsize == (6, 6)
        assert plotter.grid.aspect == "auto"
        assert plotter.grid.cmap == "cold"
        assert plotter.grid.norm == "log"
        assert plotter.grid.norm_min == 0.1
        assert plotter.grid.norm_max == 1.0
        assert plotter.grid.linthresh == 1.5
        assert plotter.grid.linscale == 2.0
        assert plotter.grid.cb_ticksize == 20
        assert plotter.grid.cb_fraction == 0.001
        assert plotter.grid.cb_pad == 10.0
        assert plotter.grid.cb_tick_values == [5.0, 6.0]
        assert plotter.grid.cb_tick_labels == [7.0, 8.0]

        assert plotter.grid.grid_pointsize == 27
        assert plotter.grid.grid_pointcolor == "k"

        assert plotter.grid.ticks.ysize == 23
        assert plotter.grid.ticks.xsize == 24
        assert plotter.grid.ticks.y_manual == [1.0, 2.0]
        assert plotter.grid.ticks.x_manual == [3.0, 4.0]

        assert plotter.grid.labels.title == "OMG"
        assert plotter.grid.labels._yunits == "hi"
        assert plotter.grid.labels._xunits == "hi2"
        assert plotter.grid.labels.titlesize == 1
        assert plotter.grid.labels.ysize == 2
        assert plotter.grid.labels.xsize == 3

        assert plotter.grid.output.path == "Path"
        assert plotter.grid.output.format == "png"
        assert plotter.grid.output.filename == "file"

    def test__plotter_mapper_property_inherits_plotter_attributes(self):

        plotter = plotters.Plotter()

        assert plotter.mapper.figsize == (7, 7)
        assert plotter.mapper.aspect == "auto"
        assert plotter.mapper.cmap == "jet"
        assert plotter.mapper.norm == "linear"
        assert plotter.mapper.norm_min == None
        assert plotter.mapper.norm_max == None
        assert plotter.mapper.linthresh == 1.0
        assert plotter.mapper.linscale == 2.0
        assert plotter.mapper.cb_ticksize == 1
        assert plotter.mapper.cb_fraction == 3.0
        assert plotter.mapper.cb_pad == 4.0
        assert plotter.mapper.cb_tick_values == None
        assert plotter.mapper.cb_tick_labels == None
        assert plotter.mapper.grid_pointsize == 5
        assert plotter.mapper.grid_pointcolor == "k"

        assert plotter.mapper.ticks.y_manual == None
        assert plotter.mapper.ticks.x_manual == None
        assert plotter.mapper.ticks.ysize == 14
        assert plotter.mapper.ticks.xsize == 15

        assert plotter.mapper.labels.title == None
        assert plotter.mapper.labels._yunits == None
        assert plotter.mapper.labels._xunits == None
        #      assert plotter.mapper.labels.titlesize == 11
        #      assert plotter.mapper.labels.ysize == 12
        #      assert plotter.mapper.labels.xsize == 13

        assert plotter.mapper.output.path == None
        assert plotter.mapper.output.format == "show"
        assert plotter.mapper.output.filename == None

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
            grid_pointsize=27,
            ticks=mat_objs.Ticks(
                ysize=23, xsize=24, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            ),
            labels=mat_objs.Labels(
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.mapper.figsize == (6, 6)
        assert plotter.mapper.aspect == "auto"
        assert plotter.mapper.cmap == "cold"
        assert plotter.mapper.norm == "log"
        assert plotter.mapper.norm_min == 0.1
        assert plotter.mapper.norm_max == 1.0
        assert plotter.mapper.linthresh == 1.5
        assert plotter.mapper.linscale == 2.0
        assert plotter.mapper.cb_ticksize == 20
        assert plotter.mapper.cb_fraction == 0.001
        assert plotter.mapper.cb_pad == 10.0
        assert plotter.mapper.cb_tick_values == [5.0, 6.0]
        assert plotter.mapper.cb_tick_labels == [7.0, 8.0]

        assert plotter.mapper.grid_pointsize == 27
        assert plotter.mapper.grid_pointcolor == "k"

        assert plotter.mapper.ticks.ysize == 23
        assert plotter.mapper.ticks.xsize == 24
        assert plotter.mapper.ticks.y_manual == [1.0, 2.0]
        assert plotter.mapper.ticks.x_manual == [3.0, 4.0]

        assert plotter.mapper.labels.title == "OMG"
        assert plotter.mapper.labels._yunits == "hi"
        assert plotter.mapper.labels._xunits == "hi2"
        assert plotter.mapper.labels.titlesize == 1
        assert plotter.mapper.labels.ysize == 2
        assert plotter.mapper.labels.xsize == 3

        assert plotter.mapper.output.path == "Path"
        assert plotter.mapper.output.format == "png"
        assert plotter.mapper.output.filename == "file"

    def test__plotter_line_property_inherits_plotter_attributes(self):

        plotter = plotters.Plotter()

        assert plotter.line.figsize == (7, 7)
        assert plotter.line.aspect == "auto"
        assert plotter.line.line_pointsize == None

        assert plotter.line.ticks.y_manual == None
        assert plotter.line.ticks.x_manual == None
        assert plotter.line.ticks.ysize == 14
        assert plotter.line.ticks.xsize == 15

        assert plotter.line.labels.title == None
        assert plotter.line.labels._yunits == None
        assert plotter.line.labels._xunits == None
        #      assert plotter.line.labels.titlesize == 11
        #      assert plotter.line.labels.ysize == 12
        #      assert plotter.line.labels.xsize == 13

        assert plotter.line.output.path == None
        assert plotter.line.output.format == "show"
        assert plotter.line.output.filename == None

        plotter = plotters.Plotter(
            figsize=(6, 6),
            aspect="auto",
            line_pointsize=27,
            ticks=mat_objs.Ticks(
                ysize=23, xsize=24, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            ),
            labels=mat_objs.Labels(
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
            include_legend=True, legend_fontsize=30,
        )

        assert plotter.line.figsize == (6, 6)

        assert plotter.line.line_pointsize == 27

        assert plotter.line.ticks.ysize == 23
        assert plotter.line.ticks.xsize == 24
        assert plotter.line.ticks.y_manual == [1.0, 2.0]
        assert plotter.line.ticks.x_manual == [3.0, 4.0]

        assert plotter.line.include_legend == True
        assert plotter.line.legend_fontsize == 30

        assert plotter.line.labels.title == "OMG"
        assert plotter.line.labels._yunits == "hi"
        assert plotter.line.labels._xunits == "hi2"
        assert plotter.line.labels.titlesize == 1
        assert plotter.line.labels.ysize == 2
        assert plotter.line.labels.xsize == 3

        assert plotter.line.output.path == "Path"
        assert plotter.line.output.format == "png"
        assert plotter.line.output.filename == "file"

    def test__open_and_close_figures(self):

        plotter = plotters.Plotter()

        assert plt.fignum_exists(num=1) == False

        plotter.setup_figure()

        assert plt.fignum_exists(num=1) == True

        plotter.close_figure()

        assert plt.fignum_exists(num=1) == False

        plotter = plotters.SubPlotter()

        assert plt.fignum_exists(num=1) == False

        plotter.setup_subplot_figure(number_subplots=4)

        assert plt.fignum_exists(num=1) == True

        plotter.close_figure()

        assert plt.fignum_exists(num=1) == False


class TestPlotter:
    def test__plotter_settings_use_figure_config_if_not_manually_input(self):

        plotter = plotters.Plotter()

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


class TestSubPlotter:

    def test__plotter_settings_use_subplot_config_if_not_manually_input(self):

        plotter = plotters.SubPlotter()

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

        plotter = plotters.SubPlotter(
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

    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = plotters.SubPlotter()

        figsize = plotter.get_subplot_figsize(
            number_subplots=1
        )

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(
            number_subplots=4
        )

        assert figsize == (13, 10)

        plotter = plotters.SubPlotter(figsize=(20, 20))

        figsize = plotter.get_subplot_figsize(
            number_subplots=4
        )

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = plotters.SubPlotter()

        rows, columns = plotter.get_subplot_rows_columns(
            number_subplots=1
        )

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.get_subplot_rows_columns(
            number_subplots=4
        )

        assert rows == 2
        assert columns == 2


class TestDecorator:
    def test__kpc_per_arcsec_extacted_from_object_if_available(self):

        dictionary = {"hi": 1}

        kpc_per_arcsec = plotters.kpc_per_arcsec_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_arcsec == None

        class MockObj(object):
            def __init__(self, param1):

                self.param1 = param1

        obj = MockObj(param1=1)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_arcsec = plotters.kpc_per_arcsec_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_arcsec == None

        class MockObj(object):
            def __init__(self, param1, kpc_per_arcsec):

                self.param1 = param1
                self.kpc_per_arcsec = kpc_per_arcsec

        obj = MockObj(param1=1, kpc_per_arcsec=2)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_arcsec = plotters.kpc_per_arcsec_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_arcsec == 2
