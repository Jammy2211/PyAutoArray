from os import path
from autoarray import conf
from autoarray.plotters import plotters, mat_objs
import matplotlib.pyplot as plt
import os
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plotter_path")
def make_plotter_setup():
    return "{}/..//test_files/plotters/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


class TestAbstractPlotterAttributes:
    def test__plot_in_kpc__from_config_or_via_manual_input(self):

        plotter = plotters.Plotter()

        assert plotter.plot_in_kpc == False

        plotter = plotters.Plotter(plot_in_kpc=True)

        assert plotter.plot_in_kpc == True

    def test__colormap__from_config_or_via_manual_input(self):
        plotter = plotters.Plotter()

        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.norm_min == None
        assert plotter.cmap.norm_max == None
        assert plotter.cmap.linthresh == 1.0
        assert plotter.cmap.linscale == 2.0

        plotter = plotters.Plotter(
            cmap=mat_objs.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            )
        )

        assert plotter.cmap.cmap == "cold"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.norm_min == 0.1
        assert plotter.cmap.norm_max == 1.0
        assert plotter.cmap.linthresh == 1.5
        assert plotter.cmap.linscale == 2.0

    def test__colorbar__from_config_or_via_manual_input(self):

        plotter = plotters.Plotter()

        assert plotter.cb.ticksize == 1
        assert plotter.cb.fraction == 3.0
        assert plotter.cb.pad == 4.0
        assert plotter.cb.tick_values == None
        assert plotter.cb.tick_labels == None

        plotter = plotters.Plotter(
            cb=mat_objs.ColorBar(
                ticksize=20,
                fraction=0.001,
                pad=10.0,
                tick_values=(1.0, 2.0),
                tick_labels=(3.0, 4.0),
            )
        )

        assert plotter.cb.ticksize == 20
        assert plotter.cb.fraction == 0.001
        assert plotter.cb.pad == 10.0
        assert plotter.cb.tick_values == (1.0, 2.0)
        assert plotter.cb.tick_labels == (3.0, 4.0)

    def test__ticks__from_config_or_via_manual_input(self):

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

    def test__labels__from_config_or_via_manual_input(self):

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
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            ),
        )

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3
        assert plotter.use_scaled_units == False

    def test__output__correctly(self):

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


class TestAbstractPlotter:
    def test__plotter_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = plotters.Plotter(
            use_scaled_units=False,
            labels=mat_objs.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            ),
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
            )
        )

        assert plotter.labels.title == "OMG0"
        assert plotter.labels._yunits == "hi0"
        assert plotter.labels._xunits == "hi20"
        assert plotter.labels.titlesize == 10
        assert plotter.labels.ysize == 20
        assert plotter.labels.xsize == 30
        assert plotter.use_scaled_units == False

    def test__plotter_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = plotters.Plotter(
            output=mat_objs.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.plotter_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        plotter = plotter.plotter_with_new_output(
            output=mat_objs.Output(path="Path0", filename="file0")
        )

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        plotter = plotter.plotter_with_new_output(
            output=mat_objs.Output(path="Path1", filename="file1", format="fits")
        )

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

    def test__plotter_with_new_unit_conversion_factor__new_outputs_are_setup_correctly_if_input(
        self
    ):

        plotter = plotters.Plotter(unit_conversion_factor=1.0)

        assert plotter.unit_conversion_factor == 1.0

        plotter = plotter.plotter_with_new_unit_conversion_factor()

        assert plotter.unit_conversion_factor == 1.0

        plotter = plotter.plotter_with_new_unit_conversion_factor(
            unit_conversion_factor=2.0
        )

        assert plotter.unit_conversion_factor == 2.0

    def test__plotter_array_property_inherits_plotter_attributes(self):

        plotter = plotters.Plotter()

        assert plotter.array.figsize == (7, 7)
        assert plotter.array.aspect == "auto"

        assert plotter.array.cmap.cmap == "jet"
        assert plotter.array.cmap.norm == "linear"
        assert plotter.array.cmap.norm_min == None
        assert plotter.array.cmap.norm_max == None
        assert plotter.array.cmap.linthresh == 1.0
        assert plotter.array.cmap.linscale == 2.0

        assert plotter.array.cb.ticksize == 1
        assert plotter.array.cb.fraction == 3.0
        assert plotter.array.cb.pad == 4.0
        assert plotter.array.cb.tick_values == None
        assert plotter.array.cb.tick_labels == None

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
            cmap=mat_objs.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            ),
            cb=mat_objs.ColorBar(
                ticksize=20,
                fraction=0.001,
                pad=10.0,
                tick_values=[5.0, 6.0],
                tick_labels=[7.0, 8.0],
            ),
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.array.figsize == (6, 6)
        assert plotter.array.aspect == "auto"

        assert plotter.array.cmap.cmap == "cold"
        assert plotter.array.cmap.norm == "log"
        assert plotter.array.cmap.norm_min == 0.1
        assert plotter.array.cmap.norm_max == 1.0
        assert plotter.array.cmap.linthresh == 1.5
        assert plotter.array.cmap.linscale == 2.0

        assert plotter.array.cb.ticksize == 20
        assert plotter.array.cb.fraction == 0.001
        assert plotter.array.cb.pad == 10.0
        assert plotter.array.cb.tick_values == [5.0, 6.0]
        assert plotter.array.cb.tick_labels == [7.0, 8.0]

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
        
        assert plotter.grid.cmap.cmap == "jet"
        assert plotter.grid.cmap.norm == "linear"
        assert plotter.grid.cmap.norm_min == None
        assert plotter.grid.cmap.norm_max == None
        assert plotter.grid.cmap.linthresh == 1.0
        assert plotter.grid.cmap.linscale == 2.0
        
        assert plotter.grid.cb.ticksize == 1
        assert plotter.grid.cb.fraction == 3.0
        assert plotter.grid.cb.pad == 4.0
        assert plotter.grid.cb.tick_values == None
        assert plotter.grid.cb.tick_labels == None
        
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
            cmap=mat_objs.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            ),
            cb=mat_objs.ColorBar(
                ticksize=20,
                fraction=0.001,
                pad=10.0,
                tick_values=[5.0, 6.0],
                tick_labels=[7.0, 8.0],
            ),
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.grid.figsize == (6, 6)
        assert plotter.grid.aspect == "auto"
        
        assert plotter.grid.cmap.cmap == "cold"
        assert plotter.grid.cmap.norm == "log"
        assert plotter.grid.cmap.norm_min == 0.1
        assert plotter.grid.cmap.norm_max == 1.0
        assert plotter.grid.cmap.linthresh == 1.5
        assert plotter.grid.cmap.linscale == 2.0
        
        assert plotter.grid.cb.ticksize == 20
        assert plotter.grid.cb.fraction == 0.001
        assert plotter.grid.cb.pad == 10.0
        assert plotter.grid.cb.tick_values == [5.0, 6.0]
        assert plotter.grid.cb.tick_labels == [7.0, 8.0]

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

        assert plotter.mapper.cmap.cmap == "jet"
        assert plotter.mapper.cmap.norm == "linear"
        assert plotter.mapper.cmap.norm_min == None
        assert plotter.mapper.cmap.norm_max == None
        assert plotter.mapper.cmap.linthresh == 1.0
        assert plotter.mapper.cmap.linscale == 2.0
        
        assert plotter.mapper.cb.ticksize == 1
        assert plotter.mapper.cb.fraction == 3.0
        assert plotter.mapper.cb.pad == 4.0
        assert plotter.mapper.cb.tick_values == None
        assert plotter.mapper.cb.tick_labels == None
        
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
            cmap=mat_objs.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            ),
            cb=mat_objs.ColorBar(
                ticksize=20,
                fraction=0.001,
                pad=10.0,
                tick_values=[5.0, 6.0],
                tick_labels=[7.0, 8.0],
            ),
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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
        )

        assert plotter.mapper.figsize == (6, 6)
        assert plotter.mapper.aspect == "auto"
        
        assert plotter.mapper.cmap.cmap == "cold"
        assert plotter.mapper.cmap.norm == "log"
        assert plotter.mapper.cmap.norm_min == 0.1
        assert plotter.mapper.cmap.norm_max == 1.0
        assert plotter.mapper.cmap.linthresh == 1.5
        assert plotter.mapper.cmap.linscale == 2.0
        
        assert plotter.mapper.cb.ticksize == 20
        assert plotter.mapper.cb.fraction == 0.001
        assert plotter.mapper.cb.pad == 10.0
        assert plotter.mapper.cb.tick_values == [5.0, 6.0]
        assert plotter.mapper.cb.tick_labels == [7.0, 8.0]

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
            output=mat_objs.Output(path="Path", format="png", filename="file"),
            include_legend=True,
            legend_fontsize=30,
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
        assert plotter.mask_pointsize == 2
        assert plotter.border_pointsize == 3
        assert plotter.point_pointsize == 4
        assert plotter.grid_pointsize == 5

        plotter = plotters.Plotter(
            figsize=(6, 6),
            aspect="auto",
            mask_pointsize=24,
            border_pointsize=25,
            point_pointsize=26,
            grid_pointsize=27,
        )

        assert plotter.figsize == (6, 6)
        assert plotter.aspect == "auto"
        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27


class TestSubPlotter:
    def test__plotter_settings_use_subplot_config_if_not_manually_input(self):

        plotter = plotters.SubPlotter()

        assert plotter.figsize == None
        assert plotter.aspect == "square"
        
        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.norm_min == None
        assert plotter.cmap.norm_max == None
        assert plotter.cmap.linthresh == 1.0
        assert plotter.cmap.linscale == 2.0
        
        assert plotter.cb.ticksize == 1
        assert plotter.cb.fraction == 3.0
        assert plotter.cb.pad == 4.0
        
        assert plotter.mask_pointsize == 2
        assert plotter.border_pointsize == 3
        assert plotter.point_pointsize == 4
        assert plotter.grid_pointsize == 5

        plotter = plotters.SubPlotter(
            figsize=(6, 6),
            aspect="auto",
            cmap=mat_objs.ColorMap(
            cmap="cold",
            norm="log",
            norm_min=0.1,
            norm_max=1.0,
            linthresh=1.5,
            linscale=2.0),
            cb=mat_objs.ColorBar(
            ticksize=20,
            fraction=0.001,
            pad=10.0),
            mask_pointsize=24,
            border_pointsize=25,
            point_pointsize=26,
            grid_pointsize=27,
        )

        assert plotter.figsize == (6, 6)
        assert plotter.aspect == "auto"

        assert plotter.cmap.cmap == "cold"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.norm_min == 0.1
        assert plotter.cmap.norm_max == 1.0
        assert plotter.cmap.linthresh == 1.5
        assert plotter.cmap.linscale == 2.0

        assert plotter.cb.ticksize == 20
        assert plotter.cb.fraction == 0.001
        assert plotter.cb.pad == 10.0

        assert plotter.mask_pointsize == 24
        assert plotter.border_pointsize == 25
        assert plotter.point_pointsize == 26
        assert plotter.grid_pointsize == 27

    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = plotters.SubPlotter()

        figsize = plotter.get_subplot_figsize(number_subplots=1)

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (13, 10)

        plotter = plotters.SubPlotter(figsize=(20, 20))

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = plotters.SubPlotter()

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=1)

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=4)

        assert rows == 2
        assert columns == 2


class TestArrayPlotter:
    def test__plot_lines__can_take_inputs_of_correct_form(self):

        plotter = plotters.Plotter()

        plotter.array.plot_lines(lines=[(1.0, 1.0), (2.0, 2.0)])
        plotter.array.plot_lines(
            lines=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0), (4.0, 4.0)]]
        )


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
