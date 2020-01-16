import autoarray as aa
import autoarray.plot as aplt
from os import path
import matplotlib.pyplot as plt
import os
import pytest
import numpy as np

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plotter_path")
def make_plotter_setup():
    return "{}/..//test_files/plot/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(autouse=True)
def set_config_path():
    aa.conf.instance = aa.conf.Config(
        path.join(directory, "../test_files/plot"), path.join(directory, "output")
    )


class TestAbstractPlotterAttributes:
    def test__units__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == None

        assert plotter.ticks.units.use_scaled == True
        assert plotter.ticks.units.in_kpc == False
        assert plotter.ticks.units.conversion_factor == None

        assert plotter.labels.units.use_scaled == True
        assert plotter.labels.units.in_kpc == False
        assert plotter.labels.units.conversion_factor == None

        plotter = aplt.Plotter(
            units=aplt.Units(in_kpc=True, use_scaled=False, conversion_factor=2.0)
        )

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 2.0

        assert plotter.ticks.units.use_scaled == False
        assert plotter.ticks.units.in_kpc == True
        assert plotter.ticks.units.conversion_factor == 2.0

        assert plotter.labels.units.use_scaled == False
        assert plotter.labels.units.in_kpc == True
        assert plotter.labels.units.conversion_factor == 2.0

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.units.use_scaled == True
        assert sub_plotter.units.in_kpc == False
        assert sub_plotter.units.conversion_factor == None

        assert sub_plotter.ticks.units.use_scaled == True
        assert sub_plotter.ticks.units.in_kpc == False
        assert sub_plotter.ticks.units.conversion_factor == None

        assert sub_plotter.labels.units.use_scaled == True
        assert sub_plotter.labels.units.in_kpc == False
        assert sub_plotter.labels.units.conversion_factor == None

        sub_plotter = aplt.SubPlotter(
            units=aplt.Units(in_kpc=True, use_scaled=False, conversion_factor=2.0)
        )

        assert sub_plotter.units.use_scaled == False
        assert sub_plotter.units.in_kpc == True
        assert sub_plotter.units.conversion_factor == 2.0

        assert sub_plotter.ticks.units.use_scaled == False
        assert sub_plotter.ticks.units.in_kpc == True
        assert sub_plotter.ticks.units.conversion_factor == 2.0

        assert sub_plotter.labels.units.use_scaled == False
        assert sub_plotter.labels.units.in_kpc == True
        assert sub_plotter.labels.units.conversion_factor == 2.0

    def test__figure__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.figure.figsize == (7, 7)
        assert plotter.figure.aspect == "auto"

        plotter = aplt.Plotter(figure=aplt.Figure(figsize=(6, 6), aspect="auto"))

        assert plotter.figure.figsize == (6, 6)
        assert plotter.figure.aspect == "auto"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.figure.figsize == None
        assert sub_plotter.figure.aspect == "square"

        sub_plotter = aplt.SubPlotter(figure=aplt.Figure(figsize=(6, 6), aspect="auto"))

        assert sub_plotter.figure.figsize == (6, 6)
        assert sub_plotter.figure.aspect == "auto"

    def test__colormap__from_config_or_via_manual_input(self):
        plotter = aplt.Plotter()

        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.norm_min == None
        assert plotter.cmap.norm_max == None
        assert plotter.cmap.linthresh == 1.0
        assert plotter.cmap.linscale == 2.0

        plotter = aplt.Plotter(
            cmap=aplt.ColorMap(
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

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.cmap.cmap == "jet"
        assert sub_plotter.cmap.norm == "linear"
        assert sub_plotter.cmap.norm_min == None
        assert sub_plotter.cmap.norm_max == None
        assert sub_plotter.cmap.linthresh == 1.0
        assert sub_plotter.cmap.linscale == 2.0

        sub_plotter = aplt.SubPlotter(
            cmap=aplt.ColorMap(
                cmap="cold",
                norm="log",
                norm_min=0.1,
                norm_max=1.0,
                linthresh=1.5,
                linscale=2.0,
            )
        )

        assert sub_plotter.cmap.cmap == "cold"
        assert sub_plotter.cmap.norm == "log"
        assert sub_plotter.cmap.norm_min == 0.1
        assert sub_plotter.cmap.norm_max == 1.0
        assert sub_plotter.cmap.linthresh == 1.5
        assert sub_plotter.cmap.linscale == 2.0

    def test__colorbar__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.cb.ticksize == 1
        assert plotter.cb.fraction == 3.0
        assert plotter.cb.pad == 4.0
        assert plotter.cb.tick_values == None
        assert plotter.cb.tick_labels == None

        plotter = aplt.Plotter(
            cb=aplt.ColorBar(
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

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.cb.ticksize == 1
        assert sub_plotter.cb.fraction == 3.0
        assert sub_plotter.cb.pad == 4.0

        sub_plotter = aplt.SubPlotter(
            cb=aplt.ColorBar(ticksize=20, fraction=0.001, pad=10.0)
        )

        assert sub_plotter.cb.ticksize == 20
        assert sub_plotter.cb.fraction == 0.001
        assert sub_plotter.cb.pad == 10.0

    def test__ticks__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.ticks.ysize == 14
        assert plotter.ticks.xsize == 15
        assert plotter.ticks.y_manual == None
        assert plotter.ticks.x_manual == None

        plotter = aplt.Plotter(
            ticks=aplt.Ticks(
                ysize=24, xsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            )
        )

        assert plotter.ticks.ysize == 24
        assert plotter.ticks.xsize == 25
        assert plotter.ticks.y_manual == [1.0, 2.0]
        assert plotter.ticks.x_manual == [3.0, 4.0]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.ticks.ysize == 24
        assert sub_plotter.ticks.xsize == 25
        assert sub_plotter.ticks.y_manual == None
        assert sub_plotter.ticks.x_manual == None

        sub_plotter = aplt.SubPlotter(
            ticks=aplt.Ticks(
                ysize=24, xsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            )
        )

        assert sub_plotter.ticks.ysize == 24
        assert sub_plotter.ticks.xsize == 25
        assert sub_plotter.ticks.y_manual == [1.0, 2.0]
        assert sub_plotter.ticks.x_manual == [3.0, 4.0]

    def test__labels__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.labels.title == None
        assert plotter.labels._yunits == None
        assert plotter.labels._xunits == None
        assert plotter.labels.titlesize == 11
        assert plotter.labels.ysize == 12
        assert plotter.labels.xsize == 13
        assert plotter.labels.units.use_scaled == True

        plotter = aplt.Plotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            ),
            units=aplt.Units(use_scaled=False),
        )

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3
        assert plotter.labels.units.use_scaled == False

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.labels.title == None
        assert sub_plotter.labels._yunits == None
        assert sub_plotter.labels._xunits == None
        assert sub_plotter.labels.titlesize == 15
        assert sub_plotter.labels.ysize == 22
        assert sub_plotter.labels.xsize == 23
        assert sub_plotter.labels.units.use_scaled == True

        sub_plotter = aplt.SubPlotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            ),
            units=aplt.Units(use_scaled=False),
        )

        assert sub_plotter.labels.title == "OMG"
        assert sub_plotter.labels._yunits == "hi"
        assert sub_plotter.labels._xunits == "hi2"
        assert sub_plotter.labels.titlesize == 1
        assert sub_plotter.labels.ysize == 2
        assert sub_plotter.labels.xsize == 3
        assert sub_plotter.labels.units.use_scaled == False

    def test__legend__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.legend.include == True
        assert plotter.legend.fontsize == 12

        plotter = aplt.Plotter(legend=aplt.Legend(include=False, fontsize=11))

        assert plotter.legend.include == False
        assert plotter.legend.fontsize == 11

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.legend.include == False
        assert sub_plotter.legend.fontsize == 13

        sub_plotter = aplt.SubPlotter(legend=aplt.Legend(include=True, fontsize=10))

        assert sub_plotter.legend.include == True
        assert sub_plotter.legend.fontsize == 10

    def test__origin_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.origin_scatterer.size == 80
        assert plotter.origin_scatterer.marker == "x"
        assert plotter.origin_scatterer.color == "k"

        plotter = aplt.Plotter(
            origin_scatterer=aplt.Scatterer(size=1, marker=".", color="k")
        )

        assert plotter.origin_scatterer.size == 1
        assert plotter.origin_scatterer.marker == "."
        assert plotter.origin_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.origin_scatterer.size == 81
        assert sub_plotter.origin_scatterer.marker == "."
        assert sub_plotter.origin_scatterer.color == "r"

        sub_plotter = aplt.SubPlotter(
            origin_scatterer=aplt.Scatterer(size=24, marker="o", color="r")
        )

        assert sub_plotter.origin_scatterer.size == 24
        assert sub_plotter.origin_scatterer.marker == "o"
        assert sub_plotter.origin_scatterer.color == "r"

    def test__mask_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.mask_scatterer.size == 12
        assert plotter.mask_scatterer.marker == "."
        assert plotter.mask_scatterer.color == "g"

        plotter = aplt.Plotter(
            mask_scatterer=aplt.Scatterer(size=1, marker="x", color="k")
        )

        assert plotter.mask_scatterer.size == 1
        assert plotter.mask_scatterer.marker == "x"
        assert plotter.mask_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.mask_scatterer.size == 8
        assert sub_plotter.mask_scatterer.marker == "."
        assert sub_plotter.mask_scatterer.color == "w"

        sub_plotter = aplt.SubPlotter(
            mask_scatterer=aplt.Scatterer(size=24, marker="o", color="r")
        )

        assert sub_plotter.mask_scatterer.size == 24
        assert sub_plotter.mask_scatterer.marker == "o"
        assert sub_plotter.mask_scatterer.color == "r"

    def test__border_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.border_scatterer.size == 13
        assert plotter.border_scatterer.marker == "+"
        assert plotter.border_scatterer.color == "c"

        plotter = aplt.Plotter(
            border_scatterer=aplt.Scatterer(size=1, marker="x", color="k")
        )

        assert plotter.border_scatterer.size == 1
        assert plotter.border_scatterer.marker == "x"
        assert plotter.border_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.border_scatterer.size == 7
        assert sub_plotter.border_scatterer.marker == "."
        assert sub_plotter.border_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter(
            border_scatterer=aplt.Scatterer(size=24, marker="o", color="r")
        )

        assert sub_plotter.border_scatterer.size == 24
        assert sub_plotter.border_scatterer.marker == "o"
        assert sub_plotter.border_scatterer.color == "r"

    def test__grid_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.grid_scatterer.size == 14
        assert plotter.grid_scatterer.marker == "x"
        assert plotter.grid_scatterer.color == "y"

        plotter = aplt.Plotter(
            grid_scatterer=aplt.Scatterer(size=1, marker="x", color="k")
        )

        assert plotter.grid_scatterer.size == 1
        assert plotter.grid_scatterer.marker == "x"
        assert plotter.grid_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.grid_scatterer.size == 6
        assert sub_plotter.grid_scatterer.marker == "."
        assert sub_plotter.grid_scatterer.color == "r"

        sub_plotter = aplt.SubPlotter(
            grid_scatterer=aplt.Scatterer(size=24, marker="o", color="r")
        )

        assert sub_plotter.grid_scatterer.size == 24
        assert sub_plotter.grid_scatterer.marker == "o"
        assert sub_plotter.grid_scatterer.color == "r"

    def test__positions_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.positions_scatterer.size == 15
        assert plotter.positions_scatterer.marker == "o"
        assert plotter.positions_scatterer.color == "r"

        plotter = aplt.Plotter(
            positions_scatterer=aplt.Scatterer(size=1, marker="x", color="k")
        )

        assert plotter.positions_scatterer.size == 1
        assert plotter.positions_scatterer.marker == "x"
        assert plotter.positions_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.positions_scatterer.size == 5
        assert sub_plotter.positions_scatterer.marker == "."
        assert sub_plotter.positions_scatterer.color == "c"

        sub_plotter = aplt.SubPlotter(
            positions_scatterer=aplt.Scatterer(size=24, marker="o", color="r")
        )

        assert sub_plotter.positions_scatterer.size == 24
        assert sub_plotter.positions_scatterer.marker == "o"
        assert sub_plotter.positions_scatterer.color == "r"

    def test__index_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.index_scatterer.size == 20
        assert plotter.index_scatterer.marker == "."
        assert plotter.index_scatterer.color == "r"

        plotter = aplt.Plotter(
            index_scatterer=aplt.Scatterer(size=1, marker="x", color="k")
        )

        assert plotter.index_scatterer.size == 1
        assert plotter.index_scatterer.marker == "x"
        assert plotter.index_scatterer.color == "k"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.index_scatterer.size == 21
        assert sub_plotter.index_scatterer.marker == "+"
        assert sub_plotter.index_scatterer.color == "b"

        sub_plotter = aplt.SubPlotter(
            index_scatterer=aplt.Scatterer(size=24, marker="o", color="r")
        )

        assert sub_plotter.index_scatterer.size == 24
        assert sub_plotter.index_scatterer.marker == "o"
        assert sub_plotter.index_scatterer.color == "r"

    def test__liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.liner.width == 3
        assert plotter.liner.style == "-"
        assert plotter.liner.color == "k"
        assert plotter.liner.pointsize == 2

        plotter = aplt.Plotter(
            liner=aplt.Liner(width=1, style=".", color="k", pointsize=3)
        )

        assert plotter.liner.width == 1
        assert plotter.liner.style == "."
        assert plotter.liner.color == "k"
        assert plotter.liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.liner.width == 1
        assert sub_plotter.liner.style == "-"
        assert sub_plotter.liner.color == "k"
        assert plotter.liner.pointsize == 3

        sub_plotter = aplt.SubPlotter(
            liner=aplt.Liner(width=24, style=".", color="r", pointsize=21)
        )

        assert sub_plotter.liner.width == 24
        assert sub_plotter.liner.style == "."
        assert sub_plotter.liner.color == "r"
        assert sub_plotter.liner.pointsize == 21

    def test__output__correctly(self):

        plotter = aplt.Plotter()

        assert plotter.output.path == None
        assert plotter.output._format == None
        assert plotter.output.format == "show"
        assert plotter.output.filename == None

        plotter = aplt.Plotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.output.path == None
        assert sub_plotter.output._format == None
        assert sub_plotter.output.format == "show"
        assert sub_plotter.output.filename == None

        sub_plotter = aplt.SubPlotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        assert sub_plotter.output.path == "Path"
        assert sub_plotter.output._format == "png"
        assert sub_plotter.output.format == "png"
        assert sub_plotter.output.filename == "file"


class TestAbstractPlotterPlots:
    def test__plot_array__works_with_all_extras_included(
        self, plotter_path, plot_patch
    ):

        array = aa.array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        mask = aa.mask.circular(
            shape_2d=array.shape_2d,
            pixel_scales=array.pixel_scales,
            radius=5.0,
            centre=(2.0, 2.0),
        )

        grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plotter_path, filename="array1", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=[(-1.0, -1.0)],
            lines=[(1.0, 1.0), (2.0, 2.0)],
            include_origin=True,
            include_border=True,
        )

        assert plotter_path + "array1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plotter_path, filename="array2", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=[[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]],
            lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]],
            include_origin=True,
            include_border=True,
        )

        assert plotter_path + "array2.png" in plot_patch.paths

        aplt.array(
            array=array,
            mask=mask,
            grid=grid,
            positions=[(-1.0, -1.0)],
            lines=[(1.0, 1.0), (2.0, 2.0)],
            include_origin=True,
            include_border=True,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plotter_path, filename="array3", format="png")
            ),
        )

        assert plotter_path + "array3.png" in plot_patch.paths

    def test__plot_grid__works_with_all_extras_included(self, plotter_path, plot_patch):
        grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)
        color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plotter_path, filename="grid1", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert plotter_path + "grid1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plotter_path, filename="grid2", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert plotter_path + "grid2.png" in plot_patch.paths

        aplt.grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]],
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plotter_path, filename="grid3", format="png")
            ),
        )

        assert plotter_path + "grid3.png" in plot_patch.paths

    def test__plot_line__works_with_all_extras_included(self, plotter_path, plot_patch):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plotter_path, filename="line1", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plotter_path + "line1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plotter_path, filename="line2", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plotter_path + "line2.png" in plot_patch.paths

        aplt.line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plotter_path, filename="line3", format="png")
            ),
        )

        assert plotter_path + "line3.png" in plot_patch.paths


class TestAbstractPlotterNew:
    def test__plotter_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, ysize=2, xsize=3
            ),
            units=aplt.Units(use_scaled=False),
        )

        plotter = plotter.plotter_with_new_labels()

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.ysize == 2
        assert plotter.labels.xsize == 3
        assert plotter.labels.units.use_scaled == False

        plotter = plotter.plotter_with_new_labels(
            labels=aplt.Labels(
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
        assert plotter.labels.units.use_scaled == False

    def test__plotter_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.plotter_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        plotter = plotter.plotter_with_new_output(
            output=aplt.Output(path="Path0", filename="file0")
        )

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        plotter = plotter.plotter_with_new_output(
            output=aplt.Output(path="Path1", filename="file1", format="fits")
        )

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

    def test__plotter_with_new_units__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            units=aplt.Units(use_scaled=True, in_kpc=True, conversion_factor=1.0)
        )

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 1.0

        assert plotter.labels.units.use_scaled == True
        assert plotter.labels.units.in_kpc == True
        assert plotter.labels.units.conversion_factor == 1.0

        assert plotter.ticks.units.use_scaled == True
        assert plotter.ticks.units.in_kpc == True
        assert plotter.ticks.units.conversion_factor == 1.0

        plotter = plotter.plotter_with_new_units(
            units=aplt.Units(use_scaled=False, in_kpc=False, conversion_factor=2.0)
        )

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 2.0

        assert plotter.labels.units.use_scaled == False
        assert plotter.labels.units.in_kpc == False
        assert plotter.labels.units.conversion_factor == 2.0

        assert plotter.ticks.units.use_scaled == False
        assert plotter.ticks.units.in_kpc == False
        assert plotter.ticks.units.conversion_factor == 2.0

    def test__open_and_close_subplot_figures(self):

        plotter = aplt.Plotter()
        plotter.figure.open()

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False

        plotter = aplt.SubPlotter()

        assert plt.fignum_exists(num=1) == False

        plotter.open_subplot_figure(number_subplots=4)

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False


class TestSubPlotter:
    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = aplt.SubPlotter()

        figsize = plotter.get_subplot_figsize(number_subplots=1)

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (13, 10)

        plotter = aplt.SubPlotter(figure=aplt.Figure(figsize=(20, 20)))

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = aplt.SubPlotter()

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=1)

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=4)

        assert rows == 2
        assert columns == 2


from autoarray.plot import plotters


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
