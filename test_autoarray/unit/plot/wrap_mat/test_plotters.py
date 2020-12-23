from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt
from os import path
import matplotlib.pyplot as plt
import pytest
import numpy as np
import shutil

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plotter"
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files", "plotter"), path.join(directory, "output")
    )


class TestAbstractPlotterAttributes:
    def test__units__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == None

        plotter = aplt.Plotter(units=aplt.Units(in_kpc=True, conversion_factor=2.0))

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 2.0

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.units.use_scaled == True
        assert sub_plotter.units.in_kpc == False
        assert sub_plotter.units.conversion_factor == None

        sub_plotter = aplt.SubPlotter(
            units=aplt.Units(use_scaled=False, conversion_factor=2.0)
        )

        assert sub_plotter.units.use_scaled == False
        assert sub_plotter.units.in_kpc == False
        assert sub_plotter.units.conversion_factor == 2.0

    def test__figure__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.figure.kwargs["figsize"] == (7, 7)
        assert plotter.figure.kwargs["aspect"] == "square"

        plotter = aplt.Plotter(figure=aplt.Figure(aspect="auto"))

        assert plotter.figure.kwargs["figsize"] == (7, 7)
        assert plotter.figure.kwargs["aspect"] == "auto"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.figure.kwargs["figsize"] == None
        assert sub_plotter.figure.kwargs["aspect"] == "square"

        sub_plotter = aplt.SubPlotter(figure=aplt.Figure.sub(figsize=(6, 6)))

        assert sub_plotter.figure.kwargs["figsize"] == (6, 6)
        assert sub_plotter.figure.kwargs["aspect"] == "square"

    def test__colormap__from_config_or_via_manual_input(self):
        plotter = aplt.Plotter()

        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.vmin == None
        assert plotter.cmap.vmax == None
        assert plotter.cmap.linthresh == 1.0
        assert plotter.cmap.linscale == 2.0

        plotter = aplt.Plotter(
            cmap=aplt.Cmap(
                cmap="cold",
                norm="log",
                vmin=0.1,
                vmax=1.0,
                linthresh=1.5,
                linscale=2.0,
            )
        )

        assert plotter.cmap.cmap == "cold"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.vmin == 0.1
        assert plotter.cmap.vmax == 1.0
        assert plotter.cmap.linthresh == 1.5
        assert plotter.cmap.linscale == 2.0

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.cmap.cmap == "jet"
        assert sub_plotter.cmap.norm == "linear"
        assert sub_plotter.cmap.vmin == None
        assert sub_plotter.cmap.vmax == None
        assert sub_plotter.cmap.linthresh == 1.0
        assert sub_plotter.cmap.linscale == 2.0

        sub_plotter = aplt.SubPlotter(
            cmap=aplt.Cmap.sub(
                cmap="cold", norm="log", vmin=0.1, vmax=1.0, linscale=2.0
            )
        )

        assert sub_plotter.cmap.cmap == "cold"
        assert sub_plotter.cmap.norm == "log"
        assert sub_plotter.cmap.vmin == 0.1
        assert sub_plotter.cmap.vmax == 1.0
        assert sub_plotter.cmap.linthresh == 1.0
        assert sub_plotter.cmap.linscale == 2.0

    def test__colorbar__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.cb.kwargs["labelsize"] == 1
        assert plotter.cb.manual_tick_values == None
        assert plotter.cb.manual_tick_labels == None

        plotter = aplt.Plotter(
            cb=aplt.Colorbar(
                labelsize=20,
                manual_tick_values=(1.0, 2.0),
                manual_tick_labels=(3.0, 4.0),
            )
        )

        assert plotter.cb.kwargs["labelsize"] == 20
        assert plotter.cb.manual_tick_values == (1.0, 2.0)
        assert plotter.cb.manual_tick_labels == (3.0, 4.0)

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.cb.kwargs["labelsize"] == 1

        sub_plotter = aplt.SubPlotter(cb=aplt.Colorbar.sub(labelsize=10))

        assert sub_plotter.cb.kwargs["labelsize"] == 10

    def test__ticks__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.ticks.labelsize == 14
        assert plotter.ticks.labelsize == 15
        assert plotter.ticks.manual_values == None
        assert plotter.ticks.manual_values == None

        plotter = aplt.Plotter(
            ticks=aplt.Ticks(
                labelsize=24, labelsize=25, y_manual=[1.0, 2.0], x_manual=[3.0, 4.0]
            )
        )

        assert plotter.ticks.labelsize == 24
        assert plotter.ticks.labelsize == 25
        assert plotter.ticks.manual_values == [1.0, 2.0]
        assert plotter.ticks.manual_values == [3.0, 4.0]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.ticks.labelsize == 24
        assert sub_plotter.ticks.labelsize == 25
        assert sub_plotter.ticks.manual_values == None
        assert sub_plotter.ticks.manual_values == None

        sub_plotter = aplt.SubPlotter(
            ticks=aplt.Ticks.sub(labelsize=25, manual_values=[1.0, 2.0], x_manual=[3.0, 4.0])
        )

        assert sub_plotter.ticks.labelsize == 24
        assert sub_plotter.ticks.labelsize == 25
        assert sub_plotter.ticks.manual_values == [1.0, 2.0]
        assert sub_plotter.ticks.manual_values == [3.0, 4.0]

    def test__labels__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.labels.title == None
        assert plotter.labels._yunits == None
        assert plotter.labels._xunits == None
        assert plotter.labels.titlesize == 11
        assert plotter.labels.labelsize == 12
        assert plotter.labels.labelsize == 13

        plotter = aplt.Plotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, labelsize=2, labelsize=3
            )
        )

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.labelsize == 2
        assert plotter.labels.labelsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.labels.title == None
        assert sub_plotter.labels._yunits == None
        assert sub_plotter.labels._xunits == None
        assert sub_plotter.labels.titlesize == 15
        assert sub_plotter.labels.labelsize == 22
        assert sub_plotter.labels.labelsize == 23

        sub_plotter = aplt.SubPlotter(
            labels=aplt.Labels.sub(
                title="OMG", yunits="hi", xunits="hi2", labelsize=2, labelsize=3
            )
        )

        assert sub_plotter.labels.title == "OMG"
        assert sub_plotter.labels._yunits == "hi"
        assert sub_plotter.labels._xunits == "hi2"
        assert sub_plotter.labels.titlesize == 15
        assert sub_plotter.labels.labelsize == 2
        assert sub_plotter.labels.labelsize == 3

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

        sub_plotter = aplt.SubPlotter(legend=aplt.Legend.sub(include=True))

        assert sub_plotter.legend.include == True
        assert sub_plotter.legend.fontsize == 13

    def test__origin_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.origin_scatter.size == 80
        assert plotter.origin_scatter.marker == "x"
        assert plotter.origin_scatter.colors == ["k"]

        plotter = aplt.Plotter(
            origin_scatter=aplt.OriginScatter(size=1, marker=".", colors="k")
        )

        assert plotter.origin_scatter.size == 1
        assert plotter.origin_scatter.marker == "."
        assert plotter.origin_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.origin_scatter.size == 81
        assert sub_plotter.origin_scatter.marker == "."
        assert sub_plotter.origin_scatter.colors == ["r"]

        sub_plotter = aplt.SubPlotter(
            origin_scatter=aplt.OriginScatter.sub(marker="o", colors=["r", "b"])
        )

        assert sub_plotter.origin_scatter.size == 81
        assert sub_plotter.origin_scatter.marker == "o"
        assert sub_plotter.origin_scatter.colors == ["r", "b"]

    def test__mask_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.mask_scatter.size == 12
        assert plotter.mask_scatter.marker == "."
        assert plotter.mask_scatter.colors == ["g"]

        plotter = aplt.Plotter(
            mask_scatter=aplt.MaskScatter(size=1, marker="x", colors="k")
        )

        assert plotter.mask_scatter.size == 1
        assert plotter.mask_scatter.marker == "x"
        assert plotter.mask_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.mask_scatter.size == 8
        assert sub_plotter.mask_scatter.marker == "."
        assert sub_plotter.mask_scatter.colors == ["w"]

        sub_plotter = aplt.SubPlotter(
            mask_scatter=aplt.MaskScatter.sub(marker="o", colors=["r", "b"])
        )

        assert sub_plotter.mask_scatter.size == 8
        assert sub_plotter.mask_scatter.marker == "o"
        assert sub_plotter.mask_scatter.colors == ["r", "b"]

    def test__border_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.border_scatter.size == 13
        assert plotter.border_scatter.marker == "+"
        assert plotter.border_scatter.colors == ["c"]

        plotter = aplt.Plotter(
            border_scatter=aplt.BorderScatter(size=1, marker="x", colors="k")
        )

        assert plotter.border_scatter.size == 1
        assert plotter.border_scatter.marker == "x"
        assert plotter.border_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.border_scatter.size == 7
        assert sub_plotter.border_scatter.marker == "."
        assert sub_plotter.border_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter(
            border_scatter=aplt.BorderScatter.sub(marker="o", colors=["r", "b"])
        )

        assert sub_plotter.border_scatter.size == 7
        assert sub_plotter.border_scatter.marker == "o"
        assert sub_plotter.border_scatter.colors == ["r", "b"]

    def test__grid_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.grid_scatter.size == 14
        assert plotter.grid_scatter.marker == "x"
        assert plotter.grid_scatter.colors == ["y"]

        plotter = aplt.Plotter(
            grid_scatter=aplt.GridScatter(size=1, marker="x", colors="k")
        )

        assert plotter.grid_scatter.size == 1
        assert plotter.grid_scatter.marker == "x"
        assert plotter.grid_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.grid_scatter.size == 6
        assert sub_plotter.grid_scatter.marker == "."
        assert sub_plotter.grid_scatter.colors == ["r"]

        sub_plotter = aplt.SubPlotter(
            grid_scatter=aplt.GridScatter.sub(marker="o", colors=["r", "b"])
        )

        assert sub_plotter.grid_scatter.size == 6
        assert sub_plotter.grid_scatter.marker == "o"
        assert sub_plotter.grid_scatter.colors == ["r", "b"]

    def test__positions_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.positions_scatter.size == 15
        assert plotter.positions_scatter.marker == "o"
        assert plotter.positions_scatter.colors == ["r", "g", "b"]

        plotter = aplt.Plotter(
            positions_scatter=aplt.PositionsScatter(size=1, marker="x", colors="k")
        )

        assert plotter.positions_scatter.size == 1
        assert plotter.positions_scatter.marker == "x"
        assert plotter.positions_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.positions_scatter.size == 5
        assert sub_plotter.positions_scatter.marker == "."
        assert sub_plotter.positions_scatter.colors == ["c", "g", "b"]

        sub_plotter = aplt.SubPlotter(
            positions_scatter=aplt.PositionsScatter.sub(
                marker="o", colors=["r", "b"]
            )
        )

        assert sub_plotter.positions_scatter.size == 5
        assert sub_plotter.positions_scatter.marker == "o"
        assert sub_plotter.positions_scatter.colors == ["r", "b"]

    def test__index_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.index_scatter.size == 20
        assert plotter.index_scatter.marker == "."
        assert plotter.index_scatter.colors == ["r", "g", "b", "y", "k", "w"]

        plotter = aplt.Plotter(
            index_scatter=aplt.IndexScatter(size=1, marker="x", colors="k")
        )

        assert plotter.index_scatter.size == 1
        assert plotter.index_scatter.marker == "x"
        assert plotter.index_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.index_scatter.size == 21
        assert sub_plotter.index_scatter.marker == "+"
        assert sub_plotter.index_scatter.colors == ["r", "g", "b", "y", "w", "k"]

        sub_plotter = aplt.SubPlotter(
            index_scatter=aplt.IndexScatter.sub(marker="o", colors="r")
        )

        assert sub_plotter.index_scatter.size == 21
        assert sub_plotter.index_scatter.marker == "o"
        assert sub_plotter.index_scatter.colors == ["r"]

    def test__pixelization_grid_scatter__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.pixelization_grid_scatter.size == 5
        assert plotter.pixelization_grid_scatter.marker == "."
        assert plotter.pixelization_grid_scatter.colors == ["r"]

        plotter = aplt.Plotter(
            pixelization_grid_scatter=aplt.PixelizationGridScatter(
                size=1, marker="x", colors="k"
            )
        )

        assert plotter.pixelization_grid_scatter.size == 1
        assert plotter.pixelization_grid_scatter.marker == "x"
        assert plotter.pixelization_grid_scatter.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.pixelization_grid_scatter.size == 6
        assert sub_plotter.pixelization_grid_scatter.marker == "o"
        assert sub_plotter.pixelization_grid_scatter.colors == ["g"]

        sub_plotter = aplt.SubPlotter(
            pixelization_grid_scatter=aplt.PixelizationGridScatter.sub(
                marker="o", colors="r"
            )
        )

        assert sub_plotter.pixelization_grid_scatter.size == 6
        assert sub_plotter.pixelization_grid_scatter.marker == "o"
        assert sub_plotter.pixelization_grid_scatter.colors == ["r"]

    def test__vector_quiver__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.vector_quiver.headlength == 0
        assert plotter.vector_quiver.pivot == "middle"
        assert plotter.vector_quiver.linewidth == 5
        assert plotter.vector_quiver.units == "xy"
        assert plotter.vector_quiver.angles == "xy"
        assert plotter.vector_quiver.headwidth == 1
        assert plotter.vector_quiver.alpha == 1.0

        plotter = aplt.Plotter(
            vector_quiver=aplt.VectorFieldQuiver(
                headlength=1,
                pivot="lol",
                linewidth=2,
                units="lol2",
                angles="lol3",
                headwidth=3,
                alpha=0.5,
            )
        )

        assert plotter.vector_quiver.headlength == 1
        assert plotter.vector_quiver.pivot == "lol"
        assert plotter.vector_quiver.linewidth == 2
        assert plotter.vector_quiver.units == "lol2"
        assert plotter.vector_quiver.angles == "lol3"
        assert plotter.vector_quiver.headwidth == 3
        assert plotter.vector_quiver.alpha == 0.5

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.vector_quiver.headlength == 0.1
        assert sub_plotter.vector_quiver.pivot == "middle1"
        assert sub_plotter.vector_quiver.linewidth == 51
        assert sub_plotter.vector_quiver.units == "xy1"
        assert sub_plotter.vector_quiver.angles == "xy1"
        assert sub_plotter.vector_quiver.headwidth == 11
        assert sub_plotter.vector_quiver.alpha == 1.1

        sub_plotter = aplt.SubPlotter(
            vector_quiver=aplt.VectorFieldQuiver(
                headlength=12,
                pivot="lol2",
                linewidth=22,
                units="lol22",
                angles="lol32",
                headwidth=32,
                alpha=0.52,
            )
        )

        assert sub_plotter.vector_quiver.headlength == 12
        assert sub_plotter.vector_quiver.pivot == "lol2"
        assert sub_plotter.vector_quiver.linewidth == 22
        assert sub_plotter.vector_quiver.units == "lol22"
        assert sub_plotter.vector_quiver.angles == "lol32"
        assert sub_plotter.vector_quiver.headwidth == 32
        assert sub_plotter.vector_quiver.alpha == 0.52

    def test__patcher__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.patcher.facecolor == "none"
        assert plotter.patcher.edgecolor == "cyan"

        plotter = aplt.Plotter(patcher=aplt.Patcher(facecolor="r", edgecolor="g"))

        assert plotter.patcher.facecolor == "r"
        assert plotter.patcher.edgecolor == "g"

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.patcher.facecolor == "none"
        assert sub_plotter.patcher.edgecolor == "y"

        sub_plotter = aplt.SubPlotter(
            patcher=aplt.Patcher.sub(facecolor="b", edgecolor="p")
        )

        assert sub_plotter.patcher.facecolor == "b"
        assert sub_plotter.patcher.edgecolor == "p"

    def test__line__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.line.width == 3
        assert plotter.line.style == "-"
        assert plotter.line.colors == ["k", "w"]
        assert plotter.line.pointsize == 2

        plotter = aplt.Plotter(
            line=aplt.Line(width=1, style=".", colors=["k", "b"], pointsize=3)
        )

        assert plotter.line.width == 1
        assert plotter.line.style == "."
        assert plotter.line.colors == ["k", "b"]
        assert plotter.line.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.line.width == 1
        assert sub_plotter.line.style == "-"
        assert sub_plotter.line.colors == ["k"]
        assert sub_plotter.line.pointsize == 20

        sub_plotter = aplt.SubPlotter(
            line=aplt.Line.sub(style=".", colors="r", pointsize=21)
        )

        assert sub_plotter.line.width == 1
        assert sub_plotter.line.style == "."
        assert sub_plotter.line.colors == ["r"]
        assert sub_plotter.line.pointsize == 21

    def test__array_over__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.array_over.alpha == 0.5

        plotter = aplt.Plotter(array_over=aplt.ArrayOverlayer(alpha=0.6))

        assert plotter.array_over.alpha == 0.6

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.array_over.alpha == 0.7

        sub_plotter = aplt.SubPlotter(
            array_over=aplt.ArrayOverlayer.sub(alpha=0.8)
        )

        assert sub_plotter.array_over.alpha == 0.8

    def test__voronoi_drawer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.voronoi_drawer.edgewidth == 0.3
        assert plotter.voronoi_drawer.edgecolor == "k"
        assert plotter.voronoi_drawer.alpha == 0.7

        plotter = aplt.Plotter(
            voronoi_drawer=aplt.VoronoiDrawer(edgewidth=0.5, edgecolor="r", alpha=1.0)
        )

        assert plotter.voronoi_drawer.edgewidth == 0.5
        assert plotter.voronoi_drawer.edgecolor == "r"
        assert plotter.voronoi_drawer.alpha == 1.0

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.voronoi_drawer.edgewidth == 1.0
        assert sub_plotter.voronoi_drawer.edgecolor == "r"
        assert sub_plotter.voronoi_drawer.alpha == 0.5

        sub_plotter = aplt.SubPlotter(
            voronoi_drawer=aplt.VoronoiDrawer.sub(edgecolor="r", alpha=1.0)
        )

        assert sub_plotter.voronoi_drawer.edgewidth == 1.0
        assert sub_plotter.voronoi_drawer.edgecolor == "r"
        assert sub_plotter.voronoi_drawer.alpha == 1.0

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

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

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

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)


class TestAbstractPlotterPlots:
    def test__plot_array__works_with_all_extras_included(self, plot_path, plot_patch):

        array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        mask = aa.Mask2D.circular(
            shape_2d=array.shape_2d,
            pixel_scales=array.pixel_scales,
            radius=5.0,
            centre=(2.0, 2.0),
        )

        grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array1", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=aa.GridIrregularGrouped([(-1.0, -1.0)]),
            lines=aa.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            array_over=array,
            include_origin=True,
            include_border=True,
        )

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array2", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]
            ),
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            array_over=array,
            include_origin=True,
            include_border=True,
        )

        assert path.join(plot_path, "array2.png") in plot_patch.paths

        aplt.Array(
            array=array,
            mask=mask,
            grid=grid,
            positions=aa.GridIrregularGrouped([(-1.0, -1.0)]),
            lines=aa.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            array_over=array,
            include=aplt.Include(origin=True, border=True),
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__plot_array__fits_files_output_correctly(self, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        arr = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array", format="fits")
        )

        plotter.plot_array(array=arr)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == np.ones(shape=(31, 31))).all()

        mask = aa.Mask2D.circular(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), radius=5.0, centre=(2.0, 2.0)
        )

        masked_array = aa.Array.manual_mask(array=arr, mask=mask)

        plotter.plot_array(array=masked_array)

        arr = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert arr.shape == (13, 13)

    def test__plot_frame__works_with_all_extras_included(self, plot_path, plot_patch):

        frame = aa.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame1", format="png")
        )

        plotter.plot_frame(frame=frame, include_origin=True)

        assert path.join(plot_path, "frame1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame2", format="png")
        )

        plotter.plot_frame(frame=frame, include_origin=True)

        assert path.join(plot_path, "frame2.png") in plot_patch.paths

        aplt.Frame(
            frame=frame,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="frame3", format="png")
            ),
        )

        assert path.join(plot_path, "frame3.png") in plot_patch.paths

    def test__plot_frame__fits_files_output_correctly(self, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        frame = aa.Frame.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="frame", format="fits")
        )

        plotter.plot_frame(frame=frame)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert (frame == np.ones(shape=(31, 31))).all()

        mask = aa.Mask2D.unmasked(shape_2d=(31, 31), pixel_scales=(1.0, 1.0))

        masked_frame = aa.Frame.manual_mask(array=frame, mask=mask)

        plotter.plot_frame(frame=masked_frame)

        frame = aa.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "frame.fits"), hdu=0
        )

        assert frame.shape == (31, 31)

    def test__plot_grid__works_with_all_extras_included(self, plot_path, plot_patch):
        grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)
        color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid1", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid2", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert path.join(plot_path, "grid2.png") in plot_patch.paths

        aplt.Grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            lines=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
        )

        assert path.join(plot_path, "grid3.png") in plot_patch.paths

    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line1", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line2", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line2.png") in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert path.join(plot_path, "line3.png") in plot_patch.paths

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            include=aplt.Include(
                inversion_pixelization_grid=True,
                inversion_grid=True,
                inversion_border=True,
            ),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            include=aplt.Include(
                inversion_pixelization_grid=True,
                inversion_grid=True,
                inversion_border=True,
            ),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths


class TestAbstractPlotterNew:
    def test__plotter_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            labels=aplt.Labels(
                title="OMG", yunits="hi", xunits="hi2", titlesize=1, labelsize=2, labelsize=3
            )
        )

        plotter = plotter.plotter_with_new_labels()

        assert plotter.labels.title == "OMG"
        assert plotter.labels._yunits == "hi"
        assert plotter.labels._xunits == "hi2"
        assert plotter.labels.titlesize == 1
        assert plotter.labels.labelsize == 2
        assert plotter.labels.labelsize == 3

        plotter = plotter.plotter_with_new_labels(
            title="OMG0", yunits="hi0", xunits="hi20", titlesize=10, labelsize=20, labelsize=30
        )

        assert plotter.labels.title == "OMG0"
        assert plotter.labels._yunits == "hi0"
        assert plotter.labels._xunits == "hi20"
        assert plotter.labels.titlesize == 10
        assert plotter.labels.labelsize == 20
        assert plotter.labels.labelsize == 30

        plotter = plotter.plotter_with_new_labels(
            title="OMG0", yunits="hi0", xunits="hi20", titlesize=10
        )

        assert plotter.labels.title == "OMG0"
        assert plotter.labels._yunits == "hi0"
        assert plotter.labels._xunits == "hi20"
        assert plotter.labels.titlesize == 10
        assert plotter.labels.labelsize == 20
        assert plotter.labels.labelsize == 30

    def test__plotter_with_new_cmap__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.Plotter(
            cmap=aplt.Cmap(
                cmap="cold",
                norm="log",
                vmin=0.1,
                vmax=1.0,
                linthresh=1.5,
                linscale=2.0,
            )
        )

        assert plotter.cmap.cmap == "cold"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.vmin == 0.1
        assert plotter.cmap.vmax == 1.0
        assert plotter.cmap.linthresh == 1.5
        assert plotter.cmap.linscale == 2.0

        plotter = plotter.plotter_with_new_cmap(
            cmap="jet",
            norm="linear",
            vmin=0.12,
            vmax=1.2,
            linthresh=1.2,
            linscale=2.2,
        )

        assert plotter.cmap.cmap == "jet"
        assert plotter.cmap.norm == "linear"
        assert plotter.cmap.vmin == 0.12
        assert plotter.cmap.vmax == 1.2
        assert plotter.cmap.linthresh == 1.2
        assert plotter.cmap.linscale == 2.2

        plotter = plotter.plotter_with_new_cmap(cmap="sand", norm="log", vmin=0.13)

        assert plotter.cmap.cmap == "sand"
        assert plotter.cmap.norm == "log"
        assert plotter.cmap.vmin == 0.13
        assert plotter.cmap.vmax == 1.2
        assert plotter.cmap.linthresh == 1.2
        assert plotter.cmap.linscale == 2.2

    def test__plotter_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.plotter_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.plotter_with_new_output(path="Path0", filename="file0")

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.plotter_with_new_output(
            path="Path1", filename="file1", format="fits"
        )

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

    def test__plotter_with_new_units__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.Plotter(
            units=aplt.Units(use_scaled=True, in_kpc=True, conversion_factor=1.0)
        )

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 1.0

        plotter = plotter.plotter_with_new_units(
            use_scaled=False, in_kpc=False, conversion_factor=2.0
        )

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 2.0

        plotter = plotter.plotter_with_new_units(conversion_factor=3.0)

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 3.0

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


from autoarray.plot.wrap_mat import plotters


class TestDecorator:
    def test__kpc_per_scaled_extacted_from_object_if_available(self):

        dictionary = {"hi": 1}

        kpc_per_scaled = plotters.kpc_per_scaled_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_scaled == None

        class MockObj:
            def __init__(self, param1):

                self.param1 = param1

        obj = MockObj(param1=1)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_scaled = plotters.kpc_per_scaled_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_scaled == None

        class MockObj:
            def __init__(self, param1, kpc_per_scaled):

                self.param1 = param1
                self.kpc_per_scaled = kpc_per_scaled

        obj = MockObj(param1=1, kpc_per_scaled=2)

        dictionary = {"hi": 1, "hello": obj}

        kpc_per_scaled = plotters.kpc_per_scaled_of_object_from_dictionary(
            dictionary=dictionary
        )

        assert kpc_per_scaled == 2
