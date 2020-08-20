from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt
from autoarray.plot import mat_objs

from os import path


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pytest
import os, shutil
import numpy as np

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


class TestFigure:
    def test__aspect_from_shape_2d(self):

        figure = aplt.Figure(aspect="auto")

        aspect = figure.aspect_from_shape_2d(shape_2d=(2, 2))

        assert aspect == "auto"

        figure = aplt.Figure(aspect="square")

        aspect = figure.aspect_from_shape_2d(shape_2d=(2, 2))

        assert aspect == 1.0

        aspect = figure.aspect_from_shape_2d(shape_2d=(4, 2))

        assert aspect == 0.5

    def test__open_and_close__open_and_close_figures_correct(self):

        figure = aplt.Figure()

        figure.open()

        assert plt.fignum_exists(num=1) == True

        figure.close()

        assert plt.fignum_exists(num=1) == False


class TestColorMap:
    def test__norm_from_array__uses_input_norm_min_and_max_if_input(self):

        cmap = aplt.ColorMap(norm_min=0.0, norm_max=1.0, norm="linear")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.Normalize)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

        cmap = aplt.ColorMap(norm_min=0.0, norm_max=1.0, norm="log")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.LogNorm)
        assert norm.vmin == 1.0e-4  # Increased from 0.0 to ensure min isn't inf
        assert norm.vmax == 1.0

        cmap = aplt.ColorMap(
            norm_min=0.0,
            norm_max=1.0,
            linthresh=2.0,
            linscale=3.0,
            norm="symmetric_log",
        )

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.SymLogNorm)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0
        assert norm.linthresh == 2.0

    def test__norm_from_array__uses_array_to_get_norm_min_and_max_if_no_manual_input(
        self
    ):

        array = aa.Array.ones(shape_2d=(2, 2))
        array[0] = 0.0

        cmap = aplt.ColorMap(norm_min=None, norm_max=None, norm="linear")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.Normalize)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

        cmap = aplt.ColorMap(norm_min=None, norm_max=None, norm="log")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.LogNorm)
        assert norm.vmin == 1.0e-4  # Increased from 0.0 to ensure min isn't inf
        assert norm.vmax == 1.0

        cmap = aplt.ColorMap(
            norm_min=None,
            norm_max=None,
            linthresh=2.0,
            linscale=3.0,
            norm="symmetric_log",
        )

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.SymLogNorm)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0
        assert norm.linthresh == 2.0


class TestColorBar:
    def test__plot__works_for_reasonable_range_of_values(self):

        figure = aplt.Figure()

        figure.open()
        plt.imshow(np.ones((2, 2)))
        cb = aplt.ColorBar(ticksize=1, fraction=1.0, pad=2.0)
        cb.set()
        figure.close()

        figure.open()
        plt.imshow(np.ones((2, 2)))
        cb = aplt.ColorBar(
            ticksize=1,
            fraction=0.1,
            pad=0.5,
            tick_values=[0.25, 0.5, 0.75],
            tick_labels=[1.0, 2.0, 3.0],
        )
        cb.set()
        figure.close()

        figure.open()
        plt.imshow(np.ones((2, 2)))
        cb = aplt.ColorBar(ticksize=1, fraction=0.1, pad=0.5)
        cb.set_with_values(cmap=aplt.ColorMap().cmap, color_values=[1.0, 2.0, 3.0])
        figure.close()


class TestTicks:
    def test__set_yx_ticks__works_for_good_values(self):

        array = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

        units = aplt.Units(use_scaled=True, conversion_factor=None)

        ticks = aplt.Ticks(ysize=34, xsize=35)

        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )

        ticks = aplt.Ticks(ysize=34, xsize=35)

        units = aplt.Units(use_scaled=False, conversion_factor=None)

        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )

        ticks = aplt.Ticks(ysize=34, xsize=35)

        units = aplt.Units(use_scaled=True, conversion_factor=2.0)

        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )

        ticks = aplt.Ticks(ysize=34, xsize=35)

        units = aplt.Units(use_scaled=False, conversion_factor=2.0)

        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=False,
        )
        ticks.set_yticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )
        ticks.set_xticks(
            array=array,
            extent=array.extent_of_zoomed_array(buffer=1),
            units=units,
            symmetric_around_centre=True,
        )


class TestLabels:
    def test__yx_units_use_plot_in_kpc_if_it_is_passed(self):

        labels = aplt.Labels()

        units = aplt.Units(in_kpc=True)

        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits_from_units(units=units) == "kpc"
        assert labels.xunits_from_units(units=units) == "kpc"

        labels = aplt.Labels()

        units = aplt.Units(in_kpc=False)

        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits_from_units(units=units) == "arcsec"
        assert labels.xunits_from_units(units=units) == "arcsec"

        labels = aplt.Labels(yunits="hi", xunits="hi2")

        units = aplt.Units(in_kpc=True)

        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits_from_units(units=units) == "hi"
        assert labels.xunits_from_units(units=units) == "hi2"

        labels = aplt.Labels(yunits="hi", xunits="hi2")

        units = aplt.Units(in_kpc=False)

        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits_from_units(units=units) == "hi"
        assert labels.xunits_from_units(units=units) == "hi2"

    def test__title_from_func__uses_func_name_if_title_is_none(self):
        def toy_func():
            pass

        labels = aplt.Labels(title=None)

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Toy_func"

        labels = aplt.Labels(title="Hi")

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Hi"

    def test__yx_units_from_func__uses_function_inputs_if_available(self):
        def toy_func():
            pass

        labels = aplt.Labels(yunits=None, xunits=None)

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == None
        assert xunits_from_func == None

        def toy_func(label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = aplt.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = aplt.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"

        def toy_func(argument, label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = aplt.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = aplt.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"


class TestLegend:
    def test__set_legend_works_for_plot(self):

        figure = aplt.Figure(aspect="auto")

        figure.open()

        liner = aplt.Liner(width=2, style="-", colors="k", pointsize=2)

        liner.draw_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear", label="hi"
        )

        legend = aplt.Legend(include=True, fontsize=1)

        legend.set()

        figure.close()


class TestOutput:
    def test__input_path_is_created(self):

        test_path = path.join(directory, "files/output_path")

        if os.path.exists(test_path):
            shutil.rmtree(test_path)

        assert not os.path.exists(test_path)

        output = aplt.Output(path=test_path)

        assert os.path.exists(test_path)

    def test__filename_from_func__returns_function_name_if_no_filename(self):
        def toy_func():
            pass

        output = aplt.Output(filename=None)

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "toy_func"

        output = aplt.Output(filename="Hi")

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "Hi"


class TestScatterer:
    def test__scatter_grid(self):

        scatterer = mat_objs.Scatterer(size=2, marker="x", colors="k")

        scatterer.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    def test__scatter_colored_grid__lists_of_coordinates_or_equivalent_2d_grids__with_color_array(
        self
    ):

        scatterer = mat_objs.Scatterer(size=2, marker="x", colors="k")

        cmap = plt.get_cmap("jet")

        scatterer.scatter_colored_grid(
            grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
            color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            cmap=cmap,
        )
        scatterer.scatter_colored_grid(
            grid=aa.Grid.uniform(shape_2d=(3, 2), pixel_scales=1.0),
            color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            cmap=cmap,
        )

    def test__scatter_grid_indexes_1d__input_grid_is_ndarray_and_indexes_are_valid(
        self
    ):

        scatterer = mat_objs.Scatterer(size=2, marker="x", colors="k")

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0), indexes=[0, 1, 2]
        )

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0), indexes=[[0, 1, 2]]
        )

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[0, 1], [2]],
        )

    def test__scatter_grid_indexes_2d__input_grid_is_ndarray_and_indexes_are_valid(
        self
    ):

        scatterer = mat_objs.Scatterer(size=2, marker="x", colors="k")

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[(0, 0), (0, 1), (0, 2)],
        )

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[(0, 0), (0, 1), (0, 2)]],
        )

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[(0, 0), (0, 1)], [(0, 2)]],
        )

        scatterer.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[[0, 0], [0, 1]], [[0, 2]]],
        )

    def test__scatter_coordinates(self):

        scatterer = mat_objs.Scatterer(size=2, marker="x", colors="k")

        scatterer.scatter_coordinates(
            coordinates=aa.GridCoordinates([(1.0, 1.0), (2.0, 2.0)])
        )
        scatterer.scatter_coordinates(
            coordinates=aa.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])
        )


class TestLiner:
    def test__draw_y_vs_x__works_for_reasonable_values(self):

        liner = aplt.Liner(width=2, style="-", colors="k", pointsize=2)

        liner.draw_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear")
        liner.draw_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy"
        )
        liner.draw_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog")
        liner.draw_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="scatter"
        )

    def test__draw_vertical_lines__works_for_reasonable_values(self):

        liner = aplt.Liner(width=2, style="-", colors="k", pointsize=2)

        liner.draw_vertical_lines(vertical_lines=[[0.0]])
        liner.draw_vertical_lines(vertical_lines=[[1.0], [2.0]])
        liner.draw_vertical_lines(vertical_lines=[[0.0]], vertical_line_labels=["hi"])
        liner.draw_vertical_lines(
            vertical_lines=[[1.0], [2.0]], vertical_line_labels=["hi1", "hi2"]
        )

    def test__draw_grid__lists_of_coordinates_or_equivalent_2d_grids(self):

        liner = aplt.Liner(width=2, style="-", colors="k")

        liner.draw_grid(grid=[[1.0, 1.0], [2.0, 2.0]])
        liner.draw_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))
        liner.draw_grid(grid=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
        liner.draw_grid(
            grid=[
                aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
                aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            ]
        )

    def test__draw_coordinates(self):

        liner = aplt.Liner(width=2, style="--", colors="k")

        liner.draw_coordinates(
            coordinates=aa.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)]])
        )
        liner.draw_coordinates(
            coordinates=aa.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])
        )

    def test__draw_rectangular_grid_lines__draws_for_valid_extent_and_shape(self):

        liner = aplt.Liner(width=2, style="--", colors="k")

        liner.draw_rectangular_grid_lines(extent=[0.0, 1.0, 0.0, 1.0], shape_2d=(3, 2))
        liner.draw_rectangular_grid_lines(
            extent=[-4.0, 8.0, -3.0, 10.0], shape_2d=(8, 3)
        )


class TestArrayOverlayer:
    def test__overlay_array__works_for_reasonable_values(self):

        array_overlay = aa.Array.manual_2d(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.5
        )

        figure = aplt.Figure(aspect="auto")

        array_overlayer = aplt.ArrayOverlayer(alpha=0.5)

        array_overlayer.overlay_array(array_overlay=array_overlay, figure=figure)


class TestVoronoiDrawer:
    def test__draws_voronoi_pixels_for_sensible_input(self, voronoi_mapper_9_3x3):

        voronoi_drawer = aplt.VoronoiDrawer(edgewidth=0.5, edgecolor="r", alpha=1.0)

        voronoi_drawer.draw_voronoi_pixels(
            mapper=voronoi_mapper_9_3x3, values=None, cmap=None, cb=None
        )

        voronoi_drawer.draw_voronoi_pixels(
            mapper=voronoi_mapper_9_3x3,
            values=np.ones(9),
            cmap="jet",
            cb=aplt.ColorBar(ticksize=1, fraction=0.1, pad=0.05),
        )
