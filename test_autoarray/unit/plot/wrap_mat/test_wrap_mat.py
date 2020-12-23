from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt
from autoarray.plot import wrap_mat

from os import path

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pytest
import os, shutil
import numpy as np

directory = path.dirname(path.realpath(__file__))


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


class TestCmap:
    def test__norm_from_array__uses_input_vmin_and_max_if_input(self):

        cmap = aplt.Cmap(vmin=0.0, vmax=1.0, norm="linear")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.Normalize)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

        cmap = aplt.Cmap(vmin=0.0, vmax=1.0, norm="log")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.LogNorm)
        assert norm.vmin == 1.0e-4  # Increased from 0.0 to ensure min isn't inf
        assert norm.vmax == 1.0

        cmap = aplt.Cmap(
            vmin=0.0, vmax=1.0, linthresh=2.0, linscale=3.0, norm="symmetric_log"
        )

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.SymLogNorm)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0
        assert norm.linthresh == 2.0

    def test__norm_from_array__uses_array_to_get_vmin_and_max_if_no_manual_input(self,):

        array = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)
        array[0] = 0.0

        cmap = aplt.Cmap(vmin=None, vmax=None, norm="linear")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.Normalize)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

        cmap = aplt.Cmap(vmin=None, vmax=None, norm="log")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.LogNorm)
        assert norm.vmin == 1.0e-4  # Increased from 0.0 to ensure min isn't inf
        assert norm.vmax == 1.0

        cmap = aplt.Cmap(
            vmin=None, vmax=None, linthresh=2.0, linscale=3.0, norm="symmetric_log"
        )

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.SymLogNorm)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0
        assert norm.linthresh == 2.0


class TestColorbar:
    def test__plot__works_for_reasonable_range_of_values(self):

        figure = aplt.Figure()

        figure.open()
        plt.imshow(np.ones((2, 2)))
        cb = aplt.Colorbar(ticksize=1, fraction=1.0, pad=2.0)
        cb.set()
        figure.close()

        figure.open()
        plt.imshow(np.ones((2, 2)))
        cb = aplt.Colorbar(
            ticksize=1,
            fraction=0.1,
            pad=0.5,
            manual_tick_values=[0.25, 0.5, 0.75],
            manual_tick_labels=[1.0, 2.0, 3.0],
        )
        cb.set()
        figure.close()

        figure.open()
        plt.imshow(np.ones((2, 2)))
        cb = aplt.Colorbar(ticksize=1, fraction=0.1, pad=0.5)
        cb.set_with_values(
            cmap=aplt.Cmap().kwargs["cmap"], color_values=[1.0, 2.0, 3.0]
        )
        figure.close()


class TestTicks:
    def test__set__yticks__works_for_good_values(self):

        array = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

        units = aplt.Units(use_scaled=True, conversion_factor=None)

        yticks = aplt.YTicks(labelsize=34)

        extent = array.extent_of_zoomed_array(buffer=1)

        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=False,
        )
        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=True,
        )

        yticks = aplt.YTicks(labelsize=34)

        units = aplt.Units(use_scaled=False, conversion_factor=None)

        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=False,
        )
        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=True,
        )

        yticks = aplt.YTicks(labelsize=34)

        units = aplt.Units(use_scaled=True, conversion_factor=2.0)

        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=False,
        )
        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=True,
        )

        yticks = aplt.YTicks(labelsize=34)

        units = aplt.Units(use_scaled=False, conversion_factor=2.0)

        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=False,
        )
        yticks.set(
            array=array,
            min_value=extent[2],
            max_value=extent[3],
            units=units,
            use_defaults=True,
        )

    def test__set__xticks__works_for_good_values(self):

        array = aa.Array.ones(shape_2d=(2, 2), pixel_scales=1.0)

        units = aplt.Units(use_scaled=True, conversion_factor=None)

        xticks = aplt.XTicks(labelsize=34)

        extent = array.extent_of_zoomed_array(buffer=1)

        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=False,
        )
        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=True,
        )

        xticks = aplt.XTicks(labelsize=34)

        units = aplt.Units(use_scaled=False, conversion_factor=None)

        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=False,
        )
        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=True,
        )

        xticks = aplt.XTicks(labelsize=34)

        units = aplt.Units(use_scaled=True, conversion_factor=2.0)

        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=False,
        )
        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=True,
        )

        xticks = aplt.XTicks(labelsize=34)

        units = aplt.Units(use_scaled=False, conversion_factor=2.0)

        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=False,
        )
        xticks.set(
            array=array,
            min_value=extent[0],
            max_value=extent[1],
            units=units,
            use_defaults=True,
        )


class TestTitle:
    def test__title_from_func__uses_func_name_if_title_is_none(self):
        def toy_func():
            pass

        label = aplt.Title(label=None)

        title_from_func = label.title_from_func(func=toy_func)

        assert title_from_func == "Toy_func"

        label = aplt.Title(label="Hi")

        title_from_func = label.title_from_func(func=toy_func)

        assert title_from_func == "Hi"


class TestLabels:
    def test__y_units_use_plot_in_kpc_if_it_is_passed(self):

        ylabel = aplt.YLabel()

        units = aplt.Units(in_kpc=True)

        assert ylabel._units == None
        assert ylabel.label_from_units(units=units) == "kpc"

        ylabel = aplt.YLabel()

        units = aplt.Units(in_kpc=False)

        assert ylabel._units == None
        assert ylabel.label_from_units(units=units) == "arcsec"

        ylabel = aplt.YLabel(units="hi")

        units = aplt.Units(in_kpc=True)

        assert ylabel._units == "hi"
        assert ylabel.label_from_units(units=units) == "hi"

        ylabel = aplt.YLabel(units="hi")

        units = aplt.Units(in_kpc=False)

        assert ylabel._units == "hi"
        assert ylabel.label_from_units(units=units) == "hi"

    def test__x_units_use_plot_in_kpc_if_it_is_passed(self):

        xlabel = aplt.XLabel()

        units = aplt.Units(in_kpc=True)

        assert xlabel._units == None
        assert xlabel.label_from_units(units=units) == "kpc"

        xlabel = aplt.XLabel()

        units = aplt.Units(in_kpc=False)

        assert xlabel._units == None
        assert xlabel.label_from_units(units=units) == "arcsec"

        xlabel = aplt.XLabel(units="hi")

        units = aplt.Units(in_kpc=True)

        assert xlabel._units == "hi"
        assert xlabel.label_from_units(units=units) == "hi"

        xlabel = aplt.XLabel(units="hi")

        units = aplt.Units(in_kpc=False)

        assert xlabel._units == "hi"
        assert xlabel.label_from_units(units=units) == "hi"

    def test__y_units_from_func__uses_function_inputs_if_available(self):
        def toy_func():
            pass

        ylabel = aplt.YLabel(units=None)

        yunits_from_func = ylabel.units_from_func(func=toy_func)

        assert yunits_from_func == None

        def toy_func(label_yunits="Hi"):
            pass

        ylabel = aplt.YLabel()

        yunits_from_func = ylabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi"

        ylabel = aplt.YLabel(units="Hi1")

        yunits_from_func = ylabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"

        def toy_func(argument, label_yunits="Hi"):
            pass

        ylabel = aplt.YLabel()

        yunits_from_func = ylabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi"

        ylabel = aplt.YLabel(units="Hi1")

        yunits_from_func = ylabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"

    def test__x_units_from_func__uses_function_inputs_if_available(self):
        def toy_func():
            pass

        xlabel = aplt.XLabel(units=None)

        yunits_from_func = xlabel.units_from_func(func=toy_func)

        assert yunits_from_func == None

        def toy_func(label_yunits="Hi"):
            pass

        xlabel = aplt.XLabel()

        yunits_from_func = xlabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi"

        xlabel = aplt.XLabel(units="Hi1")

        yunits_from_func = xlabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"

        def toy_func(argument, label_yunits="Hi"):
            pass

        xlabel = aplt.XLabel()

        yunits_from_func = xlabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi"

        xlabel = aplt.XLabel(units="Hi1")

        yunits_from_func = xlabel.units_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"


class TestLegend:
    def test__set_legend_works_for_plot(self):

        figure = aplt.Figure(aspect="auto")

        figure.open()

        line = aplt.LinePlot(linewidth=2, linestyle="-", colors="k", pointsize=2)

        line.draw_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear", label="hi"
        )

        legend = aplt.Legend(include=True, fontsize=1)

        legend.set()

        figure.close()


class TestOutput:
    def test__input_path_is_created(self):

        test_path = path.join(directory, "files", "output_path")

        if path.exists(test_path):
            shutil.rmtree(test_path)

        assert not path.exists(test_path)

        output = aplt.Output(path=test_path)

        assert path.exists(test_path)

    def test__filename_from_func__returns_function_name_if_no_filename(self):
        def toy_func():
            pass

        output = aplt.Output(filename=None)

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "toy_func"

        output = aplt.Output(filename="Hi")

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "Hi"
