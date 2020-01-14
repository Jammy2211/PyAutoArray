from os import path
import autoarray as aa
from autoarray import conf
from autoarray.plotters import mat_objs

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pytest
import os, shutil
import numpy as np

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


class TestFigure:

    def test__aspect_from_shape_2d(self):

        figure = mat_objs.Figure(aspect="auto")

        aspect = figure.aspect_from_shape_2d(shape_2d=(2,2))

        assert aspect == "auto"

        figure = mat_objs.Figure(aspect="square")

        aspect = figure.aspect_from_shape_2d(shape_2d=(2, 2))

        assert aspect == 1.0

        aspect = figure.aspect_from_shape_2d(shape_2d=(4, 2))

        assert aspect == 0.5

    def test__open_and_close__open_and_close_figures_correct(self):

        figure = mat_objs.Figure()

        assert plt.fignum_exists(num=1) == False

        figure.open()

        assert plt.fignum_exists(num=1) == True

        figure.close()

        assert plt.fignum_exists(num=1) == False


class TestColorMap:

    def test__norm_from_array__uses_input_norm_min_and_max_if_input(self):

        cmap = mat_objs.ColorMap(norm_min=0.0, norm_max=1.0, norm="linear")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.Normalize)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

        cmap = mat_objs.ColorMap(norm_min=0.0, norm_max=1.0, norm="log")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.LogNorm)
        assert norm.vmin == 1.0e-4 # Increased from 0.0 to ensure min isn't inf
        assert norm.vmax == 1.0

        cmap = mat_objs.ColorMap(norm_min=0.0, norm_max=1.0, linthresh=2.0, linscale=3.0, norm="symmetric_log")

        norm = cmap.norm_from_array(array=None)

        assert isinstance(norm, colors.SymLogNorm)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0
        assert norm.linthresh == 2.0

    def test__norm_from_array__uses_array_to_get_norm_min_and_max_if_no_manual_input(self):

        array = aa.array.ones(shape_2d=(2,2))
        array[0] = 0.0

        cmap = mat_objs.ColorMap(norm_min=None, norm_max=None, norm="linear")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.Normalize)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0

        cmap = mat_objs.ColorMap(norm_min=None, norm_max=None, norm="log")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.LogNorm)
        assert norm.vmin == 1.0e-4 # Increased from 0.0 to ensure min isn't inf
        assert norm.vmax == 1.0

        cmap = mat_objs.ColorMap(norm_min=None, norm_max=None, linthresh=2.0, linscale=3.0, norm="symmetric_log")

        norm = cmap.norm_from_array(array=array)

        assert isinstance(norm, colors.SymLogNorm)
        assert norm.vmin == 0.0
        assert norm.vmax == 1.0
        assert norm.linthresh == 2.0


class TestColorBar:

    def test__plot__works_for_reasonable_range_of_values(self):

        figure = mat_objs.Figure()

        figure.open()
        plt.imshow(np.ones((2,2)))
        cb = mat_objs.ColorBar(ticksize=1, fraction=1.0, pad=2.0)
        cb.plot()
        figure.close()

        figure.open()
        plt.imshow(np.ones((2,2)))
        cb = mat_objs.ColorBar(ticksize=1, fraction=0.1, pad=0.5, tick_values=[0.25, 0.5, 0.75],
                               tick_labels=[1.0, 2.0, 3.0])
        cb.plot()
        figure.close()

        figure.open()
        plt.imshow(np.ones((2,2)))
        cb = mat_objs.ColorBar(ticksize=1, fraction=0.1, pad=0.5)
        cb.plot_with_values(cmap=mat_objs.ColorMap().cmap, color_values=[1.0, 2.0, 3.0])
        figure.close()


class TestTicks:

    def test__set_yx_ticks__works_for_good_values(self):

        array = aa.array.ones(shape_2d=(2,2), pixel_scales=1.0)

        ticks = mat_objs.Ticks(
            ysize=34,
            xsize=35,
            units=mat_objs.Units(use_scaled=True, conversion_factor=None)
        )

        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)

        ticks = mat_objs.Ticks(
            ysize=34,
            xsize=35,
            units=mat_objs.Units(use_scaled=False, conversion_factor=None)
        )

        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)

        ticks = mat_objs.Ticks(
            ysize=34,
            xsize=35,
            units=mat_objs.Units(use_scaled=True, conversion_factor=2.0))

        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)

        ticks = mat_objs.Ticks(
            ysize=34,
            xsize=35,
            units=mat_objs.Units(use_scaled=False, conversion_factor=2.0))

        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=False)
        ticks.set_yticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)
        ticks.set_xticks(array=array, extent=array.extent_of_zoomed_array(buffer=1), symmetric_around_centre=True)

class TestLabels:

    def test__yx_units_use_plot_in_kpc_if_it_is_passed(self):

        labels = mat_objs.Labels(units=mat_objs.Units(in_kpc=True))

        assert labels.units.in_kpc == True
        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits == "kpc"
        assert labels.xunits == "kpc"

        labels = mat_objs.Labels(units=mat_objs.Units(in_kpc=False))

        assert labels.units.in_kpc == False
        assert labels._yunits == None
        assert labels._xunits == None
        assert labels.yunits == "arcsec"
        assert labels.xunits == "arcsec"

        labels = mat_objs.Labels(yunits="hi", xunits="hi2", units=mat_objs.Units(in_kpc=True))

        assert labels.units.in_kpc == True
        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits == "hi"
        assert labels.xunits == "hi2"

        labels = mat_objs.Labels(yunits="hi", xunits="hi2", units=mat_objs.Units(in_kpc=False))

        assert labels.units.in_kpc == False
        assert labels._yunits == "hi"
        assert labels._xunits == "hi2"
        assert labels.yunits == "hi"
        assert labels.xunits == "hi2"

    def test__title_from_func__uses_func_name_if_title_is_none(self):
        def toy_func():
            pass

        labels = mat_objs.Labels(title=None)

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Toy_func"

        labels = mat_objs.Labels(title="Hi")

        title_from_func = labels.title_from_func(func=toy_func)

        assert title_from_func == "Hi"

    def test__yx_units_from_func__uses_function_inputs_if_available(self):
        def toy_func():
            pass

        labels = mat_objs.Labels(yunits=None, xunits=None)

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == None
        assert xunits_from_func == None

        def toy_func(label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = mat_objs.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = mat_objs.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"

        def toy_func(argument, label_yunits="Hi", label_xunits="Hi0"):
            pass

        labels = mat_objs.Labels()

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi"
        assert xunits_from_func == "Hi0"

        labels = mat_objs.Labels(yunits="Hi1", xunits="Hi2")

        yunits_from_func = labels.yunits_from_func(func=toy_func)
        xunits_from_func = labels.xunits_from_func(func=toy_func)

        assert yunits_from_func == "Hi1"
        assert xunits_from_func == "Hi2"


class TestOutput:
    def test__input_path_is_created(self):

        test_path = path.join(directory, "../test_files/output_path")

        if os.path.exists(test_path):
            shutil.rmtree(test_path)

        assert not os.path.exists(test_path)

        output = mat_objs.Output(path=test_path)

        assert os.path.exists(test_path)

    def test__filename_from_func__returns_function_name_if_no_filename(self):
        def toy_func():
            pass

        output = mat_objs.Output(filename=None)

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "toy_func"

        output = mat_objs.Output(filename="Hi")

        filename_from_func = output.filename_from_func(func=toy_func)

        assert filename_from_func == "Hi"
