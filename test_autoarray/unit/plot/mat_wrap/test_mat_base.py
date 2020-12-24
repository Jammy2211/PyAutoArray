import autoarray as aa
import autoarray.plot as aplt

from os import path

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import shutil
import numpy as np

directory = path.dirname(path.realpath(__file__))


class TestUnits:
    def test__loads_values_from_config_if_not_manually_input(self):

        units = aplt.Units()

        assert units.use_scaled == True
        assert units.in_kpc == False
        assert units.conversion_factor == None

        units = aplt.Units(in_kpc=True, conversion_factor=2.0)

        assert units.use_scaled == True
        assert units.in_kpc == True
        assert units.conversion_factor == 2.0


class TestFigure:
    def test__loads_values_from_config_if_not_manually_input(self):

        figure = aplt.Figure()

        assert figure.kwargs["figsize"] == (7, 7)
        assert figure.kwargs["aspect"] == "square"

        figure = aplt.Figure(aspect="auto")

        assert figure.kwargs["figsize"] == (7, 7)
        assert figure.kwargs["aspect"] == "auto"

        figure = aplt.Figure(use_subplot_defaults=True)

        assert figure.kwargs["figsize"] == None
        assert figure.kwargs["aspect"] == "square"

        figure = aplt.Figure(use_subplot_defaults=True, figsize=(6, 6))

        assert figure.kwargs["figsize"] == (6, 6)
        assert figure.kwargs["aspect"] == "square"

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
    def test__loads_values_from_config_if_not_manually_input(self):

        cmap = aplt.Cmap()

        assert cmap.kwargs["cmap"] == "jet"
        assert cmap.kwargs["norm"] == "linear"

        cmap = aplt.Cmap(cmap="cold")

        assert cmap.kwargs["cmap"] == "cold"
        assert cmap.kwargs["norm"] == "linear"

        cmap = aplt.Cmap(use_subplot_defaults=True)

        assert cmap.kwargs["cmap"] == "jet"
        assert cmap.kwargs["norm"] == "linear"

        cmap = aplt.Cmap(use_subplot_defaults=True, cmap="cold")

        assert cmap.kwargs["cmap"] == "cold"
        assert cmap.kwargs["norm"] == "linear"

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
    def test__loads_values_from_config_if_not_manually_input(self):

        colorbar = aplt.Colorbar()

        assert colorbar.kwargs["labelsize"] == 1
        assert colorbar.manual_tick_values == None
        assert colorbar.manual_tick_labels == None

        colorbar = aplt.Colorbar(
            labelsize=20, manual_tick_values=(1.0, 2.0), manual_tick_labels=(3.0, 4.0)
        )

        assert colorbar.kwargs["labelsize"] == 20
        assert colorbar.manual_tick_values == (1.0, 2.0)
        assert colorbar.manual_tick_labels == (3.0, 4.0)

        colorbar = aplt.Colorbar(use_subplot_defaults=True)

        assert colorbar.kwargs["labelsize"] == 1

        colorbar = aplt.Colorbar(use_subplot_defaults=True, labelsize=10)

        assert colorbar.kwargs["labelsize"] == 10

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
        cb.set_with_color_values(
            cmap=aplt.Cmap().kwargs["cmap"], color_values=[1.0, 2.0, 3.0]
        )
        figure.close()


class TestTicksParams:
    def test__loads_values_from_config_if_not_manually_input(self):
        tick_params = aplt.TickParams()

        assert tick_params.kwargs["labelsize"] == 16

        tick_params = aplt.TickParams(labelsize=24)
        assert tick_params.kwargs["labelsize"] == 24

        tick_params = aplt.TickParams(use_subplot_defaults=True)

        assert tick_params.kwargs["labelsize"] == 10

        tick_params = aplt.TickParams(use_subplot_defaults=True, labelsize=25)

        assert tick_params.kwargs["labelsize"] == 25


class TestYTicks:
    def test__ticks_loads_values_from_config_if_not_manually_input(self):

        yticks = aplt.YTicks()

        assert yticks.kwargs["labelsize"] == 16
        assert yticks.manual_values == None
        assert yticks.manual_values == None

        yticks = aplt.YTicks(labelsize=24, manual_values=[1.0, 2.0])

        assert yticks.kwargs["labelsize"] == 24
        assert yticks.manual_values == [1.0, 2.0]

        yticks = aplt.YTicks(use_subplot_defaults=True)

        assert yticks.kwargs["labelsize"] == 10
        assert yticks.manual_values == None

        yticks = aplt.YTicks(
            use_subplot_defaults=True, labelsize=25, manual_values=[1.0, 2.0]
        )

        assert yticks.kwargs["labelsize"] == 25
        assert yticks.manual_values == [1.0, 2.0]

    def test__set__works_for_good_values(self):

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


class TestXTicks:
    def test__ticks_loads_values_from_config_if_not_manually_input(self):
        xticks = aplt.XTicks()

        assert xticks.kwargs["labelsize"] == 17
        assert xticks.manual_values == None
        assert xticks.manual_values == None

        xticks = aplt.XTicks(labelsize=24, manual_values=[1.0, 2.0])

        assert xticks.kwargs["labelsize"] == 24
        assert xticks.manual_values == [1.0, 2.0]

        xticks = aplt.XTicks(use_subplot_defaults=True)

        assert xticks.kwargs["labelsize"] == 11
        assert xticks.manual_values == None

        xticks = aplt.XTicks(
            use_subplot_defaults=True, labelsize=25, manual_values=[1.0, 2.0]
        )

        assert xticks.kwargs["labelsize"] == 25
        assert xticks.manual_values == [1.0, 2.0]

    def test__set__works_for_good_values(self):
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
    def test__loads_values_from_config_if_not_manually_input(self):

        title = aplt.Title()

        assert title.kwargs["label"] == None
        assert title.kwargs["fontsize"] == 11

        title = aplt.Title(label="OMG", fontsize=1)

        assert title.kwargs["label"] == "OMG"
        assert title.kwargs["fontsize"] == 1

        title = aplt.Title(use_subplot_defaults=True)

        assert title.kwargs["label"] == None
        assert title.kwargs["fontsize"] == 15

        title = aplt.Title(use_subplot_defaults=True, label="OMG2", fontsize=2)

        assert title.kwargs["label"] == "OMG2"
        assert title.kwargs["fontsize"] == 2

    def test__title_from_func__uses_func_name_if_title_is_none(self):
        def toy_func():
            pass

        label = aplt.Title(label=None)

        title_from_func = label.title_from_func(func=toy_func)

        assert title_from_func == "Toy_func"

        label = aplt.Title(label="Hi")

        title_from_func = label.title_from_func(func=toy_func)

        assert title_from_func == "Hi"


class TestYLabel:
    def test__loads_values_from_config_if_not_manually_input(self):

        ylabel = aplt.YLabel()

        assert ylabel._units == None
        assert ylabel.kwargs["fontsize"] == 1

        ylabel = aplt.YLabel(units="hi", fontsize=11)

        assert ylabel._units == "hi"
        assert ylabel.kwargs["fontsize"] == 11

        ylabel = aplt.YLabel(use_subplot_defaults=True)

        assert ylabel._units == None
        assert ylabel.kwargs["fontsize"] == 2

        ylabel = aplt.YLabel(use_subplot_defaults=True, units="hi2", fontsize=12)

        assert ylabel._units == "hi2"
        assert ylabel.kwargs["fontsize"] == 12

    def test__units_use_plot_in_kpc_if_it_is_passed(self):

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

    def test__units_from_func__uses_function_inputs_if_available(self):
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


class TestXLabel:
    def test__loads_values_from_config_if_not_manually_input(self):
        xlabel = aplt.XLabel()

        assert xlabel._units == None
        assert xlabel.kwargs["fontsize"] == 3

        xlabel = aplt.XLabel(units="hi", fontsize=11)

        assert xlabel._units == "hi"
        assert xlabel.kwargs["fontsize"] == 11

        xlabel = aplt.XLabel(use_subplot_defaults=True)

        assert xlabel._units == None
        assert xlabel.kwargs["fontsize"] == 4

        xlabel = aplt.XLabel(use_subplot_defaults=True, units="hi2", fontsize=12)

        assert xlabel._units == "hi2"
        assert xlabel.kwargs["fontsize"] == 12

    def test__units_use_plot_in_kpc_if_it_is_passed(self):
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

    def test__units_from_func__uses_function_inputs_if_available(self):
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
    def test__legend__from_config_or_via_manual_input(self):

        legend = aplt.Legend()

        assert legend.include == False
        assert legend.kwargs["fontsize"] == 12

        legend = aplt.Legend(include=True, fontsize=11)

        assert legend.include == True
        assert legend.kwargs["fontsize"] == 11

        legend = aplt.Legend(use_subplot_defaults=True)

        assert legend.include == False
        assert legend.kwargs["fontsize"] == 13

        legend = aplt.Legend(use_subplot_defaults=True, include=True, fontsize=14)

        assert legend.include == True
        assert legend.kwargs["fontsize"] == 14

    def test__set_legend_works_for_plot(self):

        figure = aplt.Figure(aspect="auto")

        figure.open()

        line = aplt.LinePlot(linewidth=2, linestyle="-", colors="k", pointsize=2)

        line.plot_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear", label="hi"
        )

        legend = aplt.Legend(include=True, fontsize=1)

        legend.set()

        figure.close()


class TestOutput:
    def test__constructor(self):

        output = aplt.Output()

        assert output.path == None
        assert output._format == None
        assert output.format == "show"
        assert output.filename == None

        output = aplt.Output(path="Path", format="png", filename="file")

        assert output.path == "Path"
        assert output._format == "png"
        assert output.format == "png"
        assert output.filename == "file"

        if path.exists(output.path):
            shutil.rmtree(output.path)

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
