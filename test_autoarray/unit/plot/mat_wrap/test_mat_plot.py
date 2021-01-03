import autoarray.plot as aplt
from os import path
import matplotlib.pyplot as plt
import pytest
import shutil

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plotter"
    )


class TestAbstractPlotter:
    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = aplt.MatPlot2D()
        plotter = plotter.mat_plot_for_subplot_from()

        figsize = plotter.get_subplot_figsize(number_subplots=1)

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (13, 10)

        plotter = aplt.MatPlot2D(figure=aplt.Figure(figsize=(20, 20)))

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = aplt.MatPlot2D()

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=1)

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.get_subplot_rows_columns(number_subplots=4)

        assert rows == 2
        assert columns == 2

    def test__open_and_close_subplot_figures(self):

        plotter = aplt.MatPlot2D()
        plotter.figure.open()

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False

        plotter = aplt.MatPlot2D()

        assert plt.fignum_exists(num=1) == False

        plotter.open_subplot_figure(number_subplots=4)

        assert plt.fignum_exists(num=1) == True

        plotter.figure.close()

        assert plt.fignum_exists(num=1) == False

    def test__mat_plot_with_new_labels__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.MatPlot2D(
            title=aplt.Title(label="OMG", fontsize=1),
            ylabel=aplt.YLabel(units="hi"),
            xlabel=aplt.XLabel(units="hi2"),
            tickparams=aplt.TickParams(labelsize=2),
        )

        print(plotter.title.config_dict)

        plotter = plotter.mat_plot_with_new_labels()

        assert plotter.title.config_dict["label"] == "OMG"
        assert plotter.title.config_dict["fontsize"] == 1
        assert plotter.ylabel._units == "hi"
        assert plotter.xlabel._units == "hi2"
        assert plotter.tickparams.config_dict["labelsize"] == 2

        plotter = plotter.mat_plot_with_new_labels(
            title_label="OMG0",
            title_fontsize=10,
            ylabel_units="hi0",
            xlabel_units="hi20",
            tick_params_labelsize=20,
        )

        assert plotter.title.config_dict["label"] == "OMG0"
        assert plotter.title.config_dict["fontsize"] == 10
        assert plotter.ylabel._units == "hi0"
        assert plotter.xlabel._units == "hi20"
        assert plotter.tickparams.config_dict["labelsize"] == 20

        plotter = plotter.mat_plot_with_new_labels(title_fontsize=2, title_label="OMG0")

        assert plotter.title.config_dict["label"] == "OMG0"
        assert plotter.title.config_dict["fontsize"] == 2
        assert plotter.ylabel._units == "hi0"
        assert plotter.xlabel._units == "hi20"
        assert plotter.tickparams.config_dict["labelsize"] == 20

    def test__mat_plot_with_new_cmap__new_labels_if_input__sizes_dont_change(self):

        plotter = aplt.MatPlot2D(
            cmap=aplt.Cmap(
                cmap="cold", norm="log", vmin=0.1, vmax=1.0, linthresh=1.5, linscale=2.0
            )
        )

        assert plotter.cmap.config_dict["cmap"] == "cold"
        assert plotter.cmap.config_dict["norm"] == "log"
        assert plotter.cmap.config_dict["vmin"] == 0.1
        assert plotter.cmap.config_dict["vmax"] == 1.0
        assert plotter.cmap.config_dict["linthresh"] == 1.5
        assert plotter.cmap.config_dict["linscale"] == 2.0

        plotter = plotter.mat_plot_with_new_cmap(
            cmap="jet", norm="linear", vmin=0.12, vmax=1.2, linthresh=1.2, linscale=2.2
        )

        assert plotter.cmap.config_dict["cmap"] == "jet"
        assert plotter.cmap.config_dict["norm"] == "linear"
        assert plotter.cmap.config_dict["vmin"] == 0.12
        assert plotter.cmap.config_dict["vmax"] == 1.2
        assert plotter.cmap.config_dict["linthresh"] == 1.2
        assert plotter.cmap.config_dict["linscale"] == 2.2

        plotter = plotter.mat_plot_with_new_cmap(cmap="sand", norm="log", vmin=0.13)

        assert plotter.cmap.config_dict["cmap"] == "sand"
        assert plotter.cmap.config_dict["norm"] == "log"
        assert plotter.cmap.config_dict["vmin"] == 0.13
        assert plotter.cmap.config_dict["vmax"] == 1.2
        assert plotter.cmap.config_dict["linthresh"] == 1.2
        assert plotter.cmap.config_dict["linscale"] == 2.2

    def test__mat_plot_with_new_outputs__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.MatPlot2D(
            output=aplt.Output(path="Path", format="png", filename="file")
        )

        plotter = plotter.mat_plot_with_new_output()

        assert plotter.output.path == "Path"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.mat_plot_with_new_output(path="Path0", filename="file0")

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        assert plotter.output.path == "Path0"
        assert plotter.output._format == "png"
        assert plotter.output.format == "png"
        assert plotter.output.filename == "file0"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

        plotter = plotter.mat_plot_with_new_output(
            path="Path1", filename="file1", format="fits"
        )

        assert plotter.output.path == "Path1"
        assert plotter.output._format == "fits"
        assert plotter.output.format == "fits"
        assert plotter.output.filename == "file1"

        if path.exists(plotter.output.path):
            shutil.rmtree(plotter.output.path)

    def test__mat_plot_with_new_units__new_outputs_are_setup_correctly_if_input(self):

        plotter = aplt.MatPlot2D(
            units=aplt.Units(use_scaled=True, in_kpc=True, conversion_factor=1.0)
        )

        assert plotter.units.use_scaled == True
        assert plotter.units.in_kpc == True
        assert plotter.units.conversion_factor == 1.0

        plotter = plotter.mat_plot_with_new_units(
            use_scaled=False, in_kpc=False, conversion_factor=2.0
        )

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 2.0

        plotter = plotter.mat_plot_with_new_units(conversion_factor=3.0)

        assert plotter.units.use_scaled == False
        assert plotter.units.in_kpc == False
        assert plotter.units.conversion_factor == 3.0


class TestPlotter2D:
    def test__uses_figure_or_subplot_configs_correctly(self):

        figure = aplt.Figure(figsize=(8, 8))
        cmap = aplt.Cmap(cmap="warm")

        plotter = aplt.MatPlot2D(figure=figure, cmap=cmap)

        assert plotter.figure.config_dict_figure["figsize"] == (8, 8)
        assert plotter.figure.config_dict_imshow["aspect"] == "square"
        assert plotter.cmap.config_dict["cmap"] == "warm"
        assert plotter.cmap.config_dict["norm"] == "linear"

        figure = aplt.Figure()
        cmap = aplt.Cmap()

        plotter = aplt.MatPlot2D(figure=figure, cmap=cmap)

        sub_plotter = plotter.mat_plot_for_subplot_from()

        assert sub_plotter.figure.config_dict_figure["figsize"] == None
        assert sub_plotter.figure.config_dict_imshow["aspect"] == "square"
        assert sub_plotter.cmap.config_dict["cmap"] == "jet"
        assert sub_plotter.cmap.config_dict["norm"] == "linear"
