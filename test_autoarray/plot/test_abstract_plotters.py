from os import path
import pytest
import matplotlib.pyplot as plt

import autoarray as aa
import autoarray.plot as aplt
from autoarray.plot import abstract_plotters

directory = path.dirname(path.realpath(__file__))


def test__get_subplot_shape():
    plotter = abstract_plotters.AbstractPlotter(mat_plot_2d=aplt.MatPlot2D())

    subplot_shape = plotter.mat_plot_2d.get_subplot_shape(number_subplots=1)

    assert subplot_shape == (1, 1)

    subplot_shape = plotter.mat_plot_2d.get_subplot_shape(number_subplots=3)

    assert subplot_shape == (2, 2)

    with pytest.raises(aa.exc.PlottingException):
        plotter.mat_plot_2d.get_subplot_shape(number_subplots=1000)


# def test__get_subplot_figsize():
#     plotter = abstract_plotters.AbstractPlotter(
#         mat_plot_2d=aplt.MatPlot2D(figure=aplt.Figure(figsize="auto"))
#     )
#
#     figsize = plotter.get_subplot_figsize(number_subplots=1)
#
#     assert figsize == (7, 7)
#
#     figsize = plotter.get_subplot_figsize(number_subplots=4)
#
#     assert figsize == (7, 7)
#
#     figure = aplt.Figure(figsize=(20, 20))
#
#     plotter = abstract_plotters.AbstractPlotter(
#         mat_plot_2d=aplt.MatPlot2D(figure=figure)
#     )
#
#     figsize = plotter.get_subplot_figsize(number_subplots=4)
#
#     assert figsize == (20, 20)


def test__open_and_close_subplot_figures():
    figure = aplt.Figure(figsize=(20, 20))

    plotter = abstract_plotters.AbstractPlotter(
        mat_plot_2d=aplt.MatPlot2D(figure=figure)
    )

    plotter.mat_plot_2d.figure.open()

    assert plt.fignum_exists(num=1) is True

    plotter.mat_plot_2d.figure.close()

    assert plt.fignum_exists(num=1) is False

    plotter = abstract_plotters.AbstractPlotter(
        mat_plot_2d=aplt.MatPlot2D(figure=figure)
    )

    assert plt.fignum_exists(num=1) is False

    plotter.open_subplot_figure(number_subplots=4)

    assert plt.fignum_exists(num=1) is True

    plotter.mat_plot_2d.figure.close()

    assert plt.fignum_exists(num=1) is False


def test__uses_figure_or_subplot_configs_correctly():
    figure = aplt.Figure(figsize=(8, 8))
    cmap = aplt.Cmap(cmap="warm")

    mat_plot_2d = aplt.MatPlot2D(figure=figure, cmap=cmap)

    plotter = abstract_plotters.AbstractPlotter(mat_plot_2d=mat_plot_2d)

    assert plotter.mat_plot_2d.figure.config_dict["figsize"] == (8, 8)
    assert plotter.mat_plot_2d.figure.config_dict["aspect"] == "square"
    assert plotter.mat_plot_2d.cmap.config_dict["cmap"] == "warm"
    assert plotter.mat_plot_2d.cmap.config_dict["norm"] == "linear"

    figure = aplt.Figure()
    figure.is_for_subplot = True

    cmap = aplt.Cmap()
    cmap.is_for_subplot = True

    mat_plot_2d = aplt.MatPlot2D(figure=figure, cmap=cmap)

    plotter = abstract_plotters.AbstractPlotter(mat_plot_2d=mat_plot_2d)

    assert plotter.mat_plot_2d.figure.config_dict["figsize"] == None
    assert plotter.mat_plot_2d.figure.config_dict["aspect"] == "square"
    assert plotter.mat_plot_2d.cmap.config_dict["cmap"] == "default"
    assert plotter.mat_plot_2d.cmap.config_dict["norm"] == "linear"


def test__get__visuals():
    visuals_2d = aplt.Visuals2D()

    plotter = abstract_plotters.Plotter(visuals_2d=visuals_2d)
    attr = plotter.get_2d.get(name="origin", value=1)

    assert attr == 1

    visuals_2d = aplt.Visuals2D(origin=10)

    plotter = abstract_plotters.Plotter(visuals_2d=visuals_2d)
    attr = plotter.get_2d.get(name="origin", value=2)

    assert attr == 10
