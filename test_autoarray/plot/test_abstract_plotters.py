from os import path
import pytest
import matplotlib.pyplot as plt
import autoarray.plot as aplt
from autoarray.plot import abstract_plotters

directory = path.dirname(path.realpath(__file__))


class TestAbstractPlotter:
    def test__subplot_figsize_for_number_of_subplots(self):

        plotter = abstract_plotters.AbstractPlotter()

        figsize = plotter.get_subplot_figsize(number_subplots=1)

        assert figsize == (18, 8)

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (13, 10)

        figure = aplt.Figure(figsize=(20, 20))

        plotter = abstract_plotters.AbstractPlotter(
            mat_plot_2d=aplt.MatPlot2D(figure=figure)
        )

        figsize = plotter.get_subplot_figsize(number_subplots=4)

        assert figsize == (20, 20)

    def test__plotter_number_of_subplots(self):

        plotter = abstract_plotters.AbstractPlotter(mat_plot_2d=aplt.MatPlot2D())

        rows, columns = plotter.mat_plot_2d.get_subplot_rows_columns(number_subplots=1)

        assert rows == 1
        assert columns == 2

        rows, columns = plotter.mat_plot_2d.get_subplot_rows_columns(number_subplots=4)

        assert rows == 2
        assert columns == 2

    def test__open_and_close_subplot_figures(self):

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

    def test__uses_figure_or_subplot_configs_correctly(self):

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
        assert plotter.mat_plot_2d.cmap.config_dict["cmap"] == "jet"
        assert plotter.mat_plot_2d.cmap.config_dict["norm"] == "linear"

    def test__attribute_for_visuals(self):

        visuals_2d = aplt.Visuals2D()
        include_2d = aplt.Include2D(origin=False)

        abstract_plotter = abstract_plotters.AbstractPlotter(
            visuals_2d=visuals_2d, include_2d=include_2d
        )
        attr = abstract_plotter.extractor_2d.extract(name="origin", value=1)

        assert attr == None

        include_2d = aplt.Include2D(origin=True)
        abstract_plotter = abstract_plotters.AbstractPlotter(
            visuals_2d=visuals_2d, include_2d=include_2d
        )
        attr = abstract_plotter.extractor_2d.extract(name="origin", value=1)

        assert attr == 1

        visuals_2d = aplt.Visuals2D(origin=10)

        include_2d = aplt.Include2D(origin=False)
        abstract_plotter = abstract_plotters.AbstractPlotter(
            visuals_2d=visuals_2d, include_2d=include_2d
        )
        attr = abstract_plotter.extractor_2d.extract(name="origin", value=2)

        assert attr == 10

        include_2d = aplt.Include2D(origin=True)
        abstract_plotter = abstract_plotters.AbstractPlotter(
            visuals_2d=visuals_2d, include_2d=include_2d
        )
        attr = abstract_plotter.extractor_2d.extract(name="origin", value=2)

        assert attr == 10


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )
