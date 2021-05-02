import autoarray.plot as aplt

from os import path

directory = path.dirname(path.realpath(__file__))


class TestLinePlot:
    def test___from_config_or_via_manual_input(self):

        line_plot = aplt.YXPlot()

        assert line_plot.config_dict["linewidth"] == 3
        assert line_plot.config_dict["c"] == "k"

        line_plot = aplt.YXPlot(c=["k", "b"])

        assert line_plot.config_dict["linewidth"] == 3
        assert line_plot.config_dict["c"] == ["k", "b"]

        line_plot = aplt.YXPlot()
        line_plot.is_for_subplot = True

        assert line_plot.config_dict["linewidth"] == 1
        assert line_plot.config_dict["c"] == "k"

        line_plot = aplt.YXPlot(linestyle=".")
        line_plot.is_for_subplot = True

        assert line_plot.config_dict["linewidth"] == 1
        assert line_plot.config_dict["c"] == "k"

    def test__plot_y_vs_x__works_for_reasonable_values(self):

        line = aplt.YXPlot(linewidth=2, linestyle="-", c="k")

        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear")
        line.plot_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy"
        )
        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog")

        line = aplt.YXPlot(c="k", s=2)

        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="scatter")


class TestAVXLine:
    def test___from_config_or_via_manual_input(self):

        line_plot = aplt.AXVLine()

        assert line_plot.config_dict["ymin"] == 0.5

        line_plot = aplt.AXVLine(ymin=0.7)

        assert line_plot.config_dict["ymin"] == 0.7

        line_plot = aplt.AXVLine()
        line_plot.is_for_subplot = True

        assert line_plot.config_dict["ymin"] == 0.6

        line_plot = aplt.AXVLine(ymin=0.1)
        line_plot.is_for_subplot = True

        assert line_plot.config_dict["ymin"] == 0.1

    def test__plot_vertical_lines__works_for_reasonable_values(self):

        line = aplt.AXVLine(linewidth=2, linestyle="-", c="k")

        line.axvline_vertical_line(vertical_line=0.0, label="hi")
