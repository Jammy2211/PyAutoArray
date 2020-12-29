import autoarray as aa
import autoarray.plot as aplt

from os import path

directory = path.dirname(path.realpath(__file__))


class TestLinePlot:
    def test___from_config_or_via_manual_input(self):

        line_plot = aplt.LinePlot()

        assert line_plot.config_dict["width"] == 3
        assert line_plot.colors == ["k", "w"]

        line_plot = aplt.LinePlot(colors=["k", "b"])

        assert line_plot.config_dict["width"] == 3
        assert line_plot.colors == ["k", "b"]

        line_plot = aplt.LinePlot()
        line_plot.for_subplot = True

        assert line_plot.config_dict["width"] == 1
        assert line_plot.colors == ["k"]

        line_plot = aplt.LinePlot(style=".")
        line_plot.for_subplot = True

        assert line_plot.config_dict["width"] == 1
        assert line_plot.colors == ["k"]

    def test__plot_y_vs_x__works_for_reasonable_values(self):

        line = aplt.LinePlot(linewidth=2, linestyle="-", colors="k")

        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear")
        line.plot_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy"
        )
        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog")

        line = aplt.LinePlot(colors="k", s=2)

        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="scatter")

    def test__plot_vertical_lines__works_for_reasonable_values(self):

        line = aplt.LinePlot(linewidth=2, linestyle="-", colors="k")

        line.plot_vertical_lines(vertical_lines=[[0.0]])
        line.plot_vertical_lines(vertical_lines=[[1.0], [2.0]])
        line.plot_vertical_lines(vertical_lines=[[0.0]], vertical_line_labels=["hi"])
        line.plot_vertical_lines(
            vertical_lines=[[1.0], [2.0]], vertical_line_labels=["hi1", "hi2"]
        )
