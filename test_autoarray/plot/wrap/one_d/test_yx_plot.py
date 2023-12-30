import autoarray.plot as aplt


def test__plot_y_vs_x__works_for_reasonable_values():
    line = aplt.YXPlot(linewidth=2, linestyle="-", c="k")

    line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear")
    line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy")
    line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog")

    line = aplt.YXPlot(c="k")

    line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="scatter")

    line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="errorbar")

    line.plot_y_vs_x(
        y=[1.0, 2.0, 3.0],
        x=[1.0, 2.0, 3.0],
        plot_axis_type="errorbar",
        y_errors=[1.0, 1.0, 1.0],
    )

    line.plot_y_vs_x(
        y=[1.0, 2.0, 3.0],
        x=[1.0, 2.0, 3.0],
        plot_axis_type="errorbar",
        y_errors=[1.0, 1.0, 1.0],
        x_errors=[1.0, 1.0, 1.0],
    )
