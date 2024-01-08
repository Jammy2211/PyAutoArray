import autoarray.plot as aplt


def test__set_legend_works_for_plot():
    figure = aplt.Figure(aspect="auto")

    figure.open()

    line = aplt.YXPlot(linewidth=2, linestyle="-", c="k")

    line.plot_y_vs_x(
        y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear", label="hi"
    )

    legend = aplt.Legend(fontsize=1)

    legend.set()

    figure.close()
