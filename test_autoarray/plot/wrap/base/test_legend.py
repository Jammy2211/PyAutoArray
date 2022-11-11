import autoarray.plot as aplt


def test__legend__from_config_or_via_manual_input():

    legend = aplt.Legend()

    assert legend.include is True
    assert legend.config_dict["fontsize"] == 12

    legend = aplt.Legend(include=False, fontsize=11)

    assert legend.include is False
    assert legend.config_dict["fontsize"] == 11

    legend = aplt.Legend()
    legend.is_for_subplot = True

    assert legend.include is True
    assert legend.config_dict["fontsize"] == 13

    legend = aplt.Legend(include=False, fontsize=14)
    legend.is_for_subplot = True

    assert legend.include is False
    assert legend.config_dict["fontsize"] == 14

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
