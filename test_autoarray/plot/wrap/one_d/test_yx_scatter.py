import autoarray.plot as aplt


def test___from_config_or_via_manual_input():
    line_plot = aplt.YXScatter()

    assert line_plot.config_dict["marker"] == "."
    assert line_plot.config_dict["c"] == "k"

    line_plot = aplt.YXScatter(c=["k", "b"])

    assert line_plot.config_dict["marker"] == "."
    assert line_plot.config_dict["c"] == ["k", "b"]

    line_plot = aplt.YXScatter()
    line_plot.is_for_subplot = True

    assert line_plot.config_dict["marker"] == "x"
    assert line_plot.config_dict["c"] == "k"

    line_plot = aplt.YXScatter(marker=".")
    line_plot.is_for_subplot = True

    assert line_plot.config_dict["marker"] == "."
    assert line_plot.config_dict["c"] == "k"


def test__scatter_y_vs_x__works_for_reasonable_values():
    yx_scatter = aplt.YXScatter(linewidth=2, linestyle="-", c="k")

    yx_scatter.scatter_yx(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0])
