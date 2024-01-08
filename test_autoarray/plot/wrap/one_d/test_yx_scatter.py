import autoarray.plot as aplt


def test__scatter_y_vs_x__works_for_reasonable_values():
    yx_scatter = aplt.YXScatter(linewidth=2, linestyle="-", c="k")

    yx_scatter.scatter_yx(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0])
