import autoarray.plot as aplt


def test___from_config_or_via_manual_input():
    line_plot = aplt.FillBetween()

    assert line_plot.config_dict["alpha"] == 0.6
    assert line_plot.config_dict["color"] == "k"

    line_plot = aplt.FillBetween(color="b")

    assert line_plot.config_dict["alpha"] == 0.6
    assert line_plot.config_dict["color"] == "b"

    line_plot = aplt.FillBetween()
    line_plot.is_for_subplot = True

    assert line_plot.config_dict["alpha"] == 0.5
    assert line_plot.config_dict["color"] == "k"

    line_plot = aplt.FillBetween(alpha=0.4)
    line_plot.is_for_subplot = True

    assert line_plot.config_dict["alpha"] == 0.4
    assert line_plot.config_dict["color"] == "k"


def test__plot_y_vs_x__works_for_reasonable_values():
    fill_between = aplt.FillBetween()

    fill_between.fill_between_shaded_regions(
        x=[1, 2, 3], y1=[1.0, 2.0, 3.0], y2=[2.0, 3.0, 4.0]
    )
