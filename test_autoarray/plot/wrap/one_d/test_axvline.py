import autoarray.plot as aplt


def test___from_config_or_via_manual_input():

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


def test__plot_vertical_lines__works_for_reasonable_values():

    line = aplt.AXVLine(linewidth=2, linestyle="-", c="k")

    line.axvline_vertical_line(vertical_line=0.0, label="hi")
    line.axvline_vertical_line(
        vertical_line=0.0, vertical_errors=[-1.0, 1.0], label="hi"
    )
