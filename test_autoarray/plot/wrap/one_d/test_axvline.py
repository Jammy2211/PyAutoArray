import autoarray.plot as aplt


def test__plot_vertical_lines__works_for_reasonable_values():
    line = aplt.AXVLine(linewidth=2, linestyle="-", c="k")

    line.axvline_vertical_line(vertical_line=0.0, label="hi")
    line.axvline_vertical_line(
        vertical_line=0.0, vertical_errors=[-1.0, 1.0], label="hi"
    )
