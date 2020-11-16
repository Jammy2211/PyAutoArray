import autoarray.plot as aplt
import numpy as np

aplt.Line(
    y=np.array([1.0, 2.0, 3.0]),
    x=np.array([0.5, 1.0, 1.5]),
    plot_axis_type="loglog",
    vertical_lines=[1.0, 2.0],
    label="line0",
    vertical_line_labels=["line1", "line2"],
)
