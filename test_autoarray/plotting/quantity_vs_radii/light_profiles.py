import autoarray as aa
import numpy as np

y = np.linspace(0.0, 10.0, 10)
x = np.linspace(0.0, 10.0, 10)

aa.plot.line_yx_plotters.plot_line(y=y, x=x)
