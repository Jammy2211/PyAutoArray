import autoarray.plot as aplt
import numpy as np

aplt.line(
    y=np.array([1.0, 2.0, 3.0]), x=np.array([0.5, 1.0, 1.5]), plot_axis_type="linear"
)
aplt.line(
    y=np.array([1.0, 2.0, 3.0]), x=np.array([0.5, 1.0, 1.5]), plot_axis_type="semilogy"
)
aplt.line(
    y=np.array([1.0, 2.0, 3.0]), x=np.array([0.5, 1.0, 1.5]), plot_axis_type="loglog"
)
aplt.line(
    y=np.array([1.0, 2.0, 3.0]), x=np.array([0.5, 1.0, 1.5]), plot_axis_type="scatter"
)
