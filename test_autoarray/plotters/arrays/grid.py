import autoarray as aa
from autoarray import plotter as aplt

plotter = aplt.Plotter(mask_scatterer=aplt.Scatterer(size=10, marker="x", color="k"))

array = aa.array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

plotter.plot_array(array=array, grid=grid)
