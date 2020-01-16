import autoarray as aa
from autoarray import plotter as aplt

plotter = aplt.Plotter(mask_scatterer=aplt.Scatterer(size=10, marker="x", color="k"))

array = aa.array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

mask = aa.mask.circular(
    shape_2d=array.shape_2d,
    pixel_scales=array.pixel_scales,
    radius=5.0,
    centre=(2.0, 2.0),
)

grid = aa.grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

plotter.plot_array(array=array, mask=mask, grid=grid, positions=[(-1.0, -1.0)],
lines=[(1.0, 1.0), (2.0, 2.0)], include_origin=True, include_border=True)

plotter.plot_array(array=array, mask=mask, grid=grid, positions=[[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]],
lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]], include_origin=True, include_border=True)
