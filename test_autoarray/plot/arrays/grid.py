import autoarray as aa
import autoarray.plot as aplt

array = aa.Array2D.ones(shape_native=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

grid = aa.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.5)

aplt.Array2D(array=array, grid=grid)
