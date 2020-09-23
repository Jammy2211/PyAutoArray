import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

aplt.Array(array=array, grid=grid)
