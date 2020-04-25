import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

array_overlay = aa.Array.ones(shape_2d=(7, 7), pixel_scales=array.pixel_scales)

aplt.Array(array=array, array_overlay=array_overlay)
