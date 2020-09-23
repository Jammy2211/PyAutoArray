import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

aplt.Array(array=array, positions=aa.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]))

aplt.Array(
    array=array,
    positions=aa.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]),
)
