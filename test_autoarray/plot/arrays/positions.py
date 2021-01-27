import autoarray as aa
import autoarray.plot as aplt

array = aa.Array2D.ones(shape_native=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

aplt.Array2D(array=array, positions=aa.Grid2DIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]))

aplt.Array2D(
    array=array,
    positions=aa.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]),
)
