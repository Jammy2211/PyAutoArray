import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

aplt.Array(array=array, lines=[(1.0, 1.0), (2.0, 2.0)])

aplt.Array(array=array, lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]])
