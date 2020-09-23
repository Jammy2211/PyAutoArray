import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

mask = aa.Mask.circular(
    shape_2d=array.shape_2d,
    pixel_scales=array.pixel_scales,
    radius=5.0,
    centre=(2.0, 2.0),
)

aplt.Array(array=array, mask=mask, include_border=True)
