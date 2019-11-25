import autoarray as aa

array = aa.array.ones(shape_2d=(11, 11), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

mask = aa.mask.circular(
    shape_2d=array.shape_2d,
    pixel_scales=array.pixel_scales,
    radius=3.0,
    centre=(2.0, 2.0),
)

grid = aa.grid.uniform(shape_2d=(5, 5), pixel_scales=array.pixel_scales, origin=(2.0, 2.0))

aa.plot.array(array=array, mask=mask, grid=grid, points=[[[1.0, 1.0]], [[2.0, 2.0]]])

masked_array = aa.masked.array(array_1d=array, mask=mask)

aa.plot.array(array=masked_array)

aa.plot.array(array=masked_array, use_scaled_units=False)

# aa.plot.array(array=masked_array, unit_conversion_factor=10.0)