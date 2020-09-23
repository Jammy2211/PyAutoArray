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

grid = aa.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

aplt.Array(
    array=array,
    mask=mask,
    grid=grid,
    positions=aa.GridCoordinates([(-1.0, -1.0)]),
    lines=aa.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
    include=aplt.Include(origin=True, border=True),
)

aplt.Array(
    array=array,
    mask=mask,
    grid=grid,
    positions=aa.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]),
    lines=aa.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]),
    include=aplt.Include(origin=True, border=True),
)
