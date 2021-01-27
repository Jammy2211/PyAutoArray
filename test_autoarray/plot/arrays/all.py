import autoarray as aa
import autoarray.plot as aplt

array = aa.Array2D.ones(shape_native=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

mask = aa.Mask2D.circular(
    shape_native=array.shape_native,
    pixel_scales=array.pixel_scales,
    radius=5.0,
    centre=(2.0, 2.0),
)

grid = aa.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.5)

aplt.Array2D(
    array=array,
    mask=mask,
    grid=grid,
    positions=aa.Grid2DIrregularGrouped([(-1.0, -1.0)]),
    lines=aa.Grid2DIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
    include=aplt.Include2D(origin=True, border=True),
)

aplt.Array2D(
    array=array,
    mask=mask,
    grid=grid,
    positions=aa.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]),
    lines=aa.Grid2DIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]),
    include=aplt.Include2D(origin=True, border=True),
)
