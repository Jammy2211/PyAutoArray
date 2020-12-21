import autoarray as aa
import autoarray.plot as aplt

array = aa.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)
array[0] = 3.0

plotter = aplt.Plotter(
    cb=aplt.ColorBar(fraction=0.047, pad=0.01)
)

aplt.Array(
    array=array,
    plotter=plotter
)
