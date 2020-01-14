import autoarray as aa

plotter = aa.plotter.Plotter()


array = aa.array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

# plotter.array.plot(array=array, lines=[(1.0, 1.0), (2.0, 2.0), (2.0, 4.0)])

plotter.array.plot(array=array, lines=[[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]])

