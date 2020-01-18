import autoarray as aa
import autoarray.plot as aplt

grid_7x7 = aa.grid.uniform(shape_2d=(7, 7), pixel_scales=1.0)
rectangular_grid = aa.grid_rectangular.overlay_grid(grid=grid_7x7, shape_2d=(3, 3))
rectangular_mapper = aa.mapper(grid=grid_7x7, pixelization_grid=rectangular_grid)

image = aa.array.ones(shape_2d=(7,7), pixel_scales=1.0)
image[0:4] = 5.0
noise_map = aa.array.ones(shape_2d=(7,7), pixel_scales=1.0)
imaging = aa.imaging(image=image, noise_map=noise_map)

aplt.mapper.subplot_image_and_mapper(imaging=imaging, mapper=rectangular_mapper,
                                     include=aplt.Include(inversion_grid=False, inversion_border=True, inversion_pixelization_grid=False),
                                     image_pixel_indexes=[0, 1, 2, 3],
                                     source_pixel_indexes=[[3,4], [5]])