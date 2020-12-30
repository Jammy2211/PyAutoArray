import autoarray as aa
import autoarray.plot as aplt

grid_7x7 = aa.Grid.uniform(shape_2d=(7, 7), pixel_scales=1.0)
rectangular_grid = aa.GridRectangular.overlay_grid(grid=grid_7x7, shape_2d=(3, 3))
rectangular_mapper = aa.Mapper(source_full_grid=grid_7x7, source_pixelization_grid=rectangular_grid)

image = aa.Array.ones(shape_2d=(7, 7), pixel_scales=1.0)
image[0:4] = 5.0
noise_map = aa.Array.ones(shape_2d=(7, 7), pixel_scales=1.0)
imaging = aa.Imaging(image=image, noise_map=noise_map)

aplt.Mapper.subplot_image_and_mapper(
    image=imaging,
    mapper=rectangular_mapper,
    include=aplt.Include2D(, mapper_source_border=True, mapper_source_pixelization_grid=False
    ),
    full_indexes=[0, 1, 2, 3],
    pixelization_indexes=[[3, 4], [5]],
)
