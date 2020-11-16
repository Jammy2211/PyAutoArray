import autoarray as aa
import autoarray.plot as aplt
import numpy as np

mask = aa.Mask2D.circular(shape_2d=(7, 7), pixel_scales=0.3, radius=0.6)

imaging = aa.Imaging(
    image=aa.Array.ones(shape_2d=(7, 7), pixel_scales=0.3),
    noise_map=aa.Array.ones(shape_2d=(7, 7), pixel_scales=0.3),
    psf=aa.Kernel.ones(shape_2d=(3, 3), pixel_scales=0.3),
)

masked_imaging = aa.MaskedImaging(imaging=imaging, mask=mask)

grid_7x7 = aa.Grid.from_mask(mask=mask)

grid_9 = aa.Grid.manual_1d(
    grid=[
        [0.6, -0.3],
        [0.5, -0.8],
        [0.2, 0.1],
        [0.0, 0.5],
        [-0.3, -0.8],
        [-0.6, -0.5],
        [-0.4, -1.1],
        [-1.2, 0.8],
        [-1.5, 0.9],
    ],
    shape_2d=(3, 3),
    pixel_scales=1.0,
)
voronoi_grid = aa.GridVoronoi(
    grid=grid_9,
    nearest_pixelization_1d_index_for_mask_1d_index=np.zeros(
        shape=grid_7x7.shape_1d, dtype="int"
    ),
)

voronoi_mapper = aa.Mapper(grid=grid_7x7, pixelization_grid=voronoi_grid)

regularization = aa.reg.Constant(coefficient=1.0)

inversion = aa.Inversion(
    masked_dataset=masked_imaging, mapper=voronoi_mapper, regularization=regularization
)

aplt.Inversion.subplot_inversion(
    inversion=inversion,
    image_positions=[(0.05, 0.05)],
    lines=[(0.0, 0.0), (0.1, 0.1)],
    image_pixel_indexes=[0],
    source_pixel_indexes=[5],
)
