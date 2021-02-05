import autoarray as aa
import autoarray.plot as aplt
import numpy as np

mask = aa.Mask2D.circular(shape_native=(7, 7), pixel_scales=0.3, radius=0.6)

imaging = aa.Imaging(
    image=aa.Array2D.ones(shape_native=(7, 7), pixel_scales=0.3),
    noise_map=aa.Array2D.ones(shape_native=(7, 7), pixel_scales=0.3),
    psf=aa.Kernel2D.ones(shape_native=(3, 3), pixel_scales=0.3),
)

masked_imaging = aa.MaskedImaging(imaging=imaging, mask=mask)

grid_7x7 = aa.Grid2D.from_mask(mask=mask)

grid_9 = aa.Grid2D.manual_slim(
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
    shape_native=(3, 3),
    pixel_scales=1.0,
)
voronoi_grid = aa.Grid2DVoronoi(
    grid=grid_9,
    nearest_pixelization_index_for_slim_index=np.zeros(
        shape=grid_7x7.shape_slim, dtype="int"
    ),
)

voronoi_mapper = aa.Mapper(
    source_grid_slim=grid_7x7, source_pixelization_grid=voronoi_grid
)

regularization = aa.reg.Constant(coefficient=1.0)

inversion = aa.Inversion(
    masked_dataset=masked_imaging, mapper=voronoi_mapper, regularization=regularization
)

aplt.Inversion.subplot_inversion(
    inversion=inversion,
    image_positions=[(0.05, 0.05)],
    lines=[(0.0, 0.0), (0.1, 0.1)],
    full_indexes=[0],
    pixelization_indexes=[5],
)
