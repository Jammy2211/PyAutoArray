import autoarray as aa
import autoarray.plot as aplt

mask = aa.Mask2D.circular(shape_native=(7, 7), pixel_scales=0.3, radius=0.6)

imaging = aa.Imaging(
    image=aa.Array2D.ones(shape_native=(7, 7), pixel_scales=0.3),
    noise_map=aa.Array2D.ones(shape_native=(7, 7), pixel_scales=0.3),
    psf=aa.Kernel2D.ones(shape_native=(3, 3), pixel_scales=0.3),
)

masked_imaging = aa.MaskedImaging(imaging=imaging, mask=mask)

grid_7x7 = aa.Grid2D.from_mask(mask=mask)
rectangular_grid = aa.Grid2DRectangular.overlay_grid(grid=grid_7x7, shape_native=(3, 3))
rectangular_mapper = aa.Mapper(
    source_grid_slim=grid_7x7, source_pixelization_grid=rectangular_grid
)

regularization = aa.reg.Constant(coefficient=1.0)

inversion = aa.Inversion(
    masked_dataset=masked_imaging,
    mapper=rectangular_mapper,
    regularization=regularization,
)

aplt.Inversion.subplot_inversion(
    inversion=inversion,
    image_positions=[(0.05, 0.05)],
    lines=[(0.0, 0.0), (0.1, 0.1)],
    full_indexes=[0],
    pixelization_indexes=[5],
)
