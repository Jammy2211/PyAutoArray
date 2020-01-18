import autoarray as aa
import autoarray.plot as aplt

mask = aa.mask.circular(shape_2d=(7, 7), pixel_scales=0.3, radius=0.6)

imaging = aa.imaging(
    image=aa.array.ones(shape_2d=(7, 7), pixel_scales=0.3),
    noise_map=aa.array.ones(shape_2d=(7, 7), pixel_scales=0.3),
    psf=aa.kernel.ones(shape_2d=(3, 3), pixel_scales=0.3),
)

masked_imaging = aa.masked.imaging(imaging=imaging, mask=mask)

grid_7x7 = aa.grid.from_mask(mask=mask)
rectangular_grid = aa.grid_rectangular.overlay_grid(grid=grid_7x7, shape_2d=(3, 3))
rectangular_mapper = aa.mapper(grid=grid_7x7, pixelization_grid=rectangular_grid)

regularization = aa.reg.Constant(coefficient=1.0)

inversion = aa.inversion(
    masked_dataset=masked_imaging,
    mapper=rectangular_mapper,
    regularization=regularization,
)

aplt.inversion.subplot_inversion(
    inversion=inversion,
    positions=[(0.05, 0.05)],
    lines=[(0.0, 0.0), (0.1, 0.1)],
    image_pixel_indexes=[0],
    source_pixel_indexes=[5],
)
