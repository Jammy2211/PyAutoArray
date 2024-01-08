import autoarray as aa


def test__grid_is_relocated_via_border(sub_grid_2d_7x7):
    mesh = aa.mesh.Voronoi()

    mask = aa.Mask2D.circular(
        shape_native=(60, 60),
        radius=1.0,
        pixel_scales=(0.1, 0.1),
        centre=(1.0, 1.0),
        sub_size=1,
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    image_mesh = aa.image_mesh.Overlay(
        shape=(3, 3),
    )
    image_mesh = image_mesh.image_plane_mesh_grid_from(grid=grid, adapt_data=None)

    grid[8, 0] = 100.0

    mapper_grids = mesh.mapper_grids_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh,
        relocate_pix_border=True,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert grid[8, 0] != mapper.source_plane_data_grid[8, 0]
    assert mapper.source_plane_data_grid[8, 0] < 5.0

    grid[0, 0] = 0.0
    image_mesh[0, 0] = 100.0

    mapper_grids = mesh.mapper_grids_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh,
        relocate_pix_border=True,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert isinstance(mapper, aa.MapperVoronoiNoInterp)
    assert image_mesh[0, 0] != mapper.source_plane_mesh_grid[0, 0]
    assert mapper.source_plane_mesh_grid[0, 0] < 5.0

    mesh = aa.mesh.VoronoiNN()

    mapper_grids = mesh.mapper_grids_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh,
        relocate_pix_border=True,
    )

    mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

    assert isinstance(mapper, aa.MapperVoronoi)
    assert image_mesh[0, 0] != mapper.source_plane_mesh_grid[0, 0]
    assert mapper.source_plane_mesh_grid[0, 0] < 5.0
