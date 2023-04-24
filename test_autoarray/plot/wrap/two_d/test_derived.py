import autoarray as aa
import autoarray.plot as aplt


def test__all_class_load_and_inherit_correctly(grid_2d_irregular_7x7_list):
    origin_scatter = aplt.OriginScatter()
    origin_scatter.scatter_grid(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
    )

    assert origin_scatter.config_dict["s"] == 80

    mask_scatter = aplt.MaskScatter()
    mask_scatter.scatter_grid(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
    )

    assert mask_scatter.config_dict["s"] == 12

    border_scatter = aplt.BorderScatter()
    border_scatter.scatter_grid(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
    )

    assert border_scatter.config_dict["s"] == 13

    positions_scatter = aplt.PositionsScatter()
    positions_scatter.scatter_grid(grid=grid_2d_irregular_7x7_list)

    assert positions_scatter.config_dict["s"] == 15

    index_scatter = aplt.IndexScatter()
    index_scatter.scatter_grid_list(grid_list=grid_2d_irregular_7x7_list)

    assert index_scatter.config_dict["s"] == 20

    mesh_grid_scatter = aplt.MeshGridScatter()
    mesh_grid_scatter.scatter_grid(
        grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
    )

    assert mesh_grid_scatter.config_dict["s"] == 5

    parallel_overscan_plot = aplt.ParallelOverscanPlot()
    parallel_overscan_plot.plot_rectangular_grid_lines(
        extent=[0.0, 1.0, 0.0, 1.0], shape_native=(3, 2)
    )

    assert parallel_overscan_plot.config_dict["linewidth"] == 1

    serial_overscan_plot = aplt.SerialOverscanPlot()
    serial_overscan_plot.plot_rectangular_grid_lines(
        extent=[0.0, 1.0, 0.0, 1.0], shape_native=(3, 2)
    )

    assert serial_overscan_plot.config_dict["linewidth"] == 2

    serial_prescan_plot = aplt.SerialPrescanPlot()
    serial_prescan_plot.plot_rectangular_grid_lines(
        extent=[0.0, 1.0, 0.0, 1.0], shape_native=(3, 2)
    )

    assert serial_prescan_plot.config_dict["linewidth"] == 3
