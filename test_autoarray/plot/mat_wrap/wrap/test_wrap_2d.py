import autoarray as aa
import autoarray.plot as aplt

from os import path

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

directory = path.dirname(path.realpath(__file__))


class TestArrayOverlay:
    def test__from_config_or_via_manual_input(self):

        array_overlay = aplt.ArrayOverlay()

        assert array_overlay.config_dict["alpha"] == 0.5

        array_overlay = aplt.ArrayOverlay(alpha=0.6)

        assert array_overlay.config_dict["alpha"] == 0.6

        array_overlay = aplt.ArrayOverlay()
        array_overlay.is_for_subplot = True

        assert array_overlay.config_dict["alpha"] == 0.7

        array_overlay = aplt.ArrayOverlay(alpha=0.8)
        array_overlay.is_for_subplot = True

        assert array_overlay.config_dict["alpha"] == 0.8

    def test__overlay_array__works_for_reasonable_values(self):

        arr = aa.Array2D.manual_native(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.5, origin=(2.0, 2.0)
        )

        figure = aplt.Figure(aspect="auto")

        array_overlay = aplt.ArrayOverlay(alpha=0.5)

        array_overlay.overlay_array(array=arr, figure=figure)


class TestGridScatter:
    def test__from_config_or_via_manual_input(self):

        grid_scatter = aplt.GridScatter()

        assert grid_scatter.config_dict["marker"] == "x"
        assert grid_scatter.config_dict["c"] == "y"

        grid_scatter = aplt.GridScatter(marker="x")

        assert grid_scatter.config_dict["marker"] == "x"
        assert grid_scatter.config_dict["c"] == "y"

        grid_scatter = aplt.GridScatter()
        grid_scatter.is_for_subplot = True

        assert grid_scatter.config_dict["marker"] == "."
        assert grid_scatter.config_dict["c"] == "r"

        grid_scatter = aplt.GridScatter(c=["r", "b"])
        grid_scatter.is_for_subplot = True

        assert grid_scatter.config_dict["marker"] == "."
        assert grid_scatter.config_dict["c"] == ["r", "b"]

    def test__scatter_grid(self):

        scatter = aplt.GridScatter(s=2, marker="x", c="k")

        scatter.scatter_grid(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
        )

    def test__scatter_colored_grid__lists_of_coordinates_or_equivalent_2d_grids__with_color_array(
        self,
    ):

        scatter = aplt.GridScatter(s=2, marker="x", c="k")

        cmap = plt.get_cmap("jet")

        scatter.scatter_grid_colored(
            grid=aa.Grid2DIrregular(
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)]
            ),
            color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            cmap=cmap,
        )
        scatter.scatter_grid_colored(
            grid=aa.Grid2D.uniform(shape_native=(3, 2), pixel_scales=1.0),
            color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            cmap=cmap,
        )

    def test__scatter_grid_indexes_1d__input_grid_is_ndarray_and_indexes_are_valid(
        self,
    ):

        scatter = aplt.GridScatter(s=2, marker="x", c="k")

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[0, 1, 2],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[[0, 1, 2]],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[[0, 1], [2]],
        )

    def test__scatter_grid_indexes_2d__input_grid_is_ndarray_and_indexes_are_valid(
        self,
    ):

        scatter = aplt.GridScatter(s=2, marker="x", c="k")

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[(0, 0), (0, 1), (0, 2)],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[[(0, 0), (0, 1), (0, 2)]],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[[(0, 0), (0, 1)], [(0, 2)]],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            indexes=[[[0, 0], [0, 1]], [[0, 2]]],
        )

    def test__scatter_coordinates(self):

        scatter = aplt.GridScatter(s=2, marker="x", c="k")

        scatter.scatter_grid_list(
            grid_list=[aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])]
        )


class TestGridPlot:
    def test___from_config_or_via_manual_input(self):

        grid_plot = aplt.GridPlot()

        assert grid_plot.config_dict["linewidth"] == 3
        assert grid_plot.config_dict["c"] == "k"

        grid_plot = aplt.GridPlot(c=["k", "b"])

        assert grid_plot.config_dict["linewidth"] == 3
        assert grid_plot.config_dict["c"] == ["k", "b"]

        grid_plot = aplt.GridPlot()
        grid_plot.is_for_subplot = True

        assert grid_plot.config_dict["linewidth"] == 1
        assert grid_plot.config_dict["c"] == "k"

        grid_plot = aplt.GridPlot(style=".")
        grid_plot.is_for_subplot = True

        assert grid_plot.config_dict["linewidth"] == 1
        assert grid_plot.config_dict["c"] == "k"

    def test__plot_rectangular_grid_lines__draws_for_valid_extent_and_shape(self):

        line = aplt.GridPlot(linewidth=2, linestyle="--", c="k")

        line.plot_rectangular_grid_lines(
            extent=[0.0, 1.0, 0.0, 1.0], shape_native=(3, 2)
        )
        line.plot_rectangular_grid_lines(
            extent=[-4.0, 8.0, -3.0, 10.0], shape_native=(8, 3)
        )

    def test__plot_grid_list(self):

        line = aplt.GridPlot(linewidth=2, linestyle="--", c="k")

        line.plot_grid_list(grid_list=[aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])])
        line.plot_grid_list(
            grid_list=[
                aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)]),
                aa.Grid2DIrregular([(3.0, 3.0)]),
            ]
        )


class TestGridErrorbar:
    def test__from_config_or_via_manual_input(self):

        grid_errorbar = aplt.GridErrorbar()

        assert grid_errorbar.config_dict["marker"] == "o"
        assert grid_errorbar.config_dict["c"] == "k"

        grid_errorbar = aplt.GridErrorbar(marker="x")

        assert grid_errorbar.config_dict["marker"] == "x"
        assert grid_errorbar.config_dict["c"] == "k"

        grid_errorbar = aplt.GridErrorbar()
        grid_errorbar.is_for_subplot = True

        assert grid_errorbar.config_dict["marker"] == "."
        assert grid_errorbar.config_dict["c"] == "b"

        grid_errorbar = aplt.GridErrorbar(c=["r", "b"])
        grid_errorbar.is_for_subplot = True

        assert grid_errorbar.config_dict["marker"] == "."
        assert grid_errorbar.config_dict["c"] == ["r", "b"]

    def test__errorbar_grid(self):

        errorbar = aplt.GridErrorbar(marker="x", c="k")

        errorbar.errorbar_grid(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
        )

        errorbar = aplt.GridErrorbar(marker="x", c="k")

        errorbar.errorbar_grid(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0),
            y_errors=[1.0] * 9,
            x_errors=[1.0] * 9,
        )

    def test__errorbar_coordinates(self):

        errorbar = aplt.GridErrorbar(marker="x", c="k")

        errorbar.errorbar_grid_list(
            grid_list=[aa.Grid2DIrregular([(1.0, 1.0), (2.0, 2.0)])],
            y_errors=[1.0] * 2,
            x_errors=[1.0] * 2,
        )


class TestVectorFieldQuiver:
    def test__from_config_or_via_manual_input(self):

        vector_field_quiver = aplt.VectorFieldQuiver()

        assert vector_field_quiver.config_dict["headlength"] == 0

        vector_field_quiver = aplt.VectorFieldQuiver(headlength=1)

        assert vector_field_quiver.config_dict["headlength"] == 1

        vector_field_quiver = aplt.VectorFieldQuiver()
        vector_field_quiver.is_for_subplot = True

        assert vector_field_quiver.config_dict["headlength"] == 0.1

        vector_field_quiver = aplt.VectorFieldQuiver(headlength=12)
        vector_field_quiver.is_for_subplot = True

        assert vector_field_quiver.config_dict["headlength"] == 12

    def test__quiver_vector_field(self):

        quiver = aplt.VectorFieldQuiver(
            headlength=5,
            pivot="middle",
            linewidth=3,
            units="xy",
            angles="xy",
            headwidth=6,
            alpha=1.0,
        )

        vector_field = aa.VectorField2DIrregular(
            vectors=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
        )

        quiver.quiver_vector_field(vector_field=vector_field)


class TestPatcher:
    def test__from_config_or_via_manual_input(self):

        patch_overlay = aplt.PatchOverlay()

        assert patch_overlay.config_dict["facecolor"] == None
        assert patch_overlay.config_dict["edgecolor"] == "c"

        patch_overlay = aplt.PatchOverlay(facecolor="r", edgecolor="g")

        assert patch_overlay.config_dict["facecolor"] == "r"
        assert patch_overlay.config_dict["edgecolor"] == "g"

        patch_overlay = aplt.PatchOverlay()
        patch_overlay.is_for_subplot = True

        assert patch_overlay.config_dict["facecolor"] == None
        assert patch_overlay.config_dict["edgecolor"] == "y"

        patch_overlay = aplt.PatchOverlay(facecolor="b", edgecolor="p")
        patch_overlay.is_for_subplot = True

        assert patch_overlay.config_dict["facecolor"] == "b"
        assert patch_overlay.config_dict["edgecolor"] == "p"

    def test__add_patches(self):

        patch_overlay = aplt.PatchOverlay(facecolor="c", edgecolor="none")

        patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
        patch_1 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)

        patch_overlay.overlay_patches(patches=[patch_0, patch_1])


class TestVoronoiDrawer:
    def test__from_config_or_via_manual_input(self):

        voronoi_drawer = aplt.VoronoiDrawer()

        assert voronoi_drawer.config_dict["linewidth"] == 0.3
        assert voronoi_drawer.config_dict["edgecolor"] == "k"

        voronoi_drawer = aplt.VoronoiDrawer(linewidth=0.5)

        assert voronoi_drawer.config_dict["linewidth"] == 0.5
        assert voronoi_drawer.config_dict["edgecolor"] == "k"

        voronoi_drawer = aplt.VoronoiDrawer()
        voronoi_drawer.is_for_subplot = True

        assert voronoi_drawer.config_dict["linewidth"] == 1.0
        assert voronoi_drawer.config_dict["edgecolor"] == "r"

        voronoi_drawer = aplt.VoronoiDrawer(edgecolor="b")
        voronoi_drawer.is_for_subplot = True

        assert voronoi_drawer.config_dict["linewidth"] == 1.0
        assert voronoi_drawer.config_dict["edgecolor"] == "b"

    def test__draws_voronoi_pixels_for_sensible_input(self, voronoi_mapper_9_3x3):

        voronoi_drawer = aplt.VoronoiDrawer(linewidth=0.5, edgecolor="r", alpha=1.0)

        voronoi_drawer.draw_voronoi_pixels(
            mapper=voronoi_mapper_9_3x3, values=None, cmap=aplt.Cmap(), colorbar=None
        )

        values = np.ones(9)
        values[0] = 0.0

        voronoi_drawer.draw_voronoi_pixels(
            mapper=voronoi_mapper_9_3x3,
            values=values,
            cmap=aplt.Cmap(),
            colorbar=aplt.Colorbar(fraction=0.1, pad=0.05),
        )


class TestDerivedClasses:
    def test__all_class_load_and_inherit_correctly(self, grid_2d_irregular_7x7_list):

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

        pixelization_grid_scatter = aplt.PixelizationGridScatter()
        pixelization_grid_scatter.scatter_grid(
            grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)
        )

        assert pixelization_grid_scatter.config_dict["s"] == 5

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
