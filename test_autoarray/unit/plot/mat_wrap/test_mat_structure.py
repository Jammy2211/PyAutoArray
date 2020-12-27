import autoarray as aa
import autoarray.plot as aplt

from os import path

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

directory = path.dirname(path.realpath(__file__))


class TestArrayOverlay:
    def test__from_config_or_via_manual_input(self):

        array_overlaylay = aplt.ArrayOverlay()

        assert array_overlaylay.config_dict["alpha"] == 0.5

        array_overlaylay = aplt.ArrayOverlay(alpha=0.6)

        assert array_overlaylay.config_dict["alpha"] == 0.6

        array_overlaylay = aplt.ArrayOverlay(use_subplot_defaults=True)

        assert array_overlaylay.config_dict["alpha"] == 0.7

        array_overlaylay = aplt.ArrayOverlay(use_subplot_defaults=True, alpha=0.8)

        assert array_overlaylay.config_dict["alpha"] == 0.8

    def test__overlay_array__works_for_reasonable_values(self):

        array_overlay = aa.Array.manual_2d(
            array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.5
        )

        figure = aplt.Figure(aspect="auto")

        array_overlaylay = aplt.ArrayOverlay(alpha=0.5)

        array_overlaylay.overlay_array(array=array_overlay, figure=figure)


class TestGridScatter:
    def test__from_config_or_via_manual_input(self):

        grid_scatter = aplt.GridScatter()

        assert grid_scatter.config_dict["marker"] == "x"
        assert grid_scatter.colors == ["y"]

        grid_scatter = aplt.GridScatter(marker="x")

        assert grid_scatter.config_dict["marker"] == "x"
        assert grid_scatter.colors == ["y"]

        grid_scatter = aplt.GridScatter(use_subplot_defaults=True)

        assert grid_scatter.config_dict["marker"] == "."
        assert grid_scatter.colors == ["r"]

        grid_scatter = aplt.GridScatter(use_subplot_defaults=True, colors=["r", "b"])

        assert grid_scatter.config_dict["marker"] == "."
        assert grid_scatter.colors == ["r", "b"]

    def test__scatter_grid(self):

        scatter = aplt.GridScatter(size=2, marker="x", colors="k")

        scatter.scatter_grid(grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0))

    def test__scatter_colored_grid__lists_of_coordinates_or_equivalent_2d_grids__with_color_array(
        self,
    ):

        scatter = aplt.GridScatter(size=2, marker="x", colors="k")

        cmap = plt.get_cmap("jet")

        scatter.scatter_grid_colored(
            grid=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
            color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            cmap=cmap,
        )
        scatter.scatter_grid_colored(
            grid=aa.Grid.uniform(shape_2d=(3, 2), pixel_scales=1.0),
            color_array=np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            cmap=cmap,
        )

    def test__scatter_grid_indexes_1d__input_grid_is_ndarray_and_indexes_are_valid(
        self,
    ):

        scatter = aplt.GridScatter(size=2, marker="x", colors="k")

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0), indexes=[0, 1, 2]
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0), indexes=[[0, 1, 2]]
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[0, 1], [2]],
        )

    def test__scatter_grid_indexes_2d__input_grid_is_ndarray_and_indexes_are_valid(
        self,
    ):

        scatter = aplt.GridScatter(size=2, marker="x", colors="k")

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[(0, 0), (0, 1), (0, 2)],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[(0, 0), (0, 1), (0, 2)]],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[(0, 0), (0, 1)], [(0, 2)]],
        )

        scatter.scatter_grid_indexes(
            grid=aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=1.0),
            indexes=[[[0, 0], [0, 1]], [[0, 2]]],
        )

    def test__scatter_coordinates(self):

        scatter = aplt.GridScatter(size=2, marker="x", colors="k")

        scatter.scatter_grid_grouped(
            grid_grouped=aa.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)])
        )
        scatter.scatter_grid_grouped(
            grid_grouped=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
            )
        )


class TestLinePlot:
    def test___from_config_or_via_manual_input(self):

        line_plot = aplt.LinePlot()

        assert line_plot.config_dict["width"] == 3
        assert line_plot.colors == ["k", "w"]

        line_plot = aplt.LinePlot(colors=["k", "b"])

        assert line_plot.config_dict["width"] == 3
        assert line_plot.colors == ["k", "b"]

        line_plot = aplt.LinePlot(use_subplot_defaults=True)

        assert line_plot.config_dict["width"] == 1
        assert line_plot.colors == ["k"]

        line_plot = aplt.LinePlot(use_subplot_defaults=True, style=".")

        assert line_plot.config_dict["width"] == 1
        assert line_plot.colors == ["k"]

    def test__draw_y_vs_x__works_for_reasonable_values(self):

        line = aplt.LinePlot(linewidth=2, linestyle="-", colors="k")

        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="linear")
        line.plot_y_vs_x(
            y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="semilogy"
        )
        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="loglog")

        line = aplt.LinePlot(colors="k", s=2)

        line.plot_y_vs_x(y=[1.0, 2.0, 3.0], x=[1.0, 2.0, 3.0], plot_axis_type="scatter")

    def test__draw_vertical_lines__works_for_reasonable_values(self):

        line = aplt.LinePlot(linewidth=2, linestyle="-", colors="k")

        line.plot_vertical_lines(vertical_lines=[[0.0]])
        line.plot_vertical_lines(vertical_lines=[[1.0], [2.0]])
        line.plot_vertical_lines(vertical_lines=[[0.0]], vertical_line_labels=["hi"])
        line.plot_vertical_lines(
            vertical_lines=[[1.0], [2.0]], vertical_line_labels=["hi1", "hi2"]
        )

    def test__draw_coordinates(self):

        line = aplt.LinePlot(linewidth=2, linestyle="--", colors="k")

        line.plot_grid_grouped(
            grid_grouped=aa.GridIrregularGrouped([[(1.0, 1.0), (2.0, 2.0)]])
        )
        line.plot_grid_grouped(
            grid_grouped=aa.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
            )
        )

    def test__draw_rectangular_grid_lines__draws_for_valid_extent_and_shape(self):

        line = aplt.LinePlot(linewidth=2, linestyle="--", colors="k")

        line.plot_rectangular_grid_lines(extent=[0.0, 1.0, 0.0, 1.0], shape_2d=(3, 2))
        line.plot_rectangular_grid_lines(
            extent=[-4.0, 8.0, -3.0, 10.0], shape_2d=(8, 3)
        )


class TestVectorFieldQuiver:
    def test__from_config_or_via_manual_input(self):

        vector_field_quiver = aplt.VectorFieldQuiver()

        assert vector_field_quiver.config_dict["headlength"] == 0

        vector_field_quiver = aplt.VectorFieldQuiver(headlength=1)

        assert vector_field_quiver.config_dict["headlength"] == 1

        vector_field_quiver = aplt.VectorFieldQuiver(use_subplot_defaults=True)

        assert vector_field_quiver.config_dict["headlength"] == 0.1

        vector_field_quiver = aplt.VectorFieldQuiver(headlength=12)

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

        vector_field = aa.VectorFieldIrregular(
            vectors=[(1.0, 2.0), (2.0, 1.0)], grid=[(-1.0, 0.0), (-2.0, 0.0)]
        )

        quiver.quiver_vector_field(vector_field=vector_field)


class TestPatcher:
    def test__from_config_or_via_manual_input(self):

        patch_overlay = aplt.PatchOverlay()

        assert patch_overlay.config_dict["facecolor"] == None
        assert patch_overlay.config_dict["edgecolor"] == "cyan"

        patch_overlay = aplt.PatchOverlay(facecolor="r", edgecolor="g")

        assert patch_overlay.config_dict["facecolor"] == "r"
        assert patch_overlay.config_dict["edgecolor"] == "g"

        patch_overlay = aplt.PatchOverlay(use_subplot_defaults=True)

        assert patch_overlay.config_dict["facecolor"] == None
        assert patch_overlay.config_dict["edgecolor"] == "y"

        patch_overlay = aplt.PatchOverlay(
            use_subplot_defaults=True, facecolor="b", edgecolor="p"
        )

        assert patch_overlay.config_dict["facecolor"] == "b"
        assert patch_overlay.config_dict["edgecolor"] == "p"

    def test__add_patches(self):

        patch_overlay = aplt.PatchOverlay(facecolor="cy", edgecolor="none")

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

        voronoi_drawer = aplt.VoronoiDrawer(use_subplot_defaults=True)

        assert voronoi_drawer.config_dict["linewidth"] == 1.0
        assert voronoi_drawer.config_dict["edgecolor"] == "r"

        voronoi_drawer = aplt.VoronoiDrawer(use_subplot_defaults=True, edgecolor="b")

        assert voronoi_drawer.config_dict["linewidth"] == 1.0
        assert voronoi_drawer.config_dict["edgecolor"] == "b"

    def test__draws_voronoi_pixels_for_sensible_input(self, voronoi_mapper_9_3x3):

        voronoi_drawer = aplt.VoronoiDrawer(linewidth=0.5, linecolor="r", alpha=1.0)

        voronoi_drawer.draw_voronoi_pixels(
            mapper=voronoi_mapper_9_3x3, values=None, cmap=None, cb=None
        )

        voronoi_drawer.draw_voronoi_pixels(
            mapper=voronoi_mapper_9_3x3,
            values=np.ones(9),
            cmap="jet",
            cb=aplt.Colorbar(ticksize=1, fraction=0.1, pad=0.05),
        )
