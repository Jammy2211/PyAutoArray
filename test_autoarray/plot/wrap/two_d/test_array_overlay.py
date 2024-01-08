import autoarray as aa
import autoarray.plot as aplt


def test__overlay_array__works_for_reasonable_values():
    arr = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.5, origin=(2.0, 2.0)
    )

    figure = aplt.Figure(aspect="auto")

    array_overlay = aplt.ArrayOverlay(alpha=0.5)

    array_overlay.overlay_array(array=arr, figure=figure)
