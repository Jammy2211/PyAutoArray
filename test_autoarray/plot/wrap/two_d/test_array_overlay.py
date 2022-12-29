import autoarray as aa
import autoarray.plot as aplt


def test__from_config_or_via_manual_input():

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


def test__overlay_array__works_for_reasonable_values():

    arr = aa.Array2D.without_mask(
        array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.5, origin=(2.0, 2.0)
    )

    figure = aplt.Figure(aspect="auto")

    array_overlay = aplt.ArrayOverlay(alpha=0.5)

    array_overlay.overlay_array(array=arr, figure=figure)
