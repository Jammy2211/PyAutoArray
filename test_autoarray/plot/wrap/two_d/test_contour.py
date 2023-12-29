import autoarray as aa
import autoarray.plot as aplt


def test__contour__works_for_reasonable_values():
    arr = aa.Array2D.no_mask(
        values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=0.5, origin=(2.0, 2.0)
    )

    contour = aplt.Contour()

    contour.set(array=arr, extent=[0.0, 1.0, 0.0, 1.0])
