from os import path
import pytest

import autoarray.plot as aplt

@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__via_array_1d_from(array_1d_7):

    visuals_1d = aplt.Visuals1D(origin=(1.0, 1.0))
    include_1d = aplt.Include1D(origin=True, mask=True)

    get_visuals = aplt.GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_array_1d_from(array_1d=array_1d_7)

    assert visuals_1d_via.origin == (1.0, 1.0)
    assert (visuals_1d_via.mask == array_1d_7.mask).all()

    include_1d = aplt.Include1D(origin=False, mask=False)

    get_visuals = aplt.GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_array_1d_from(array_1d=array_1d_7)

    assert visuals_1d_via.origin == (1.0, 1.0)
    assert visuals_1d_via.mask == None
