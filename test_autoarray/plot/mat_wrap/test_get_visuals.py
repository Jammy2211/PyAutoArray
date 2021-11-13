from os import path
import pytest

import autoarray as aa
import autoarray.plot as aplt

from autoarray.plot.mat_wrap.get_visuals import GetVisuals2D


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__visuals_in_constructor_use_imaging_and_include(mask_2d_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vector_field=2)
    include_2d = aplt.Include2D(origin=True, mask=True, border=True)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == mask_2d_7x7).all()
    assert (visuals_2d_via.border == mask_2d_7x7.border_grid_sub_1.binned).all()
    assert visuals_2d_via.vector_field == 2

    include_2d = aplt.Include2D(origin=False, mask=False, border=False)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.vector_field == 2
