import autoarray as aa
import autoarray.plot as aplt

import matplotlib.colors as colors


def test__loads_values_from_config_if_not_manually_input():

    cmap = aplt.Cmap()

    assert cmap.config_dict["cmap"] == "default"
    assert cmap.config_dict["norm"] == "linear"

    cmap = aplt.Cmap(cmap="cold")

    assert cmap.config_dict["cmap"] == "cold"
    assert cmap.config_dict["norm"] == "linear"

    cmap = aplt.Cmap()
    cmap.is_for_subplot = True

    assert cmap.config_dict["cmap"] == "default"
    assert cmap.config_dict["norm"] == "linear"

    cmap = aplt.Cmap(cmap="cold")
    cmap.is_for_subplot = True

    assert cmap.config_dict["cmap"] == "cold"
    assert cmap.config_dict["norm"] == "linear"


def test__norm_from__uses_input_vmin_and_max_if_input():

    cmap = aplt.Cmap(vmin=0.0, vmax=1.0, norm="linear")

    norm = cmap.norm_from(array=None)

    assert isinstance(norm, colors.Normalize)
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0

    cmap = aplt.Cmap(vmin=0.0, vmax=1.0, norm="log")

    norm = cmap.norm_from(array=None)

    assert isinstance(norm, colors.LogNorm)
    assert norm.vmin == 1.0e-4  # Increased from 0.0 to ensure min isn't inf
    assert norm.vmax == 1.0

    cmap = aplt.Cmap(
        vmin=0.0, vmax=1.0, linthresh=2.0, linscale=3.0, norm="symmetric_log"
    )

    norm = cmap.norm_from(array=None)

    assert isinstance(norm, colors.SymLogNorm)
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0
    assert norm.linthresh == 2.0


def test__norm_from__cmap_symmetric_true():

    cmap = aplt.Cmap(vmin=-0.5, vmax=1.0, norm="linear", symmetric=True)

    norm = cmap.norm_from(array=None)

    assert isinstance(norm, colors.Normalize)
    assert norm.vmin == -1.0
    assert norm.vmax == 1.0

    cmap = aplt.Cmap(vmin=-2.0, vmax=1.0, norm="linear")
    cmap = cmap.symmetric

    norm = cmap.norm_from(array=None)

    assert isinstance(norm, colors.Normalize)
    assert norm.vmin == -2.0
    assert norm.vmax == 2.0


def test__norm_from__uses_array_to_get_vmin_and_max_if_no_manual_input():

    array = aa.Array2D.ones(shape_native=(2, 2), pixel_scales=1.0)
    array[0] = 0.0

    cmap = aplt.Cmap(vmin=None, vmax=None, norm="linear")

    norm = cmap.norm_from(array=array)

    assert isinstance(norm, colors.Normalize)
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0

    cmap = aplt.Cmap(vmin=None, vmax=None, norm="log")

    norm = cmap.norm_from(array=array)

    assert isinstance(norm, colors.LogNorm)
    assert norm.vmin == 1.0e-4  # Increased from 0.0 to ensure min isn't inf
    assert norm.vmax == 1.0

    cmap = aplt.Cmap(
        vmin=None, vmax=None, linthresh=2.0, linscale=3.0, norm="symmetric_log"
    )

    norm = cmap.norm_from(array=array)

    assert isinstance(norm, colors.SymLogNorm)
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0
    assert norm.linthresh == 2.0
