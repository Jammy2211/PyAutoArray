import autoarray.plot as aplt

from matplotlib.patches import Ellipse


def test__from_config_or_via_manual_input():

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


def test__add_patches():

    patch_overlay = aplt.PatchOverlay(facecolor="c", edgecolor="none")

    patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
    patch_1 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)

    patch_overlay.overlay_patches(patches=[patch_0, patch_1])
