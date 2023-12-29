import autoarray.plot as aplt

from matplotlib.patches import Ellipse


def test__add_patches():
    patch_overlay = aplt.PatchOverlay(facecolor="c", edgecolor="none")

    patch_0 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)
    patch_1 = Ellipse(xy=(1.0, 2.0), height=1.0, width=2.0, angle=1.0)

    patch_overlay.overlay_patches(patches=[patch_0, patch_1])
