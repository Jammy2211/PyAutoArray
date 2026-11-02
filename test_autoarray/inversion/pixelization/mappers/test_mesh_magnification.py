"""
`Mapper.data_pixel_area_for_pix_from` and `Mapper.mesh_magnification` (PyAutoArray#673).

The tests reuse the identity lens of `mesh_geometry/test_areas_for_magnification.py`: the source-plane data grid is
the image grid itself, so the true magnification of every source-plane region is exactly 1.

The checks are:

(a) the image-plane area mapped to the pixels sums to the mask area, and `sum(mesh_magnification * areas)` gives it
    back, for every mesh;
(b) away from the mesh edge each pixel's magnification is close to 1, within a per-mesh tolerance;
(c) a pixel with zero source-plane area has a magnification of zero.

Per-pixel tolerances (median |mu - 1| over pixels whose centre is inside |y|, |x| < 1.5 on this module's coarse 24x24
grid, 2x2 sub-grid and 12x12-equivalent mesh) are set ~2x the measured value, quoted in brackets. The image area a
pixel receives follows the mapper's interpolation weights, while its source-plane area follows
`areas_for_magnification`, so the per-pixel error is set by how well those match per node:

- Tight: RectangularUniform [9.1e-4], RectangularBilinearAdaptDensity [5.5e-4], RectangularRTUAdaptDensity
  [1.3e-3], Delaunay [2.9e-3], KNearestNeighbor [1.3e-3].
- Looser: RectangularRTUAdaptImage [7.9e-3], RectangularBilinearAdaptImage [1.6e-2] (a steep adapt-image CDF on this
  coarse grid); KNNBarycentric [2.5e-2] (discontinuous clipped weights).
- Documented approximation: DelaunayNN [9.2e-2] uses the Delaunay dual areas, exact only for the linear interpolant,
  not for its Sibson weights. The error averages out over a source (phase-4 total magnification 4.3e-3) but not
  per pixel.
"""

import numpy as np
import pytest

from test_autoarray.inversion.pixelization.mesh_geometry.test_areas_for_magnification import (
    PIXEL_SCALE,
    SHAPE,
    _setup,
)

MEDIAN_TOLERANCE = {
    "RectangularUniform": 2.0e-3,
    "RectangularBilinearAdaptDensity": 1.5e-3,
    "RectangularBilinearAdaptImage": 3.5e-2,
    "RectangularRTUAdaptDensity": 3.0e-3,
    "RectangularRTUAdaptImage": 1.6e-2,
    "Delaunay": 6.0e-3,
    "DelaunayNN": 0.2,
    "KNearestNeighbor": 3.0e-3,
    "KNNBarycentric": 5.0e-2,
}

MASK_AREA = (SHAPE * PIXEL_SCALE) ** 2


@pytest.mark.parametrize("name", list(MEDIAN_TOLERANCE))
def test__image_areas_and_magnification_integrate_to_the_mask_area(name):
    _, mapper, areas, _, _ = _setup(name)

    image_areas = mapper.data_pixel_area_for_pix_from()

    assert image_areas.shape == (mapper.params,)
    assert image_areas.sum() == pytest.approx(MASK_AREA, rel=1.0e-10)
    assert np.sum(mapper.mesh_magnification * areas) == pytest.approx(
        MASK_AREA, rel=1.0e-6
    )


@pytest.mark.parametrize("name", list(MEDIAN_TOLERANCE))
def test__identity_lens__interior_pixels_have_unit_magnification(name):
    _, mapper, areas, _, _ = _setup(name)

    centres = np.asarray(mapper.source_plane_mesh_grid)
    interior = (np.abs(centres) < 1.5).all(axis=1) & (areas > 0.0)

    mu = mapper.mesh_magnification[interior]

    assert np.all(np.isfinite(mapper.mesh_magnification))
    assert np.median(np.abs(mu - 1.0)) < MEDIAN_TOLERANCE[name]


def test__zero_area_pixels_have_zero_magnification():
    _, mapper, _, _, _ = _setup("RectangularBilinearAdaptDensity")

    areas = np.asarray(mapper.mesh_geometry.areas_for_magnification)

    assert np.any(areas == 0.0)
    assert np.all(mapper.mesh_magnification[areas == 0.0] == 0.0)
