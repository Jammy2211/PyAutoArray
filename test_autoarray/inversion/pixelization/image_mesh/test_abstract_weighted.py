import numpy as np
import pytest

import autoarray as aa


def test__weight_map_from():
    adapt_data = np.array([-1.0, 1.0, 3.0])

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert weight_map == pytest.approx([0.33333, 0.33333, 1.0], 1.0e-4)

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=2.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert weight_map == pytest.approx([0.11111, 0.11111, 1.0], 1.0e-4)

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=1.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert weight_map == pytest.approx([1.0, 1.0, 1.0], 1.0e-4)
