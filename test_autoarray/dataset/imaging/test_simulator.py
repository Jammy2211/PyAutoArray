import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="image_central_delta_3x3")
def make_array_2d_7x7():
    return aa.Array2D.no_mask(
        values=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=0.1,
    )


def test__via_image_from__all_features_off(image_central_delta_3x3):
    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert dataset.pixel_scales == (0.1, 0.1)


def test__via_image_from__noise_off__noise_map_is_noise_value(image_central_delta_3x3):
    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        noise_if_add_noise_false=0.2,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()

    assert np.allclose(dataset.noise_map.native, 0.2 * np.ones((3, 3)))


def test__via_image_from__psf_blurs_image_with_edge_trimming(image_central_delta_3x3):
    psf = aa.Kernel2D.no_mask(
        values=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )

    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        psf=psf,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
        normalize_psf=False,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    ).all()


def test__via_image_from__include_poisson_noise_in_noise_map(image_central_delta_3x3):
    image = image_central_delta_3x3 + 1.0

    simulator = aa.SimulatorImaging(
        exposure_time=20.0, add_poisson_noise_to_data=True,

        include_poisson_noise_in_noise_map=True,
        noise_seed=1
    )

    dataset = simulator.via_image_from(image=image)

    assert dataset.data.native == pytest.approx(
        np.array([[1.05, 1.3, 1.25], [1.05, 2.1, 1.2], [1.05, 1.3, 1.15]]), 1e-2
    )

    assert dataset.noise_map.native == pytest.approx(
        np.array([[0.229, 0.255, 0.25], [0.229, 0.324, 0.245], [0.229, 0.255, 0.240]]),
        1e-2,
    )

def test__via_image_from__disable_poisson_noise_in_noise_map(image_central_delta_3x3):
    image = image_central_delta_3x3 + 1.0

    simulator = aa.SimulatorImaging(
        exposure_time=20.0, add_poisson_noise_to_data=True,
        include_poisson_noise_in_noise_map=False,
        noise_if_add_noise_false=3.0,
        noise_seed=1
    )

    dataset = simulator.via_image_from(image=image)

    assert dataset.data.native == pytest.approx(
        np.array([[1.05, 1.3, 1.25], [1.05, 2.1, 1.2], [1.05, 1.3, 1.15]]), 1e-2
    )

    assert dataset.noise_map.native == pytest.approx(
        3.0 * np.ones((3, 3)),
        1e-2,
    )


def test__via_image_from__background_sky_on(image_central_delta_3x3):
    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        background_sky_level=16.0,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    dataset = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        dataset.data.native
        == np.array([[1.0, 5.0, 4.0], [1.0, 2.0, 1.0], [5.0, 2.0, 7.0]])
    ).all()

    assert dataset.noise_map.native[0, 0] == pytest.approx(4.12310, 1.0e-4)


def test__via_image_from__psf_and_noise_both_on(image_central_delta_3x3):
    image = image_central_delta_3x3 + 1.0

    psf = aa.Kernel2D.no_mask(
        values=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )

    simulator = aa.SimulatorImaging(
        exposure_time=20.0,
        psf=psf,
        add_poisson_noise_to_data=True,
        noise_seed=1,
        normalize_psf=False,
    )

    dataset = simulator.via_image_from(image=image)

    assert dataset.data.native == pytest.approx(
        np.array([[4.1, 6.65, 4.45], [6.15, 8.15, 6.5], [4.1, 6.7, 4.25]]), 1e-2
    )
