import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="image_central_delta_3x3")
def make_array_2d_7x7():
    return aa.Array2D.manual_native(
        array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=0.1,
    )


def test__via_image_from__all_features_off(image_central_delta_3x3):

    simulator = aa.SimulatorImaging(exposure_time=1.0, add_poisson_noise=False)

    imaging = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        imaging.image.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert imaging.pixel_scales == (0.1, 0.1)


def test__via_image_from__noise_off__noise_map_is_noise_value(image_central_delta_3x3):

    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        add_poisson_noise=False,
        noise_if_add_noise_false=0.2,
        noise_seed=1,
    )

    imaging = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        imaging.image.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert (imaging.noise_map.native == 0.2 * np.ones((3, 3))).all()


def test__via_image_from__psf_blurs_image_with_edge_trimming(image_central_delta_3x3):

    psf = aa.Kernel2D.manual_native(
        array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )

    simulator = aa.SimulatorImaging(
        exposure_time=1.0, psf=psf, add_poisson_noise=False, normalize_psf=False
    )

    imaging = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        imaging.image.native
        == np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])
    ).all()


def test__via_image_from__setup_with_noise(image_central_delta_3x3):

    simulator = aa.SimulatorImaging(
        exposure_time=20.0, add_poisson_noise=True, noise_seed=1
    )

    imaging = simulator.via_image_from(image=image_central_delta_3x3)

    assert imaging.image.native == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 0.0]]), 1e-2
    )

    # Because of the value is 1.05, the estimated Poisson noise_map_1d is:
    # sqrt((1.05 * 20))/20 = 0.2291

    assert imaging.noise_map.native == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.2291, 0.0], [0.0, 0.0, 0.0]]), 1e-2
    )


def test__via_image_from__background_sky_on(image_central_delta_3x3):

    simulator = aa.SimulatorImaging(
        exposure_time=1.0,
        background_sky_level=16.0,
        add_poisson_noise=True,
        noise_seed=1,
    )

    imaging = simulator.via_image_from(image=image_central_delta_3x3)

    assert (
        imaging.image.native
        == np.array([[1.0, 5.0, 4.0], [1.0, 2.0, 1.0], [5.0, 2.0, 7.0]])
    ).all()

    assert imaging.noise_map.native[0, 0] == pytest.approx(4.12310, 1.0e-4)


def test__via_image_from__psf_and_noise_both_on(image_central_delta_3x3):

    psf = aa.Kernel2D.manual_native(
        array=np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]]),
        pixel_scales=1.0,
    )

    simulator = aa.SimulatorImaging(
        exposure_time=20.0,
        psf=psf,
        add_poisson_noise=True,
        noise_seed=1,
        normalize_psf=False,
    )

    imaging = simulator.via_image_from(image=image_central_delta_3x3)

    assert imaging.image.native == pytest.approx(
        np.array([[0.0, 1.05, 0.0], [1.3, 2.35, 1.05], [0.0, 1.05, 0.0]]), 1e-2
    )
