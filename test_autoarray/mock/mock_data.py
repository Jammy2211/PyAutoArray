import numpy as np

import autoarray as aa
from autoarray.data import imaging, interferometer
from autoarray.operators import fourier_transform


class MockImage(object):
    def __new__(cls, shape, value, pixel_scales=1.0):
        return aa.array.manual_2d(
            array=value * np.ones(shape=shape), pixel_scales=pixel_scales
        )


class MockNoiseMap(object):
    def __new__(cls, shape, value, pixel_scales=1.0):
        return aa.array.manual_2d(
            array=value * np.ones(shape=shape), pixel_scales=pixel_scales
        )


class MockBackgroundNoiseMap(object):
    def __new__(cls, shape, value, pixel_scales=1.0):
        return aa.array.manual_2d(
            array=value * np.ones(shape=shape), pixel_scales=pixel_scales
        )


class MockPoissonNoiseMap(object):
    def __new__(cls, shape, value, pixel_scales=1.0):
        return aa.array.manual_2d(
            array=value * np.ones(shape=shape), pixel_scales=pixel_scales
        )


class MockExposureTimeMap(object):
    def __new__(cls, shape, value, pixel_scales=1.0):
        return aa.array.manual_2d(
            array=value * np.ones(shape=shape), pixel_scales=pixel_scales
        )


class MockBackgrondSkyMap(object):
    def __new__(cls, shape, value, pixel_scales=1.0):
        return aa.array.manual_2d(
            array=value * np.ones(shape=shape), pixel_scales=pixel_scales
        )


class MockPSF(object):
    def __new__(cls, shape_2d, value, pixel_scales=1.0, *args, **kwargs):
        return aa.kernel.full(
            fill_value=value,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            origin=(0.0, 0.0),
        )


class MockImage1D(np.ndarray):
    def __new__(cls, shape, value, pixel_scales=1.0):
        array = value * np.ones(shape=shape)

        obj = np.array(array, dtype="float64").view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = (0.0, 0.0)

        return obj


class MockNoiseMap1D(np.ndarray):
    def __new__(cls, shape, value, pixel_scales=1.0):
        array = value * np.ones(shape=shape)

        obj = np.array(array, dtype="float64").view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = (0.0, 0.0)

        return obj


class MockImaging(imaging.Imaging):
    def __init__(
        self,
        image,
        pixel_scales,
        psf,
        noise_map,
        background_noise_map,
        poisson_noise_map,
        exposure_time_map,
        background_sky_map,
        name,
    ):
        super(MockImaging, self).__init__(
            image=image,
            pixel_scales=pixel_scales,
            psf=psf,
            noise_map=noise_map,
            background_noise_map=background_noise_map,
            poisson_noise_map=poisson_noise_map,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            name=name,
        )


class MockPrimaryBeam(object):
    def __new__(cls, shape_2d, value, pixel_scales=1.0, *args, **kwargs):
        return aa.kernel.full(
            fill_value=value,
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            origin=(0.0, 0.0),
        )


class MockVisibilities(np.ndarray):
    def __new__(cls, shape, value, pixel_scales=1.0):
        array = value * np.ones(shape=(shape, 2))

        obj = np.array(array, dtype="float64").view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = (0.0, 0.0)

        return obj


class MockVisibilitiesNoiseMap(np.ndarray):
    def __new__(cls, shape, value, pixel_scales=1.0):
        array = value * np.ones(shape=shape)

        obj = np.array(array, dtype="float64").view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = (0.0, 0.0)

        return obj


class MockUVWavelengths(np.ndarray):
    def __new__(cls, shape, value, pixel_scales=1.0):
        array = np.array(
            [
                [-55636.4609375, 171376.90625],
                [-6903.21923828, 51155.578125],
                [-63488.4140625, 4141.28369141],
                [55502.828125, 47016.7265625],
                [54160.75390625, -99354.1796875],
                [-9327.66308594, -95212.90625],
                [0.0, 0.0],
            ]
        )

        obj = np.array(array, dtype="float64").view(cls)
        obj.pixel_scales = pixel_scales
        obj.origin = (0.0, 0.0)

        return obj


class MockInterferometer(interferometer.Interferometer):
    def __init__(
        self, shape_2d, visibilities, pixel_scales, primary_beam, noise_map, uv_wavelengths
    ):
        super(MockInterferometer, self).__init__(
            shape_2d=shape_2d,
            visibilities=visibilities,
            pixel_scales=pixel_scales,
            primary_beam=primary_beam,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
        )


class MockTransformer(fourier_transform.Transformer):
    def __init__(self, uv_wavelengths, grid_radians):
        super(MockTransformer, self).__init__(
            uv_wavelengths=uv_wavelengths, grid_radians=grid_radians
        )
