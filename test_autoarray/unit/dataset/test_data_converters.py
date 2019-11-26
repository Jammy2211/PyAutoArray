import numpy as np
import autoarray as aa


class TestNoiseMapFromWeightMap:
    def test__weight_map_no_zeros__uses_1_over_sqrt_value(self):
        weight_map = aa.array.manual_2d([[1.0, 4.0, 16.0], [1.0, 4.0, 16.0]])

        noise_map = aa.data_converter.noise_map_from_weight_map(
            weight_map=weight_map.in_2d
        )

        assert (noise_map.in_2d == np.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]])).all()

    def test__weight_map_no_zeros__zeros_set_to_10000000(self):

        weight_map = aa.array.manual_2d([[1.0, 4.0, 0.0], [1.0, 4.0, 16.0]])

        noise_map = aa.data_converter.noise_map_from_weight_map(weight_map=weight_map)

        assert (
            noise_map.in_2d == np.array([[1.0, 0.5, 1.0e8], [1.0, 0.5, 0.25]])
        ).all()


class TestFromInverseAbstractNoiseMap:
    def test__inverse_noise_map_no_zeros__uses_1_over_value(self):
        inverse_noise_map = aa.array.manual_2d([[1.0, 4.0, 16.0], [1.0, 4.0, 16.0]])

        noise_map = aa.data_converter.noise_map_from_inverse_noise_map(
            inverse_noise_map=inverse_noise_map
        )

        assert (
            noise_map.in_2d == np.array([[1.0, 0.25, 0.0625], [1.0, 0.25, 0.0625]])
        ).all()


class TestFromImageAndBackgroundNoiseMap:
    def test__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(
        self
    ):
        imaging = aa.array.ones(shape_2d=(2, 2))
        background_noise_map = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            gain=1.0,
            exposure_time_map=exposure_time_map,
        )

        assert (
            noise_map.in_2d
            == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(2.0), np.sqrt(2.0)]])
        ).all()

    def test__image_all_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_3s(
        self
    ):
        imaging = aa.array.full(fill_value=2.0, shape_2d=(2, 2))
        background_noise_map = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            gain=1.0,
            exposure_time_map=exposure_time_map,
        )

        assert (
            noise_map.in_2d
            == np.array([[np.sqrt(3.0), np.sqrt(3.0)], [np.sqrt(3.0), np.sqrt(3.0)]])
        ).all()

    def test__image_all_1s__bg_noise_all_2s__exposure_time_all_1s__noise_map_all_sqrt_5s(
        self
    ):
        imaging = aa.array.ones(shape_2d=(2, 2))
        background_noise_map = aa.array.full(fill_value=2.0, shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            gain=1.0,
            exposure_time_map=exposure_time_map,
        )

        assert (
            noise_map.in_2d
            == np.array([[np.sqrt(5.0), np.sqrt(5.0)], [np.sqrt(5.0), np.sqrt(5.0)]])
        ).all()

    def test__image_all_1s__bg_noise_all_1s__exposure_time_all_2s__noise_map_all_sqrt_6s_over_2(
        self
    ):
        imaging = aa.array.ones(shape_2d=(2, 2))
        background_noise_map = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.full(fill_value=2.0, shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            gain=1.0,
            exposure_time_map=exposure_time_map,
        )

        assert (
            noise_map.in_2d
            == np.array(
                [
                    [np.sqrt(6.0) / 2.0, np.sqrt(6.0) / 2.0],
                    [np.sqrt(6.0) / 2.0, np.sqrt(6.0) / 2.0],
                ]
            )
        ).all()

    def test__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_1s(
        self
    ):
        imaging = aa.array.full(fill_value=-2.0, shape_2d=(2, 2))
        background_noise_map = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            gain=1.0,
            exposure_time_map=exposure_time_map,
        )

        assert (noise_map.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

    def test__same_as_above__use_different_values_in_different_array_elemets(self):
        imaging = aa.array.manual_2d([[1.0, 2.0], [2.0, 3.0]])
        background_noise_map = aa.array.manual_2d([[1.0, 1.0], [2.0, 3.0]])
        exposure_time_map = aa.array.manual_2d([[4.0, 3.0], [2.0, 1.0]])

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            gain=1.0,
            exposure_time_map=exposure_time_map,
        )

        assert (
            noise_map.in_2d
            == np.array(
                [
                    [np.sqrt(20.0) / 4.0, np.sqrt(15.0) / 3.0],
                    [np.sqrt(20.0) / 2.0, np.sqrt(12.0)],
                ]
            )
        ).all()

    def test__convert_from_electrons__image_all_1s__bg_noise_all_1s__exposure_time_all_1s__noise_map_all_sqrt_2s(
        self
    ):
        imaging = aa.array.ones(shape_2d=(2, 2))
        background_noise_map = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            gain=2.0,
            convert_from_electrons=True,
        )

        assert (
            noise_map.in_2d
            == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(2.0), np.sqrt(2.0)]])
        ).all()

    def test__convert_from_electrons__image_all_negative_2s__bg_noise_all_1s__exposure_time_all_10s__noise_map_all_1s(
        self
    ):
        imaging = aa.array.full(fill_value=-2.0, shape_2d=(2, 2))
        background_noise_map = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.full(fill_value=10.0, shape_2d=(2, 2))

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            gain=1.0,
            convert_from_electrons=True,
        )

        assert (noise_map.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

    def test__convert_from_electrons__same_as_above__use_different_values_in_different_array_elemets(
        self
    ):
        imaging = aa.array.manual_2d([[1.0, 2.0], [2.0, 3.0]])
        background_noise_map = aa.array.manual_2d([[1.0, 1.0], [2.0, 3.0]])
        exposure_time_map = aa.array.manual_2d([[10.0, 11.0], [12.0, 13.0]])

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            gain=4.0,
            convert_from_electrons=True,
        )

        assert (
            noise_map.in_2d
            == np.array([[np.sqrt(2.0), np.sqrt(3.0)], [np.sqrt(6.0), np.sqrt(12.0)]])
        ).all()

    def test__convert_from_adus__same_as_above__gain_is_1__same_values(self):
        imaging = aa.array.manual_2d([[1.0, 2.0], [2.0, 3.0]])
        background_noise_map = aa.array.manual_2d([[1.0, 1.0], [2.0, 3.0]])
        exposure_time_map = aa.array.manual_2d([[10.0, 11.0], [12.0, 13.0]])

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            gain=1.0,
            convert_from_adus=True,
        )

        assert (
            noise_map.in_2d
            == np.array([[np.sqrt(2.0), np.sqrt(3.0)], [np.sqrt(6.0), np.sqrt(12.0)]])
        ).all()

    def test__convert_from_adus__same_as_above__gain_is_2__values_change(self):
        imaging = aa.array.manual_2d([[1.0, 2.0], [2.0, 3.0]])
        background_noise_map = aa.array.manual_2d([[1.0, 1.0], [2.0, 3.0]])
        exposure_time_map = aa.array.manual_2d([[10.0, 11.0], [12.0, 13.0]])

        noise_map = aa.data_converter.noise_map_from_image_and_background_noise_map(
            image=imaging,
            background_noise_map=background_noise_map,
            exposure_time_map=exposure_time_map,
            gain=2.0,
            convert_from_adus=True,
        )

        assert (
            noise_map.in_2d
            == np.array(
                [
                    [np.sqrt(6.0) / 2.0, np.sqrt(8.0) / 2.0],
                    [np.sqrt(20.0) / 2.0, np.sqrt(42.0) / 2.0],
                ]
            )
        ).all()


class TestFromImageAndExposureTimeMap:
    def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s(self):
        imaging = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        poisson_noise_map = aa.data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=imaging, exposure_time_map=exposure_time_map, gain=1.0
        )

        assert (poisson_noise_map.in_2d == np.array([[1.0, 1.0], [1.0, 1.0]])).all()

    def test__image_all_2s_and_3s__exposure_time_all_1s__noise_map_all_sqrt_2s_and_3s(
        self
    ):
        imaging = aa.array.manual_2d([[2.0, 2.0], [3.0, 3.0]])
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        poisson_noise_map = aa.data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=imaging, exposure_time_map=exposure_time_map, gain=1.0
        )

        assert (
            poisson_noise_map.in_2d
            == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]])
        ).all()

    def test__image_all_1s__exposure_time_all__2s_and_3s__noise_map_all_sqrt_2s_and_3s(
        self
    ):
        imaging = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.manual_2d([[2.0, 2.0], [3.0, 3.0]])

        poisson_noise_map = aa.data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=imaging, exposure_time_map=exposure_time_map, gain=1.0
        )

        assert (
            poisson_noise_map.in_2d
            == np.array(
                [
                    [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
                    [np.sqrt(3.0) / 3.0, np.sqrt(3.0) / 3.0],
                ]
            )
        ).all()

    def test__image_all_1s__exposure_time_all_1s__noise_map_all_1s__gain_is_2__ignores_gain(
        self
    ):
        imaging = aa.array.ones(shape_2d=(2, 2))
        exposure_time_map = aa.array.ones(shape_2d=(2, 2))

        poisson_noise_map = aa.data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=imaging, exposure_time_map=exposure_time_map, gain=2.0
        )

        assert (
            poisson_noise_map.in_2d
            == np.array([[np.sqrt(1.0), np.sqrt(1.0)], [np.sqrt(1.0), np.sqrt(1.0)]])
        ).all()

    def test__convert_from_electrons_is_true__image_already_in_counts_so_exposure_time_ignored(
        self
    ):
        imaging = aa.array.manual_2d([[2.0, 2.0], [3.0, 3.0]])
        exposure_time_map = aa.array.full(fill_value=10.0, shape_2d=(2, 2))

        poisson_noise_map = aa.data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=imaging,
            exposure_time_map=exposure_time_map,
            gain=4.0,
            convert_from_electrons=True,
        )

        assert (
            poisson_noise_map.in_2d
            == np.array([[np.sqrt(2.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(3.0)]])
        ).all()

    def test__same_as_above__convert_from_adus__includes_gain_multiplication(self):
        imaging = aa.array.manual_2d([[2.0, 2.0], [3.0, 3.0]])
        exposure_time_map = aa.array.full(fill_value=10.0, shape_2d=(2, 2))

        poisson_noise_map = aa.data_converter.poisson_noise_map_from_image_and_exposure_time_map(
            image=imaging,
            exposure_time_map=exposure_time_map,
            gain=2.0,
            convert_from_adus=True,
        )

        assert (
            poisson_noise_map.in_2d
            == np.array(
                [
                    [np.sqrt(2.0 * 2.0) / 2.0, np.sqrt(2.0 * 2.0) / 2.0],
                    [np.sqrt(2.0 * 3.0) / 2.0, np.sqrt(2.0 * 3.0) / 2.0],
                ]
            )
        ).all()
