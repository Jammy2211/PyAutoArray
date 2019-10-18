import autoarray as aa

import numpy as np


class TestDataFit:
    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        mask = aa.mask.manual(
            mask_2d=np.array([[False, False], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.array_masked.manual_1d(array=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask)
        noise_map = aa.array_masked.manual_1d(array=np.array([2.0, 2.0, 2.0, 2.0]), mask=mask)
        model_data = aa.array_masked.manual_1d(array=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask)

        fit = aa.fit(
            mask=mask, data=data, noise_map=noise_map, model_data=model_data
        )

        assert (fit.mask == np.array([[False, False], [False, False]])).all()

        assert (fit.data.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (fit.data.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.noise_map.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (fit.noise_map.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        assert (fit.signal_to_noise_map.in_1d == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (fit.signal_to_noise_map.in_2d == np.array([[0.5, 1.0], [1.5, 2.0]])).all()

        assert (fit.model_data.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (fit.model_data.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.residual_map.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (fit.residual_map.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert (
            fit.normalized_residual_map.in_1d == np.array([0.0, 0.0, 0.0, 0.0])
        ).all()
        assert (
            fit.normalized_residual_map.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])
        ).all()

        assert (fit.chi_squared_map.in_1d == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (fit.chi_squared_map.in_2d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == np.sum(
            np.log(2 * np.pi * noise_map ** 2.0)
        )
        assert fit.likelihood == -0.5 * (
            fit.chi_squared + fit.noise_normalization
        )

    def test__image_and_model_mismatch__no_masking__check_values_are_correct(self):

        mask = aa.mask.manual(
            mask_2d=np.array([[False, True], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.array_masked.manual_1d(array=np.array([1.0, 2.0, 3.0]), mask=mask)
        noise_map = aa.array_masked.manual_1d(array=np.array([2.0, 2.0, 2.0]), mask=mask)
        model_data = aa.array_masked.manual_1d(array=np.array([1.0, 1.0, 1.0]), mask=mask)

        fit = aa.fit(
            data=data, noise_map=noise_map, mask=mask, model_data=model_data
        )

        assert (fit.mask == np.array([[False, True], [False, False]])).all()

        assert (fit.data.in_1d == np.array([1.0, 2.0, 3.0])).all()
        assert (fit.data.in_2d == np.array([[1.0, 0.0], [2.0, 3.0]])).all()

        assert (fit.noise_map.in_1d == np.array([2.0, 2.0, 2.0])).all()
        assert (fit.noise_map.in_2d == np.array([[2.0, 0.0], [2.0, 2.0]])).all()

        assert (fit.signal_to_noise_map.in_1d == np.array([0.5, 1.0, 1.5])).all()
        assert (fit.signal_to_noise_map.in_2d == np.array([[0.5, 0.0], [1.0, 1.5]])).all()

        assert (fit.model_data.in_1d == np.array([1.0, 1.0, 1.0, ])).all()
        assert (fit.model_data.in_2d == np.array([[1.0, 0.0], [1.0, 1.0]])).all()

        assert (fit.residual_map.in_1d == np.array([0.0, 1.0, 2.0, ])).all()
        assert (fit.residual_map.in_2d == np.array([[0.0, 0.0], [1.0, 2.0]])).all()

        assert (
            fit.normalized_residual_map.in_1d
            == np.array([0.0, (1.0 / 2.0), (2.0 / 2.0)])
        ).all()
        assert (
            fit.normalized_residual_map.in_2d
            == np.array([[0.0, 0.0], [(1.0 / 2.0), (2.0 / 2.0)]])
        ).all()

        assert (
            fit.chi_squared_map.in_1d
            == np.array(
                [0.0, (1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0]
            )
        ).all()
        assert (
            fit.chi_squared_map.in_2d
            == np.array(
                [[0.0, 0.0], [(1.0 / 2.0) ** 2.0, (2.0 / 2.0) ** 2.0]]
            )
        ).all()

        assert (
            fit.chi_squared
            == (1.0 / 2.0) ** 2.0 + (2.0 / 2.0) ** 2.0
        )
        assert fit.reduced_chi_squared == fit.chi_squared / 3.0
        assert fit.noise_normalization == np.sum(
            np.log(2 * np.pi * noise_map ** 2.0)
        )
        assert fit.likelihood == -0.5 * (
            fit.chi_squared + fit.noise_normalization
        )
