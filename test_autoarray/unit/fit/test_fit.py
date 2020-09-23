import autoarray as aa

import numpy as np

from test_autoarray import mock


class TestFitImaging:
    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        mask = aa.Mask.manual(
            mask=np.array([[False, False], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.Array.manual_mask(array=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask)
        noise_map = aa.Array.manual_mask(
            array=np.array([2.0, 2.0, 2.0, 2.0]), mask=mask
        )

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        masked_imaging = aa.MaskedImaging(imaging=imaging, mask=mask)

        model_image = aa.Array.manual_mask(
            array=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask
        )

        fit = aa.FitImaging(masked_imaging=masked_imaging, model_image=model_image)

        assert (fit.mask == np.array([[False, False], [False, False]])).all()

        assert (fit.image.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (fit.image.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.noise_map.in_1d == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (fit.noise_map.in_2d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        assert (fit.signal_to_noise_map.in_1d == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (
            fit.signal_to_noise_map.in_2d == np.array([[0.5, 1.0], [1.5, 2.0]])
        ).all()

        assert (fit.model_image.in_1d == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (fit.model_image.in_2d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

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
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__image_and_model_are_different__inclue_masking__check_values_are_correct(
        self
    ):

        mask = aa.Mask.manual(
            mask=np.array([[False, False], [True, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.Array.manual_mask(array=np.array([1.0, 2.0, 4.0]), mask=mask)
        noise_map = aa.Array.manual_mask(array=np.array([2.0, 2.0, 2.0]), mask=mask)

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        masked_imaging = aa.MaskedImaging(imaging=imaging, mask=mask)

        model_image = aa.Array.manual_mask(array=np.array([1.0, 2.0, 3.0]), mask=mask)

        fit = aa.FitImaging(masked_imaging=masked_imaging, model_image=model_image)

        assert (fit.mask == np.array([[False, False], [True, False]])).all()

        assert (fit.image.in_1d == np.array([1.0, 2.0, 4.0])).all()
        assert (fit.image.in_2d == np.array([[1.0, 2.0], [0.0, 4.0]])).all()

        assert (fit.noise_map.in_1d == np.array([2.0, 2.0, 2.0])).all()
        assert (fit.noise_map.in_2d == np.array([[2.0, 2.0], [0.0, 2.0]])).all()

        assert (fit.signal_to_noise_map.in_1d == np.array([0.5, 1.0, 2.0])).all()
        assert (
            fit.signal_to_noise_map.in_2d == np.array([[0.5, 1.0], [0.0, 2.0]])
        ).all()

        assert (fit.model_image.in_1d == np.array([1.0, 2.0, 3.0])).all()
        assert (fit.model_image.in_2d == np.array([[1.0, 2.0], [0.0, 3.0]])).all()

        assert (fit.residual_map.in_1d == np.array([0.0, 0.0, 1.0])).all()
        assert (fit.residual_map.in_2d == np.array([[0.0, 0.0], [0.0, 1.0]])).all()

        assert (fit.normalized_residual_map.in_1d == np.array([0.0, 0.0, 0.5])).all()
        assert (
            fit.normalized_residual_map.in_2d == np.array([[0.0, 0.0], [0.0, 0.5]])
        ).all()

        assert (fit.chi_squared_map.in_1d == np.array([0.0, 0.0, 0.25])).all()
        assert (fit.chi_squared_map.in_2d == np.array([[0.0, 0.0], [0.0, 0.25]])).all()

        assert fit.chi_squared == 0.25
        assert fit.reduced_chi_squared == 0.25 / 3.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__image_and_model_are_identical__inversion_included__changes_certain_properties(
        self
    ):

        mask = aa.Mask.manual(
            mask=np.array([[False, False], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.Array.manual_mask(array=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask)
        noise_map = aa.Array.manual_mask(
            array=np.array([2.0, 2.0, 2.0, 2.0]), mask=mask
        )

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        masked_imaging = aa.MaskedImaging(imaging=imaging, mask=mask)

        model_image = aa.Array.manual_mask(
            array=np.array([1.0, 2.0, 3.0, 4.0]), mask=mask
        )

        inversion = mock.MockFitInversion(
            regularization_term=2.0,
            log_det_curvature_reg_matrix_term=3.0,
            log_det_regularization_matrix_term=4.0,
        )

        fit = aa.FitImaging(
            masked_imaging=masked_imaging, model_image=model_image, inversion=inversion
        )

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

        assert fit.log_likelihood_with_regularization == -0.5 * (
            fit.chi_squared + 2.0 + fit.noise_normalization
        )
        assert fit.log_evidence == -0.5 * (
            fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
        )
        assert fit.figure_of_merit == fit.log_evidence


class TestFitInterferometer:
    def test__visibilities_and_model_are_identical__no_masking__check_values_are_correct(
        self
    ):

        visibilities_mask = np.full(fill_value=False, shape=(2, 2))

        real_space_mask = aa.Mask.manual(
            mask=np.array([[False, False], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.Visibilities.manual_1d(visibilities=[[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.VisibilitiesNoiseMap.manual_1d(
            visibilities=[[2.0, 2.0], [2.0, 2.0]]
        )

        interferometer = aa.Interferometer(
            visibilities=data, noise_map=noise_map, uv_wavelengths=np.ones(shape=(2, 2))
        )

        masked_interferometer = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        model_visibilities = aa.Visibilities.manual_1d(
            visibilities=[[1.0, 2.0], [3.0, 4.0]]
        )

        fit = aa.FitInterferometer(
            masked_interferometer=masked_interferometer,
            model_visibilities=model_visibilities,
        )

        assert (
            fit.visibilities_mask == np.array([[False, False], [False, False]])
        ).all()

        assert (fit.visibilities.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.noise_map.in_1d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        assert (
            fit.signal_to_noise_map.in_1d == np.array([[0.5, 1.0], [1.5, 2.0]])
        ).all()

        assert (
            fit.model_visibilities.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])
        ).all()

        assert (fit.residual_map.in_1d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert (
            fit.normalized_residual_map.in_1d == np.array([[0.0, 0.0], [0.0, 0.0]])
        ).all()

        assert (fit.chi_squared_map.in_1d == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__visibilities_and_model_are_different__no_masking__check_values_are_correct(
        self
    ):

        visibilities_mask = np.full(fill_value=False, shape=(2, 2))

        real_space_mask = aa.Mask.manual(
            mask=np.array([[False, False], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.Visibilities.manual_1d(visibilities=[[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.VisibilitiesNoiseMap.manual_1d(
            visibilities=[[2.0, 2.0], [2.0, 2.0]]
        )

        interferometer = aa.Interferometer(
            visibilities=data, noise_map=noise_map, uv_wavelengths=np.ones(shape=(2, 2))
        )

        masked_interferometer = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        model_visibilities = aa.Visibilities.manual_1d(
            visibilities=[[1.0, 2.0], [3.0, 3.0]]
        )

        fit = aa.FitInterferometer(
            masked_interferometer=masked_interferometer,
            model_visibilities=model_visibilities,
        )

        assert (
            fit.visibilities_mask == np.array([[False, False], [False, False]])
        ).all()

        assert (fit.visibilities.in_1d == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.noise_map.in_1d == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        assert (
            fit.signal_to_noise_map.in_1d == np.array([[0.5, 1.0], [1.5, 2.0]])
        ).all()

        assert (
            fit.model_visibilities.in_1d == np.array([[1.0, 2.0], [3.0, 3.0]])
        ).all()

        assert (fit.residual_map.in_1d == np.array([[0.0, 0.0], [0.0, 1.0]])).all()

        assert (
            fit.normalized_residual_map.in_1d == np.array([[0.0, 0.0], [0.0, 0.5]])
        ).all()

        assert (fit.chi_squared_map.in_1d == np.array([[0.0, 0.0], [0.0, 0.25]])).all()

        assert fit.chi_squared == 0.25
        assert fit.reduced_chi_squared == 0.25 / 4.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__visibilities_and_model_are_identical__inversion_included__changes_certain_properties(
        self
    ):

        visibilities_mask = np.full(fill_value=False, shape=(2, 2))

        real_space_mask = aa.Mask.manual(
            mask=np.array([[False, False], [False, False]]),
            sub_size=1,
            pixel_scales=(1.0, 1.0),
        )

        data = aa.Visibilities.manual_1d(visibilities=[[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.VisibilitiesNoiseMap.manual_1d(
            visibilities=[[2.0, 2.0], [2.0, 2.0]]
        )

        interferometer = aa.Interferometer(
            visibilities=data, noise_map=noise_map, uv_wavelengths=np.ones(shape=(2, 2))
        )

        masked_interferometer = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        model_visibilities = aa.Visibilities.manual_1d(
            visibilities=[[1.0, 2.0], [3.0, 4.0]]
        )

        inversion = mock.MockFitInversion(
            regularization_term=2.0,
            log_det_curvature_reg_matrix_term=3.0,
            log_det_regularization_matrix_term=4.0,
        )

        fit = aa.FitInterferometer(
            masked_interferometer=masked_interferometer,
            model_visibilities=model_visibilities,
            inversion=inversion,
        )

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

        assert fit.log_likelihood_with_regularization == -0.5 * (
            fit.chi_squared + 2.0 + fit.noise_normalization
        )
        assert fit.log_evidence == -0.5 * (
            fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
        )
        assert fit.figure_of_merit == fit.log_evidence
