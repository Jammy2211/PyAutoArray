import autoarray as aa

import numpy as np
import pytest

from autoarray.mock.mock import MockFit


class TestFitImaging:
    def test__image_and_model_are_identical__no_masking__check_values_are_correct(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)
        noise_map = aa.Array2D.manual_mask(array=[2.0, 2.0, 2.0, 2.0], mask=mask)

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        masked_imaging = imaging.apply_mask(mask=mask)

        model_image = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

        fit = aa.FitData(
            data=masked_imaging.image,
            noise_map=masked_imaging.noise_map,
            model_data=model_image,
            mask=mask,
            use_mask_in_fit=False,
        )

        fit = aa.FitImaging(imaging=masked_imaging, fit=fit)

        assert (fit.mask == np.array([[False, False], [False, False]])).all()

        assert (fit.image.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (fit.image.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.noise_map.slim == np.array([2.0, 2.0, 2.0, 2.0])).all()
        assert (fit.noise_map.native == np.array([[2.0, 2.0], [2.0, 2.0]])).all()

        assert (fit.signal_to_noise_map.slim == np.array([0.5, 1.0, 1.5, 2.0])).all()
        assert (
            fit.signal_to_noise_map.native == np.array([[0.5, 1.0], [1.5, 2.0]])
        ).all()
        assert (
            fit.potential_chi_squared_map.slim == np.array([0.25, 1.0, 2.25, 4.0])
        ).all()

        assert (fit.model_image.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
        assert (fit.model_image.native == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        assert (fit.residual_map.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (fit.residual_map.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert (
            fit.normalized_residual_map.slim == np.array([0.0, 0.0, 0.0, 0.0])
        ).all()
        assert (
            fit.normalized_residual_map.native == np.array([[0.0, 0.0], [0.0, 0.0]])
        ).all()

        assert (fit.chi_squared_map.slim == np.array([0.0, 0.0, 0.0, 0.0])).all()
        assert (fit.chi_squared_map.native == np.array([[0.0, 0.0], [0.0, 0.0]])).all()

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__image_and_model_are_different__include_masking__check_values_are_correct(
        self,
    ):

        mask = aa.Mask2D.manual(
            mask=[[False, False], [True, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Array2D.manual_mask(array=[1.0, 2.0, 4.0], mask=mask)
        noise_map = aa.Array2D.manual_mask(array=[2.0, 2.0, 2.0], mask=mask)

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        model_image = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0], mask=mask)

        fit = aa.FitData(
            data=imaging.image,
            noise_map=imaging.noise_map,
            model_data=model_image,
            mask=mask,
            use_mask_in_fit=False,
        )

        fit = aa.FitImaging(imaging=imaging, fit=fit)

        assert (fit.mask == np.array([[False, False], [True, False]])).all()

        assert (fit.image.slim == np.array([1.0, 2.0, 4.0])).all()
        assert (fit.image.native == np.array([[1.0, 2.0], [0.0, 4.0]])).all()

        assert (fit.noise_map.slim == np.array([2.0, 2.0, 2.0])).all()
        assert (fit.noise_map.native == np.array([[2.0, 2.0], [0.0, 2.0]])).all()

        assert (fit.signal_to_noise_map.slim == np.array([0.5, 1.0, 2.0])).all()
        assert (
            fit.signal_to_noise_map.native == np.array([[0.5, 1.0], [0.0, 2.0]])
        ).all()

        assert (fit.model_image.slim == np.array([1.0, 2.0, 3.0])).all()
        assert (fit.model_image.native == np.array([[1.0, 2.0], [0.0, 3.0]])).all()

        assert (fit.residual_map.slim == np.array([0.0, 0.0, 1.0])).all()
        assert (fit.residual_map.native == np.array([[0.0, 0.0], [0.0, 1.0]])).all()

        assert (fit.normalized_residual_map.slim == np.array([0.0, 0.0, 0.5])).all()
        assert (
            fit.normalized_residual_map.native == np.array([[0.0, 0.0], [0.0, 0.5]])
        ).all()

        assert (fit.chi_squared_map.slim == np.array([0.0, 0.0, 0.25])).all()
        assert (fit.chi_squared_map.native == np.array([[0.0, 0.0], [0.0, 0.25]])).all()

        assert fit.chi_squared == 0.25
        assert fit.reduced_chi_squared == 0.25 / 3.0
        assert fit.noise_normalization == np.sum(np.log(2 * np.pi * noise_map ** 2.0))
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__image_and_model_are_identical__inversion_included__changes_certain_properties(
        self,
    ):

        mask = aa.Mask2D.manual(
            mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)
        noise_map = aa.Array2D.manual_mask(array=[2.0, 2.0, 2.0, 2.0], mask=mask)

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        masked_imaging = imaging.apply_mask(mask=mask)

        model_image = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

        inversion = MockFit(
            regularization_term=2.0,
            log_det_curvature_reg_matrix_term=3.0,
            log_det_regularization_matrix_term=4.0,
        )

        fit = aa.FitData(
            data=masked_imaging.image,
            noise_map=masked_imaging.noise_map,
            model_data=model_image,
            mask=mask,
            inversion=inversion,
            use_mask_in_fit=False,
        )

        fit = aa.FitImaging(imaging=masked_imaging, fit=fit)

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
        self,
    ):

        real_space_mask = aa.Mask2D.manual(
            mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Visibilities.manual_slim(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
        noise_map = aa.VisibilitiesNoiseMap.manual_slim(
            visibilities=[2.0 + 2.0j, 2.0 + 2.0j]
        )

        interferometer = aa.Interferometer(
            visibilities=data,
            noise_map=noise_map,
            uv_wavelengths=np.ones(shape=(2, 2)),
            real_space_mask=real_space_mask,
        )

        model_visibilities = aa.Visibilities.manual_slim(
            visibilities=[1.0 + 2.0j, 3.0 + 4.0j]
        )

        fit = aa.FitDataComplex(
            data=interferometer.visibilities,
            noise_map=interferometer.noise_map,
            model_data=model_visibilities,
            use_mask_in_fit=False,
        )

        fit = aa.FitInterferometer(interferometer=interferometer, fit=fit)

        assert (fit.visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()

        assert (fit.noise_map.slim == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()

        assert (
            fit.signal_to_noise_map.slim == np.array([0.5 + 1.0j, 1.5 + 2.0j])
        ).all()

        assert (fit.model_visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()

        assert (fit.residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()

        assert (
            fit.normalized_residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])
        ).all()

        assert (fit.chi_squared_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == pytest.approx(
            4.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
        )
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__visibilities_and_model_are_different__no_masking__check_values_are_correct(
        self,
    ):

        real_space_mask = aa.Mask2D.manual(
            mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Visibilities.manual_slim(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
        noise_map = aa.VisibilitiesNoiseMap.manual_slim(
            visibilities=[2.0 + 2.0j, 2.0 + 2.0j]
        )

        interferometer = aa.Interferometer(
            visibilities=data,
            noise_map=noise_map,
            uv_wavelengths=np.ones(shape=(2, 2)),
            real_space_mask=real_space_mask,
        )

        model_visibilities = aa.Visibilities.manual_slim(
            visibilities=[1.0 + 2.0j, 3.0 + 3.0j]
        )

        fit = aa.FitDataComplex(
            data=interferometer.visibilities,
            noise_map=interferometer.noise_map,
            model_data=model_visibilities,
            use_mask_in_fit=False,
        )

        fit = aa.FitInterferometer(interferometer=interferometer, fit=fit)

        assert (fit.visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()

        assert (fit.noise_map.slim == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()

        assert (
            fit.signal_to_noise_map.slim == np.array([0.5 + 1.0j, 1.5 + 2.0j])
        ).all()

        assert (fit.model_visibilities.slim == np.array([1.0 + 2.0j, 3.0 + 3.0j])).all()

        assert (fit.residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 1.0j])).all()

        assert (
            fit.normalized_residual_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.5j])
        ).all()

        assert (fit.chi_squared_map.slim == np.array([0.0 + 0.0j, 0.0 + 0.25j])).all()

        assert fit.chi_squared == 0.25
        assert fit.reduced_chi_squared == 0.25 / 2.0
        assert fit.noise_normalization == pytest.approx(
            4.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
        )
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

    def test__visibilities_and_model_are_identical__inversion_included__changes_certain_properties(
        self,
    ):

        real_space_mask = aa.Mask2D.manual(
            mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Visibilities.manual_slim(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])
        noise_map = aa.VisibilitiesNoiseMap.manual_slim(
            visibilities=[2.0 + 2.0j, 2.0 + 2.0j]
        )

        interferometer = aa.Interferometer(
            visibilities=data,
            noise_map=noise_map,
            uv_wavelengths=np.ones(shape=(2, 2)),
            real_space_mask=real_space_mask,
        )

        model_visibilities = aa.Visibilities.manual_slim(
            visibilities=[1.0 + 2.0j, 3.0 + 4.0j]
        )

        inversion = MockFit(
            regularization_term=2.0,
            log_det_curvature_reg_matrix_term=3.0,
            log_det_regularization_matrix_term=4.0,
        )

        fit = aa.FitDataComplex(
            data=interferometer.visibilities,
            noise_map=interferometer.noise_map,
            model_data=model_visibilities,
            inversion=inversion,
            use_mask_in_fit=False,
        )

        fit = aa.FitInterferometer(interferometer=interferometer, fit=fit)

        assert fit.chi_squared == 0.0
        assert fit.reduced_chi_squared == 0.0
        assert fit.noise_normalization == pytest.approx(
            4.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
        )
        assert fit.log_likelihood == -0.5 * (fit.chi_squared + fit.noise_normalization)

        assert fit.log_likelihood_with_regularization == -0.5 * (
            fit.chi_squared + 2.0 + fit.noise_normalization
        )
        assert fit.log_evidence == -0.5 * (
            fit.chi_squared + 2.0 + 3.0 - 4.0 + fit.noise_normalization
        )
        assert fit.figure_of_merit == fit.log_evidence

    def test__dirty_quantities(
        self, transformer_7x7_7, interferometer_7, fit_interferometer_7
    ):

        fit_interferometer_7.dataset.transformer = transformer_7x7_7

        dirty_image = transformer_7x7_7.image_from(
            visibilities=interferometer_7.visibilities
        )
        assert (fit_interferometer_7.dirty_image == dirty_image).all()

        dirty_noise_map = transformer_7x7_7.image_from(
            visibilities=interferometer_7.noise_map
        )
        assert (fit_interferometer_7.dirty_noise_map == dirty_noise_map).all()

        dirty_signal_to_noise_map = transformer_7x7_7.image_from(
            visibilities=interferometer_7.signal_to_noise_map
        )
        assert (
            fit_interferometer_7.dirty_signal_to_noise_map == dirty_signal_to_noise_map
        ).all()

        dirty_model_image = transformer_7x7_7.image_from(
            visibilities=fit_interferometer_7.model_visibilities
        )
        assert (fit_interferometer_7.dirty_model_image == dirty_model_image).all()

        dirty_residual_map = transformer_7x7_7.image_from(
            visibilities=fit_interferometer_7.residual_map
        )
        assert (fit_interferometer_7.dirty_residual_map == dirty_residual_map).all()

        dirty_normalized_residual_map = transformer_7x7_7.image_from(
            visibilities=fit_interferometer_7.normalized_residual_map
        )
        assert (
            fit_interferometer_7.dirty_normalized_residual_map
            == dirty_normalized_residual_map
        ).all()

        dirty_chi_squared_map = transformer_7x7_7.image_from(
            visibilities=fit_interferometer_7.chi_squared_map
        )
        assert (
            fit_interferometer_7.dirty_chi_squared_map == dirty_chi_squared_map
        ).all()


class TestProfilingDict:
    def test__profiles_appropriate_functions(self):

        mask = aa.Mask2D.manual(
            mask=[[False, False], [False, False]], sub_size=1, pixel_scales=(1.0, 1.0)
        )

        data = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)
        noise_map = aa.Array2D.manual_mask(array=[2.0, 2.0, 2.0, 2.0], mask=mask)

        imaging = aa.Imaging(image=data, noise_map=noise_map)

        masked_imaging = imaging.apply_mask(mask=mask)

        model_image = aa.Array2D.manual_mask(array=[1.0, 2.0, 3.0, 4.0], mask=mask)

        profiling_dict = {}

        fit = aa.FitData(
            data=masked_imaging.image,
            noise_map=masked_imaging.noise_map,
            model_data=model_image,
            mask=mask,
            use_mask_in_fit=False,
            profiling_dict=profiling_dict,
        )

        fit = aa.FitImaging(
            imaging=masked_imaging, fit=fit, profiling_dict=profiling_dict
        )
        fit.figure_of_merit

        assert "figure_of_merit_0" in fit.profiling_dict
