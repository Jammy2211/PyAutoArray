import numpy as np

import autoarray as aa


def test__set_w_tilde():
    # fit inversion is None, so no need to bother with w_tilde.

    fit_0 = aa.m.MockFitImaging(inversion=None)
    fit_1 = aa.m.MockFitImaging(inversion=None)

    preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fit are different but there is an inversion, so we should not preload w_tilde and use w_tilde.

    inversion = aa.m.MockInversion(linear_obj_list=[aa.m.MockMapper()])

    fit_0 = aa.m.MockFitImaging(
        inversion=inversion,
        noise_map=aa.Array2D.zeros(shape_native=(3, 1), pixel_scales=0.1),
    )
    fit_1 = aa.m.MockFitImaging(
        inversion=inversion,
        noise_map=aa.Array2D.ones(shape_native=(3, 1), pixel_scales=0.1),
    )

    preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    assert preloads.w_tilde is None
    assert preloads.use_w_tilde is False

    # Noise maps of fits are the same so preload w_tilde and use it.

    noise_map = aa.Array2D.ones(shape_native=(5, 5), pixel_scales=0.1)

    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        ),
        pixel_scales=noise_map.pixel_scales,
    )

    dataset = aa.m.MockDataset(psf=aa.Kernel2D.no_blur(pixel_scales=1.0), mask=mask)

    fit_0 = aa.m.MockFitImaging(
        inversion=inversion, dataset=dataset, noise_map=noise_map
    )
    fit_1 = aa.m.MockFitImaging(
        inversion=inversion, dataset=dataset, noise_map=noise_map
    )

    preloads = aa.Preloads(w_tilde=1, use_w_tilde=1)
    preloads.set_w_tilde_imaging(fit_0=fit_0, fit_1=fit_1)

    (
        curvature_preload,
        indexes,
        lengths,
    ) = aa.util.inversion_imaging.w_tilde_curvature_preload_imaging_from(
        noise_map_native=np.array(fit_0.noise_map.native),
        kernel_native=np.array(fit_0.dataset.psf.native),
        native_index_for_slim_index=np.array(
            fit_0.dataset.mask.derive_indexes.native_for_slim
        ),
    )

    assert preloads.w_tilde.curvature_preload[0] == curvature_preload[0]
    assert preloads.w_tilde.indexes[0] == indexes[0]
    assert preloads.w_tilde.lengths[0] == lengths[0]
    assert preloads.w_tilde.noise_map_value == 1.0
    assert preloads.use_w_tilde == True


