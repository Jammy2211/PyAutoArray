import os
from os import path
import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray.structures import visibilities as vis

test_data_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "visibilities"
)


class TestVisibilitiesAPI:
    def test__manual__makes_visibilities_without_other_inputs(self):

        visibilities = aa.Visibilities.manual_1d(visibilities=[1.0 + 2.0j, 3.0 + 4.0j])

        assert type(visibilities) == vis.Visibilities
        assert (visibilities.in_1d == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()
        assert (visibilities.in_array == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert (visibilities.amplitudes == np.array([np.sqrt(5), 5.0])).all()
        assert visibilities.phases == pytest.approx(
            np.array([1.10714872, 0.92729522]), 1.0e-4
        )

        visibilities = aa.Visibilities.manual_1d(
            visibilities=[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]
        )

        assert type(visibilities) == vis.Visibilities
        assert (
            visibilities.in_1d == np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        ).all()
        assert (
            visibilities.in_array == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ).all()

    def test__manual__makes_visibilities_with_converted_input_as_list(self):

        visibilities = aa.Visibilities.manual_1d(visibilities=[[1.0, 2.0], [3.0, 4.0]])

        assert type(visibilities) == vis.Visibilities
        assert (visibilities.in_1d == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()
        assert (visibilities.amplitudes == np.array([np.sqrt(5), 5.0])).all()
        assert visibilities.phases == pytest.approx(
            np.array([1.10714872, 0.92729522]), 1.0e-4
        )

        visibilities = aa.Visibilities.manual_1d(
            visibilities=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )

        assert type(visibilities) == vis.Visibilities
        assert (
            visibilities.in_1d == np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j])
        ).all()

    def test__full__makes_visibilities_without_other_inputs(self):

        visibilities = aa.Visibilities.ones(shape_1d=(2,))

        assert type(visibilities) == vis.Visibilities
        print(visibilities)
        print(visibilities.in_1d)
        assert (visibilities.in_1d == np.array([1.0 + 1.0j, 1.0 + 1.0j])).all()

        visibilities = aa.Visibilities.full(fill_value=2.0, shape_1d=(2,))

        assert type(visibilities) == vis.Visibilities
        assert (visibilities.in_1d == np.array([2.0 + 2.0j, 2.0 + 2.0j])).all()

    def test__ones_zeros__makes_visibilities_without_other_inputs(self):

        visibilities = aa.Visibilities.ones(shape_1d=(2,))

        assert type(visibilities) == vis.Visibilities
        assert (visibilities.in_1d == np.array([1.0 + 1.0j, 1.0 + 1.0j])).all()

        visibilities = aa.Visibilities.zeros(shape_1d=(2,))

        assert type(visibilities) == vis.Visibilities
        assert (visibilities.in_1d == np.array([0.0 + 0.0j, 0.0 + 0.0j])).all()

    def test__from_fits__makes_visibilities_without_other_inputs(self):

        visibilities = aa.Visibilities.from_fits(
            file_path=path.join(test_data_dir, "3x2_ones.fits"), hdu=0
        )

        assert type(visibilities) == vis.Visibilities
        assert (
            visibilities.in_1d == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
        ).all()

        visibilities = aa.Visibilities.from_fits(
            file_path=path.join(test_data_dir, "3x2_twos.fits"), hdu=0
        )

        assert type(visibilities) == vis.Visibilities
        assert (
            visibilities.in_1d == np.array([2.0 + 2.0j, 2.0 + 2.0j, 2.0 + 2.0j])
        ).all()


class TestVisibilities:
    def test__output_to_fits(self):

        visibilities = aa.Visibilities.from_fits(
            file_path=path.join(test_data_dir, "3x2_ones.fits"), hdu=0
        )

        output_data_dir = path.join(test_data_dir, "output_test")

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        visibilities.output_to_fits(
            file_path=path.join(output_data_dir, "visibilities.fits")
        )

        visibilities_from_out = aa.Visibilities.from_fits(
            file_path=path.join(output_data_dir, "visibilities.fits"), hdu=0
        )
        assert (
            visibilities.in_1d == np.array([1.0 + 1.0j, 1.0 + 1.0j, 1.0 + 1.0j])
        ).all()


class TestVisibilitiesNoiseMap:
    def test__visibilities_noise_has_attributes(self):

        noise_map = aa.VisibilitiesNoiseMap.manual_1d(
            visibilities=[[1.0, 2.0], [3.0, 4.0]]
        )

        assert type(noise_map) == vis.VisibilitiesNoiseMap
        assert (noise_map.in_1d == np.array([1.0 + 2.0j, 3.0 + 4.0j])).all()
        assert (noise_map.amplitudes == np.array([np.sqrt(5), 5.0])).all()
        assert noise_map.phases == pytest.approx(
            np.array([1.10714872, 0.92729522]), 1.0e-4
        )

        noise_map = aa.VisibilitiesNoiseMap.manual_1d(
            visibilities=[[1.0, 1.0], [2.0, 2.0], [2.0, 4.0]]
        )

        assert (
            noise_map.Wop.todense()
            == np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                ]
            )
        ).all()

        # assert (
        #     noise_map.Wop.todense()
        #     == np.array(
        #         [
        #             [1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        #             [0.0 + 0.0j, 0.5 + 0.5j, 0.0 + 0.0j],
        #             [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.25j],
        #         ]
        #     )
        # ).all()
