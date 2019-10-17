import logging
import os

import numpy as np
import shutil

import autoarray as aa
from autoarray.data import abstract_data

logger = logging.getLogger(__name__)

test_data_dir = "{}/../test_files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)

test_positions_dir = "{}/../test_files/positions/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_ratio_of_each(self):
        array = aa.array.manual_2d([[1.0, 2.0], [3.0, 4.0]])
        noise_map = aa.array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(data=array, noise_map=noise_map)

        assert (data.signal_to_noise_map.in_2d == np.array([[0.1, 0.2], [0.1, 1.0]])).all()
        assert data.signal_to_noise_max == 1.0

    def test__same_as_above__but_image_has_negative_values__replaced_with_zeros(self):
        array = aa.array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])

        noise_map = aa.array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(data=array, noise_map=noise_map)

        assert (data.signal_to_noise_map.in_2d == np.array([[0.0, 0.2], [0.1, 0.0]])).all()
        assert data.signal_to_noise_max == 0.2


class TestAbsoluteSignalToNoise:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = aa.array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])

        noise_map = aa.array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(data=array, noise_map=noise_map)

        assert (
            data.absolute_signal_to_noise_map.in_2d == np.array([[0.1, 0.2], [0.1, 1.0]])
        ).all()
        assert data.absolute_signal_to_noise_max == 1.0


class TestPotentialChiSquaredMap:
    def test__image_and_noise_are_values__signal_to_noise_is_absolute_image_value_over_noise(
        self
    ):
        array = aa.array.manual_2d([[-1.0, 2.0], [3.0, -4.0]])
        noise_map = aa.array.manual_2d([[10.0, 10.0], [30.0, 4.0]])

        data = abstract_data.AbstractData(data=array, noise_map=noise_map)

        assert (
            data.potential_chi_squared_map.in_2d
            == np.array([[0.1 ** 2.0, 0.2 ** 2.0], [0.1 ** 2.0, 1.0 ** 2.0]])
        ).all()
        assert data.potential_chi_squared_max == 1.0


class TestExposureTimeMap(object):

    def test__from_background_noise_map__covnerts_to_exposure_times(self):
        background_noise_map = aa.array.manual_2d([[1.0, 4.0, 8.0], [1.0, 4.0, 8.0]])

        exposure_time_map = abstract_data.ExposureTimeMap.from_exposure_time_and_inverse_noise_map(
            exposure_time=1.0,
            inverse_noise_map=background_noise_map,
        )

        assert (
            exposure_time_map.in_2d
            == np.array([[0.125, 0.5, 1.0], [0.125, 0.5, 1.0]])
        ).all()

        exposure_time_map = abstract_data.ExposureTimeMap.from_exposure_time_and_inverse_noise_map(
            exposure_time=3.0,
            inverse_noise_map=background_noise_map,
        )

        assert (
            exposure_time_map.in_2d
            == np.array([[0.375, 1.5, 3.0], [0.375, 1.5, 3.0]])
        ).all()


class TestPositionsToFile(object):
    def test__load_positions__retains_list_structure(self):
        positions = abstract_data.load_positions(
            positions_path=test_positions_dir + "positions_test.dat"
        )

        assert positions == [
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0], [5.0, 6.0]],
        ]

    def test__output_positions(self):
        positions = [[[4.0, 4.0], [5.0, 5.0]], [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]]]

        output_data_dir = "{}/../test_files/positions/output_test/".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        if os.path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        abstract_data.output_positions(
            positions=positions, positions_path=output_data_dir + "positions_test.dat"
        )

        positions = abstract_data.load_positions(
            positions_path=output_data_dir + "positions_test.dat"
        )

        assert positions == [
            [[4.0, 4.0], [5.0, 5.0]],
            [[6.0, 6.0], [7.0, 7.0], [8.0, 8.0]],
        ]
