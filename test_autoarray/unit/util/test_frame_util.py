import numpy as np
import pytest
import autoarray as aa


class TestRotations:
    def test__rotate_array__all_4_rotations_with_rotation_back(self):

        arr = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])

        arr_bl = aa.util.frame.rotate_array_from_roe_corner(
            array=arr, roe_corner=(1, 0)
        )

        assert arr_bl == pytest.approx(
            np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]), 1.0e-4
        )

        arr_bl = aa.util.frame.rotate_array_from_roe_corner(
            array=arr_bl, roe_corner=(1, 0)
        )

        assert arr_bl == pytest.approx(np.array(arr), 1.0e-4)

        arr_br = aa.util.frame.rotate_array_from_roe_corner(
            array=arr, roe_corner=(1, 1)
        )

        assert arr_br == pytest.approx(
            np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 0.0]]), 1.0e-4
        )

        arr_br = aa.util.frame.rotate_array_from_roe_corner(
            array=arr_br, roe_corner=(1, 1)
        )

        assert arr_br == pytest.approx(np.array(arr), 1.0e-4)

        arr_tl = aa.util.frame.rotate_array_from_roe_corner(
            array=arr, roe_corner=(0, 0)
        )

        assert arr_tl == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]]), 1.0e-4
        )

        arr_tl = aa.util.frame.rotate_array_from_roe_corner(
            array=arr_tl, roe_corner=(0, 0)
        )

        assert arr_tl == pytest.approx(np.array(arr), 1.0e-4)

        arr_tr = aa.util.frame.rotate_array_from_roe_corner(
            array=arr, roe_corner=(0, 1)
        )

        assert arr_tr == pytest.approx(
            np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 0.0]]), 1.0e-4
        )

        arr_tr = aa.util.frame.rotate_array_from_roe_corner(
            array=arr_tr, roe_corner=(0, 1)
        )

        assert arr_tr == pytest.approx(np.array(arr), 1.0e-4)

    def test__rotate_region__all_4_rotations_with_rotation_back(self):

        region = (0, 2, 1, 3)

        shape_2d = (8, 10)

        region_bl = aa.util.frame.rotate_region_from_roe_corner(
            region=region, shape_2d=shape_2d, roe_corner=(1, 0)
        )

        assert region_bl == (0, 2, 1, 3)

        region_bl = aa.util.frame.rotate_region_from_roe_corner(
            region=region_bl, shape_2d=shape_2d, roe_corner=(1, 0)
        )

        assert region_bl == (0, 2, 1, 3)

        region_br = aa.util.frame.rotate_region_from_roe_corner(
            region=region, shape_2d=shape_2d, roe_corner=(1, 1)
        )

        assert region_br == (0, 2, 7, 9)

        region_br = aa.util.frame.rotate_region_from_roe_corner(
            region=region_br, shape_2d=shape_2d, roe_corner=(1, 1)
        )

        assert region_br == (0, 2, 1, 3)

        region_tl = aa.util.frame.rotate_region_from_roe_corner(
            region=region, shape_2d=shape_2d, roe_corner=(0, 0)
        )

        assert region_tl == (6, 8, 1, 3)

        region_tl = aa.util.frame.rotate_region_from_roe_corner(
            region=region_tl, shape_2d=shape_2d, roe_corner=(0, 0)
        )

        assert region_tl == (0, 2, 1, 3)

        region_tr = aa.util.frame.rotate_region_from_roe_corner(
            region=region, shape_2d=shape_2d, roe_corner=(0, 1)
        )

        assert region_tr == (6, 8, 7, 9)

        region_tr = aa.util.frame.rotate_region_from_roe_corner(
            region=region_tr, shape_2d=shape_2d, roe_corner=(0, 1)
        )

        assert region_tr == (0, 2, 1, 3)

    # def test__rotate_ci_pattern__all_4_rotations_with_rotation_back(self):
    #
    #     pattern = ci_pattern.CIPatternUniform(
    #         regions=[(0, 1, 1, 2), (0, 2, 0, 2)], normalization=10.0
    #     )
    #
    #     shape_2d = (2, 2)
    #
    #     pattern_bl = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern, shape_2d=shape_2d, roe_corner=(1, 0)
    #     )
    #
    #     assert pattern_bl.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]
    #     assert pattern_bl.normalization == 10.0
    #
    #     pattern_bl = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern_bl, shape_2d=shape_2d, roe_corner=(1, 0)
    #     )
    #
    #     assert pattern_bl.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]
    #
    #     pattern_br = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern, shape_2d=shape_2d, roe_corner=(1, 1)
    #     )
    #
    #     assert pattern_br.regions == [(0, 1, 0, 1), (0, 2, 0, 2)]
    #
    #     pattern_br = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern_br, shape_2d=shape_2d, roe_corner=(1, 1)
    #     )
    #
    #     assert pattern_br.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]
    #
    #     pattern_tl = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern, shape_2d=shape_2d, roe_corner=(0, 0)
    #     )
    #
    #     assert pattern_tl.regions == [(1, 2, 1, 2), (0, 2, 0, 2)]
    #
    #     pattern_tl = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern_tl, shape_2d=shape_2d, roe_corner=(0, 0)
    #     )
    #
    #     assert pattern_tl.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]
    #
    #     pattern_tr = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern, shape_2d=shape_2d, roe_corner=(0, 1)
    #     )
    #
    #     assert pattern_tr.regions == [(1, 2, 0, 1), (0, 2, 0, 2)]
    #
    #     pattern_tr = aa.util.frame.rotate_ci_pattern_from_roe_corner(
    #         ci_pattern=pattern_tr, shape_2d=shape_2d, roe_corner=(0, 1)
    #     )
    #
    #     assert pattern_tr.regions == [(0, 1, 1, 2), (0, 2, 0, 2)]


class TestRegionAfterExtraction:
    def test__simple_test_cases(self):

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(0, 6, 0, 6)
        )

        assert region == (2, 4, 2, 4)

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(3, 5, 3, 5)
        )

        assert region == (0, 1, 0, 1)

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(2, 5, 2, 5)
        )

        assert region == (0, 2, 0, 2)

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(0, 3, 0, 3)
        )

        assert region == (2, 3, 2, 3)

    def test__regions_do_not_overlap__returns_none(self):

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(0, 6, 0, 1)
        )

        assert region == None

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(0, 1, 0, 6)
        )

        assert region == None

        region = aa.util.frame.region_after_extraction(
            original_region=(2, 4, 2, 4), extraction_region=(0, 1, 0, 1)
        )

        assert region == None

        region = aa.util.frame.region_after_extraction(
            original_region=None, extraction_region=(0, 6, 0, 1)
        )

        assert region == None


class Testx0x1AfterExtraction:
    def test__case_1__original_region_at_0__1d_extracted_region_is_fully_within_original_region(
        self
    ):

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=0, x1o=6, x0e=2, x1e=4)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=0, x1o=6, x0e=3, x1e=5)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=0, x1o=6, x0e=4, x1e=6)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=0, x1o=6, x0e=5, x1e=6)

        assert x0 == 0
        assert x1 == 1

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=0, x1o=6, x0e=2, x1e=5)

        assert x0 == 0
        assert x1 == 3

    def test__case_2__original_region_offset_from_0__1d_extracted_region_is_fully_within_original_region(
        self
    ):

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=2, x1e=4)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=3, x1e=5)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=4, x1e=6)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=5, x1e=6)

        assert x0 == 0
        assert x1 == 1

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=2, x1e=5)

        assert x0 == 0
        assert x1 == 3

    def test__case_3__original_region_offset_from_0__1d_extracted_region_partly_overlaps_to_left_original_region(
        self
    ):

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=1, x1e=3)

        assert x0 == 1
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=1, x1e=4)

        assert x0 == 1
        assert x1 == 3

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=3)

        assert x0 == 2
        assert x1 == 3

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=5)

        assert x0 == 2
        assert x1 == 5

    def test__case_4__original_region_offset_from_0__1d_extracted_region_partly_overlaps_to_right_original_region(
        self
    ):

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=5, x1e=7)

        assert x0 == 0
        assert x1 == 1

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=5, x1e=8)

        assert x0 == 0
        assert x1 == 1

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=4, x1e=7)

        assert x0 == 0
        assert x1 == 2

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=2, x1e=8)

        assert x0 == 0
        assert x1 == 4

    def test__case_5__extraction_region_over_full_original_region(self):

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=8)

        assert x0 == 2
        assert x1 == 6

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=7)

        assert x0 == 2
        assert x1 == 6

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=1, x1e=8)

        assert x0 == 1
        assert x1 == 5

    def test__case_6__extraction_region_misses_original_region(self):

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=7, x1e=8)

        assert x0 == None
        assert x1 == None

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=6, x1e=8)

        assert x0 == None
        assert x1 == None

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=1)

        assert x0 == None
        assert x1 == None

        x0, x1 = aa.util.frame.x0x1_after_extraction(x0o=2, x1o=6, x0e=0, x1e=2)

        assert x0 == None
        assert x1 == None
