import numpy as np
import pytest
import os

import autoarray as aa


class TestLine:
    def test___init_etc(self):
        line = aa.Line(
            line=[1.23, 4.56, 7.89],
            name=None,
            location=[3, 2],
            date=None,
            background=0.5,
            flux=None,
        )

        assert line.length == 3
        assert line.flux == 7.89

        line = aa.Line(
            line=None, name=None, location=None, date=None, background=None, flux=None
        )

        assert line.length is None

        line.line = [1, 2, 3, 4, 5]

        assert line.length == 5


class TestLineCollection:
    def test__collection__init_etc(self):
        line_1 = aa.Line(line=[1, 2, 3], name=100)
        line_2 = aa.Line(line=[4, 5, 6], name=100)
        line_3 = aa.Line(line=[7, 8, 9], name=200)
        line_4 = aa.Line(line=[0, 0, 0], name=200)

        lines = aa.LineCollection(lines=[line_1, line_2, line_3, line_4])

        assert lines.n_lines == 4

        assert lines.line[0] == pytest.approx(line_1.line)
        assert lines.line[1] == pytest.approx(line_2.line)
        assert lines.line[2] == pytest.approx(line_3.line)
        assert lines.line[3] == pytest.approx(line_4.line)

        assert lines.names == pytest.approx([100, 100, 200, 200])

        assert all(lines.dates == [None, None, None, None])

    def test__collection__append(self):
        line_1 = aa.Line(line=[1, 2, 3], name=100)
        line_2 = aa.Line(line=[4, 5, 6], name=100)
        line_3 = aa.Line(line=[7, 8, 9], name=200)
        line_4 = aa.Line(line=[0, 0, 0], name=200)

        lines = aa.LineCollection(lines=[line_1, line_2])

        lines.append([line_3, line_4])

        assert lines.n_lines == 4

        assert lines.line[0] == pytest.approx(line_1.line)
        assert lines.line[1] == pytest.approx(line_2.line)
        assert lines.line[2] == pytest.approx(line_3.line)
        assert lines.line[3] == pytest.approx(line_4.line)

        assert lines.names == pytest.approx([100, 100, 200, 200])

        assert all(lines.dates == [None, None, None, None])

    def test__collection__append_from_None(self):
        line_1 = aa.Line(line=[1, 2, 3], name=100)
        line_2 = aa.Line(line=[4, 5, 6], name=100)
        line_3 = aa.Line(line=[7, 8, 9], name=200)
        line_4 = aa.Line(line=[0, 0, 0], name=200)

        lines = aa.LineCollection()

        assert lines.lines is None

        lines.append([line_1, line_2])
        lines.append([line_3, line_4])

        assert lines.n_lines == 4

        assert lines.line[0] == pytest.approx(line_1.line)
        assert lines.line[1] == pytest.approx(line_2.line)
        assert lines.line[2] == pytest.approx(line_3.line)
        assert lines.line[3] == pytest.approx(line_4.line)

        assert lines.names == pytest.approx([100, 100, 200, 200])

        assert all(lines.dates == [None, None, None, None])

    def test__collection__save_load(self):
        line_1 = aa.Line(line=[1, 2, 3], name=100)
        line_2 = aa.Line(line=[4, 5, 6], name=100)
        line_3 = aa.Line(line=[7, 8, 9], name=200)
        line_4 = aa.Line(line=[0, 0, 0], name=200)

        lines_1 = aa.LineCollection(lines=[line_1, line_2])
        lines_2 = aa.LineCollection(lines=[line_3, line_4])

        # Path to this file
        path = os.path.dirname(os.path.realpath(__file__))
        filename = path + "test__collection__save_load"
        print(filename)

        lines_2.save(filename=filename)

        # Load and append the saved lines
        lines_1.load(filename=filename)

        assert lines_1.n_lines == 4

        assert lines_1.line[0] == pytest.approx(line_1.line)
        assert lines_1.line[1] == pytest.approx(line_2.line)
        assert lines_1.line[2] == pytest.approx(line_3.line)
        assert lines_1.line[3] == pytest.approx(line_4.line)

        assert lines_1.names == pytest.approx([100, 100, 200, 200])

        assert all(lines_1.dates == [None, None, None, None])

        # Remove the file
        assert os.path.exists(filename + ".pickle")
        os.remove(filename + ".pickle")

    def test__find_consistent_lines(self):
        line_1 = aa.Line(line=[1, 2, 3], location=[0, 0], name=100)
        line_2 = aa.Line(line=[4, 5, 6], location=[1, 1], name=100)
        line_3 = aa.Line(line=[7, 8, 9], location=[2, 2], name=100)
        line_4 = aa.Line(line=[0, 0, 0], location=[0, 0], name=200)
        line_5 = aa.Line(line=[3, 2, 1], location=[1, 1], name=200)
        line_6 = aa.Line(line=[3, 1, 0], location=[3, 4], name=200)
        line_7 = aa.Line(line=[2, 1, 0], location=[0, 0], name=300)
        line_8 = aa.Line(line=[1, 1, 0], location=[1, 2], name=300)
        line_9 = aa.Line(line=[1, 0, 0], location=[4, 4], name=300)

        lines = aa.LineCollection(
            lines=[
                line_1,
                line_2,
                line_3,
                line_4,
                line_5,
                line_6,
                line_7,
                line_8,
                line_9,
            ]
        )

        # aa.Lines present in every image
        consistent_lines = lines.find_consistent_lines(fraction_present=1)

        assert consistent_lines == pytest.approx([0, 3, 6])
        assert lines.names[consistent_lines] == pytest.approx([100, 200, 300])

        # aa.Lines present in at least 2/3 images
        consistent_lines = lines.find_consistent_lines(fraction_present=2 / 3)

        assert consistent_lines == pytest.approx([0, 1, 3, 4, 6])

        # aa.Lines present in at least 1/2 images
        consistent_lines = lines.find_consistent_lines(fraction_present=0.5)

        assert consistent_lines == pytest.approx([0, 1, 3, 4, 6])

        # aa.Lines present in at least 1/3 images (in this case, all lines)
        consistent_lines = lines.find_consistent_lines(fraction_present=1 / 3)

        assert consistent_lines == pytest.approx([0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test__generate_stacked_lines_from_bins(self):
        # Stack lines in 3 row bins and 2 flux bins
        # Different dates but only 1 bin so ignored
        # All same backgrounds so ignored

        # 1. Low row, low flux
        line_1 = aa.Line(line=[3, 2.5, 2], location=[0, 0], date=1, background=0)
        line_2 = aa.Line(line=[2.5, 2, 1.5], location=[1, 0], date=2, background=0)
        line_3 = aa.Line(line=[2, 1.5, 1], location=[2, 0], date=3, background=0)
        # 2. Low row, high flux
        line_4 = aa.Line(line=[10, 8, 6], location=[3, 0], date=4, background=0)
        line_5 = aa.Line(line=[9, 7, 5], location=[4, 0], date=5, background=0)
        # 3. Mid row, low flux
        line_6 = aa.Line(line=[3, 2, 1], location=[10, 0], date=6, background=0)
        line_7 = aa.Line(line=[2.5, 1.5, 0.5], location=[11, 0], date=7, background=0)
        line_8 = aa.Line(line=[2, 1, 0], location=[12, 0], date=8, background=0)
        # 4. Mid row, high flux
        line_9 = aa.Line(line=[10, 9, 8], location=[10, 0], date=9, background=0)
        line_10 = aa.Line(line=[8, 8, 8], location=[11, 0], date=10, background=0)
        # 5. High row, low flux
        line_11 = aa.Line(line=[3, 2, 1], location=[20, 0], date=11, background=0)
        # 6. High row, high flux: no lines, empty bin

        # Below minimum flux, discarded
        line_12 = aa.Line(line=[0.5, 0, 0], location=[21, 0], date=12, background=0)
        # Above maximum flux, discarded
        line_13 = aa.Line(line=[11, 10, 9], location=[0, 0], date=13, background=0)

        lines = aa.LineCollection(
            lines=[
                line_1,
                line_2,
                line_3,
                line_4,
                line_5,
                line_6,
                line_7,
                line_8,
                line_9,
                line_10,
                line_11,
                line_12,
                line_13,
            ]
        )

        stacked_lines = lines.generate_stacked_lines_from_bins(
            n_row_bins=3,
            row_min=None,
            row_max=None,
            n_flux_bins=2,
            flux_min=1,
            flux_max=10,
            flux_scale="log",
            n_date_bins=1,
        )

        assert stacked_lines.n_lines == 6

        stack_1, stack_2, stack_3, stack_4, stack_5, stack_6 = stacked_lines.lines

        assert stack_1.n_stacked == 3
        assert stack_1.line == pytest.approx([2.5, 2, 1.5])
        assert stack_1.noise_map == pytest.approx([np.sqrt(0.5) / 3] * 3)
        assert stack_1.location == pytest.approx([0, 0])
        assert stack_1.flux == 1

        assert stack_2.n_stacked == 2
        assert stack_2.line == pytest.approx([9.5, 7.5, 5.5])
        assert stack_2.noise_map == pytest.approx([np.sqrt(0.5) / 2] * 3)
        assert stack_2.location == pytest.approx([0, 0])
        assert stack_2.flux == 10 ** 0.5

        assert stack_3.n_stacked == 3
        assert stack_3.line == pytest.approx([2.5, 1.5, 0.5])
        assert stack_3.noise_map == pytest.approx([np.sqrt(0.5) / 3] * 3)
        assert stack_3.location == pytest.approx([7, 0])
        assert stack_3.flux == 1

        assert stack_4.n_stacked == 2
        assert stack_4.line == pytest.approx([9, 8.5, 8])
        assert stack_4.noise_map == pytest.approx([np.sqrt(0.5), np.sqrt(0.5) / 2, 0])
        assert stack_4.location == pytest.approx([7, 0])
        assert stack_4.flux == 10 ** 0.5

        assert stack_5.n_stacked == 1
        assert stack_5.line == pytest.approx([3, 2, 1])
        assert stack_5.noise_map == pytest.approx([0, 0, 0])
        assert stack_5.location == pytest.approx([14, 0])
        assert stack_5.flux == 1

        assert stack_6.n_stacked == 0
        assert stack_6.line == pytest.approx([0, 0, 0])
        assert stack_6.noise_map == pytest.approx([0, 0, 0])
        assert stack_6.location == pytest.approx([14, 0])
        assert stack_6.flux == 10 ** 0.5

        # Also test return_bin_info=True
        (
            stacked_lines,
            row_bins,
            flux_bins,
            date_bins,
            background_bins,
        ) = lines.generate_stacked_lines_from_bins(
            n_row_bins=3,
            row_min=None,
            row_max=None,
            n_flux_bins=2,
            flux_min=1,
            flux_max=10,
            flux_scale="log",
            n_date_bins=1,
            return_bin_info=True,
        )

        assert stacked_lines.n_lines == 6

        assert row_bins == pytest.approx([0, 7, 14, 21])
        assert flux_bins == pytest.approx([1, 10 ** 0.5, 10])
        assert date_bins == pytest.approx([1, 13])
        assert background_bins == pytest.approx([0, 0])

    def test__generate_stacked_lines_from_bins__custom_bins(self):
        # Stack lines in 3 row bins and 2 flux bins
        # Different dates but only 1 bin so ignored
        # All same backgrounds so ignored

        # 1. Low row, low flux
        line_1 = aa.Line(line=[3, 2.5, 2], location=[0, 0], date=1, background=0)
        line_2 = aa.Line(line=[2.5, 2, 1.5], location=[1, 0], date=2, background=0)
        line_3 = aa.Line(line=[2, 1.5, 1], location=[2, 0], date=3, background=0)
        # 2. Low row, high flux
        line_4 = aa.Line(line=[10, 8, 6], location=[3, 0], date=4, background=0)
        line_5 = aa.Line(line=[9, 7, 5], location=[4, 0], date=5, background=0)
        # 3. Mid row, low flux
        line_6 = aa.Line(line=[3, 2, 1], location=[10, 0], date=6, background=0)
        line_7 = aa.Line(line=[2.5, 1.5, 0.5], location=[11, 0], date=7, background=0)
        line_8 = aa.Line(line=[2, 1, 0], location=[12, 0], date=8, background=0)
        # 4. Mid row, high flux
        line_9 = aa.Line(line=[10, 9, 8], location=[10, 0], date=9, background=0)
        line_10 = aa.Line(line=[8, 8, 8], location=[11, 0], date=10, background=0)
        # 5. High row, low flux
        line_11 = aa.Line(line=[3, 2, 1], location=[20, 0], date=11, background=0)
        # 6. High row, high flux: no lines, empty bin

        # Below minimum flux, discarded
        line_12 = aa.Line(line=[0.5, 0, 0], location=[21, 0], date=12, background=0)
        # Above maximum flux, discarded
        line_13 = aa.Line(line=[11, 10, 9], location=[0, 0], date=13, background=0)

        lines = aa.LineCollection(
            lines=[
                line_1,
                line_2,
                line_3,
                line_4,
                line_5,
                line_6,
                line_7,
                line_8,
                line_9,
                line_10,
                line_11,
                line_12,
                line_13,
            ]
        )

        stacked_lines = lines.generate_stacked_lines_from_bins(
            row_bins=[0, 7, 14, 999.9], flux_bins=[1, np.pi, 10], n_date_bins=1
        )

        assert stacked_lines.n_lines == 6

        stack_1, stack_2, stack_3, stack_4, stack_5, stack_6 = stacked_lines.lines

        # for line in stacked_lines.lines:
        #     print(line.line)
        #     print(line.noise_map)
        # exit()

        assert stack_1.n_stacked == 3
        assert stack_1.line == pytest.approx([2.5, 2, 1.5])
        assert stack_1.noise_map == pytest.approx([np.sqrt(0.5) / 3] * 3)
        assert stack_1.location == pytest.approx([0, 0])
        assert stack_1.flux == 1

        assert stack_2.n_stacked == 2
        assert stack_2.line == pytest.approx([9.5, 7.5, 5.5])
        assert stack_2.noise_map == pytest.approx([np.sqrt(0.5) / 2] * 3)
        assert stack_2.location == pytest.approx([0, 0])
        assert stack_2.flux == np.pi

        assert stack_3.n_stacked == 3
        assert stack_3.line == pytest.approx([2.5, 1.5, 0.5])
        assert stack_3.noise_map == pytest.approx([np.sqrt(0.5) / 3] * 3)
        assert stack_3.location == pytest.approx([7, 0])
        assert stack_3.flux == 1

        assert stack_4.n_stacked == 2
        assert stack_4.line == pytest.approx([9, 8.5, 8])
        assert stack_4.noise_map == pytest.approx([np.sqrt(0.5), np.sqrt(0.5) / 2, 0])
        assert stack_4.location == pytest.approx([7, 0])
        assert stack_4.flux == np.pi

        assert stack_5.n_stacked == 1
        assert stack_5.line == pytest.approx([3, 2, 1])
        assert stack_5.noise_map == pytest.approx([0, 0, 0])
        assert stack_5.location == pytest.approx([14, 0])
        assert stack_5.flux == 1

        assert stack_6.n_stacked == 0
        assert stack_6.line == pytest.approx([0, 0, 0])
        assert stack_6.noise_map == pytest.approx([0, 0, 0])
        assert stack_6.location == pytest.approx([14, 0])
        assert stack_6.flux == np.pi
