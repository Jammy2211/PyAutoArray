from os import path
import pytest
import autoarray as aa
import autoarray.plot as aplt
from autoarray.dataset.plot.imaging_plots import subplot_imaging_dataset


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__individual_attributes_are_output(
    imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    aplt.plot_array_2d(
        array=imaging_7x7.data,
        positions=grid_2d_irregular_7x7_list,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert path.join(plot_path, "data.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=imaging_7x7.noise_map,
        output_path=plot_path,
        output_filename="noise_map",
        output_format="png",
    )
    assert path.join(plot_path, "noise_map.png") in plot_patch.paths

    if imaging_7x7.psf is not None:
        aplt.plot_array_2d(
            array=imaging_7x7.psf.kernel,
            output_path=plot_path,
            output_filename="psf",
            output_format="png",
        )
        assert path.join(plot_path, "psf.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=imaging_7x7.signal_to_noise_map,
        output_path=plot_path,
        output_filename="signal_to_noise_map",
        output_format="png",
    )
    assert path.join(plot_path, "signal_to_noise_map.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=imaging_7x7.grids.over_sample_size_lp,
        output_path=plot_path,
        output_filename="over_sample_size_lp",
        output_format="png",
    )
    assert path.join(plot_path, "over_sample_size_lp.png") in plot_patch.paths

    aplt.plot_array_2d(
        array=imaging_7x7.grids.over_sample_size_pixelization,
        output_path=plot_path,
        output_filename="over_sample_size_pixelization",
        output_format="png",
    )
    assert path.join(plot_path, "over_sample_size_pixelization.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.plot_array_2d(
        array=imaging_7x7.data,
        output_path=plot_path,
        output_filename="data",
        output_format="png",
    )
    assert path.join(plot_path, "data.png") in plot_patch.paths
    assert not path.join(plot_path, "noise_map.png") in plot_patch.paths


def test__subplot_is_output(
    imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    subplot_imaging_dataset(
        dataset=imaging_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "dataset.png") in plot_patch.paths


def test__output_as_fits__correct_output_format(
    imaging_7x7, grid_2d_irregular_7x7_list, mask_2d_7x7, plot_path, plot_patch
):
    aplt.plot_array_2d(
        array=imaging_7x7.data,
        output_path=plot_path,
        output_filename="data",
        output_format="fits",
    )

    image_from_plot = aa.ndarray_via_fits_from(
        file_path=path.join(plot_path, "data.fits"), hdu=0
    )

    assert image_from_plot.shape == (7, 7)
