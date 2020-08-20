import numpy as np


def get_real_space_mask_from(mask, fit):
    """Get the masks of the fit if the masks should be plotted on the fit.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
    mask : bool
        If *True*, the masks is plotted on the fit's datas.
    """
    if mask:
        return fit.settings_masked_dataset.mask
    else:
        return None


def radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):
    return (maximum_radius - minimum_radius) / radii_points


def quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):
    return list(
        np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1)
    )


def quantity_and_annuli_radii_from_minimum_and_maximum_radii_and_radii_points(
    minimum_radius, maximum_radius, radii_points
):

    radii_bin_size = radii_bin_size_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        radii_points=radii_points,
    )

    quantity_radii = list(
        np.linspace(
            start=minimum_radius + radii_bin_size / 2.0,
            stop=maximum_radius - radii_bin_size / 2.0,
            num=radii_points,
        )
    )
    annuli_radii = list(
        np.linspace(start=minimum_radius, stop=maximum_radius, num=radii_points + 1)
    )

    return quantity_radii, annuli_radii
