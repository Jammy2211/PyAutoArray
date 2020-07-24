from copy import deepcopy

from autoarray.structures import region as reg


def rotate_array_from_roe_corner(array, roe_corner):

    if roe_corner == (1, 0):
        return array.copy()
    elif roe_corner == (0, 0):
        return array[::-1, :].copy()
    elif roe_corner == (1, 1):
        return array[:, ::-1].copy()
    elif roe_corner == (0, 1):
        array = array[::-1, :].copy()
        return array[:, ::-1]


def rotate_region_from_roe_corner(region, shape_2d, roe_corner):

    if region is None:
        return None

    if roe_corner == (1, 0):
        return reg.Region(region=region)
    elif roe_corner == (0, 0):
        return reg.Region(
            region=(
                shape_2d[0] - region[1],
                shape_2d[0] - region[0],
                region[2],
                region[3],
            )
        )
    elif roe_corner == (1, 1):
        return reg.Region(
            region=(
                region[0],
                region[1],
                shape_2d[1] - region[3],
                shape_2d[1] - region[2],
            )
        )
    elif roe_corner == (0, 1):
        return reg.Region(
            region=(
                shape_2d[0] - region[1],
                shape_2d[0] - region[0],
                shape_2d[1] - region[3],
                shape_2d[1] - region[2],
            )
        )


def rotate_ci_pattern_from_roe_corner(ci_pattern, shape_2d, roe_corner):

    new_ci_pattern = deepcopy(ci_pattern)

    new_ci_pattern.regions = [
        rotate_region_from_roe_corner(
            region=region, shape_2d=shape_2d, roe_corner=roe_corner
        )
        for region in ci_pattern.regions
    ]

    return new_ci_pattern


def region_after_extraction(original_region, extraction_region):

    if original_region is None:
        return None

    y0, y1 = x0x1_after_extraction(
        x0o=original_region[0],
        x1o=original_region[1],
        x0e=extraction_region[0],
        x1e=extraction_region[1],
    )
    x0, x1 = x0x1_after_extraction(
        x0o=original_region[2],
        x1o=original_region[3],
        x0e=extraction_region[2],
        x1e=extraction_region[3],
    )

    if None in [y0, y1, x0, x1]:
        return None
    else:
        return (y0, y1, x0, x1)


def x0x1_after_extraction(x0o, x1o, x0e, x1e):
    """When we extract a frame from a frame, we also update the extracted frame's regions by mapping each region
    from their coordinates on the original frame (which has a shape_2d) to the extracted frame (which is a 2D section
    on this frame).

    This function compares the 1D coordinates of a regions original coordinates on a frame to the 1D coordinates of the
    extracted frame, determining where the original region lies on the extracted frame.

    For example, for a 1D array with shape 8 we may have a region whose 1D coordinates span x0o=2 -> x1o=6. From the
    original 1D array we then extract the region x0e=5 -> x1e = 7. This looks as follows:

                                eeeeeeeee
                                5        7      e = extracted region
          oooooooooooooooooooooooooo            o = original region
         2                          6           - = original array (which has shape = 8
      ------------------------------------
     0                                    8

     In the above example this function will recognise that the extracted region will contain a small section of the
     original region and for the extracted region give it coordinates (0, 1). This function covers all possible
     ways the original region and extracted frame could over lap.

    If the extraction completely the region a None is returned."""

    if x0e >= x0o and x0e <= x1o:
        x0 = 0
    elif x0e <= x0o:
        x0 = x0o - x0e
    elif x0e >= x0o:
        x0 = 0

    if x1e >= x0o and x1e <= x1o:
        x1 = x1e - x0e
    elif x1e > x1o:
        x1 = x1o - x0e

    try:
        if x0 < 0 or x1 < 0 or x0 == x1:
            return None, None
        else:
            return x0, x1
    except UnboundLocalError:
        return None, None
