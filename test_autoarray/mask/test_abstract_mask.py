import numpy as np

import autoarray as aa





def test__mask_new_sub_size_from():
    mask = aa.Mask2D.all_false(shape_native=(3, 3), sub_size=4, pixel_scales=1.0)

    mask_new = mask.mask_new_sub_size_from(mask=mask)

    assert (mask_new == np.full(fill_value=False, shape=(3, 3))).all()
    assert mask_new.sub_size == 1

    mask_new = mask.mask_new_sub_size_from(mask=mask, sub_size=8)

    assert (mask_new == np.full(fill_value=False, shape=(3, 3))).all()
    assert mask_new.sub_size == 8
