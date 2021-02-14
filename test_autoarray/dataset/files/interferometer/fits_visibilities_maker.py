from astropy.io import fits
import numpy as np
from os import path

file_path = "{}".format(path.dirname(path.realpath(__file__)))

array1 = np.ones((3, 2))
array2 = 2.0 * np.ones((3, 2))
array3 = 3.0 * np.ones((3, 2))
array4 = 4.0 * np.ones((3, 2))

fits.writeto(
    data=array1, filename=path.join(file_path, "3x2_ones.fits"), overwrite=True
)
fits.writeto(
    data=array2, filename=path.join(file_path, "3x2_twos.fits"), overwrite=True
)
fits.writeto(
    data=array3, filename=path.join(file_path, "3x2_threes.fits"), overwrite=True
)
fits.writeto(
    data=array4, filename=path.join(file_path, "3x2_fours.fits"), overwrite=True
)

array12 = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])

fits.writeto(
    data=array12, filename=path.join(file_path, "3x2_ones_twos.fits"), overwrite=True
)

array34 = np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])

fits.writeto(
    data=array34, filename=path.join(file_path, "3x2_threes_fours.fits"), overwrite=True
)

array56 = np.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])

fits.writeto(
    data=array56, filename=path.join(file_path, "3x2_fives_sixes.fits"), overwrite=True
)

new_hdul = fits.HDUList()
new_hdul.append(fits.ImageHDU(array1))
new_hdul.append(fits.ImageHDU(array2))
new_hdul.append(fits.ImageHDU(array3))
new_hdul.append(fits.ImageHDU(array4))

new_hdul.writeto(path.join(file_path, "3x2_multiple_hdu.fits"), overwrite=True)
