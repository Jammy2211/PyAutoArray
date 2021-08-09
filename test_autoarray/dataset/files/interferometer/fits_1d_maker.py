from astropy.io import fits
import numpy as np
from os import path

file_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "array"
)

array1 = np.ones((3))
array2 = 2.0 * np.ones((3))
array3 = 3.0 * np.ones((3))
array4 = 4.0 * np.ones((3))
array5 = 5.0 * np.ones((3))
array6 = 6.0 * np.ones((3))
array7 = 7.0 * np.ones((3))
array8 = 8.0 * np.ones((3))

# fits.writeto(simulator=array1, filename=path + "3_ones.fits", overwrite=True)
# fits.writeto(simulator=array2, filename=path + "3_twos.fits")
# fits.writeto(simulator=array3, filename=path + "3_threes.fits")
# fits.writeto(simulator=array4, filename=path + "3_fours.fits")
# fits.writeto(simulator=array5, filename=path + "3_fives.fits")
# fits.writeto(simulator=array6, filename=path + "3_sixes.fits")
# fits.writeto(simulator=array7, filename=path + "3_sevens.fits")
# fits.writeto(simulator=array8, filename=path + "3_eights.fits")

hdu_list = fits.HDUList()
hdu_list.append(fits.ImageHDU(array1))
hdu_list.append(fits.ImageHDU(array2))
hdu_list.append(fits.ImageHDU(array3))
hdu_list.append(fits.ImageHDU(array4))
hdu_list.append(fits.ImageHDU(array5))
hdu_list.append(fits.ImageHDU(array6))
hdu_list.append(fits.ImageHDU(array7))
hdu_list.append(fits.ImageHDU(array8))

hdu_list.writeto(path.join(file_path, "3_multiple_hdu.fits"))
