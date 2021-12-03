import scipy.ndimage as ndimage
import pdb
import astropy.io.fits as pyfits
import numpy as np

data = pyfits.getdata('rings_gauss.fits')
rot_data = ndimage.rotate(data, 10.0, reshape=False)
pyfits.writeto('rot1_rings_gauss.fits', rot_data, overwrite=True)



pdb.set_trace()