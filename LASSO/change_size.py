from congrid import *
import numpy as np
import astropy.io.fits as pyfits
import pdb

ims = pyfits.getdata('conv_psf3_w3.fits')

new_im = np.zeros([2, 80, 80])

for i in range(ims.shape[0]):
    new_im[i,:,:] = bin_ndarray(ims[i,:,:], (80,80))

pyfits.writeto('r_conv_psf3_w3.fits', new_im, overwrite=True)

pdb.set_trace()