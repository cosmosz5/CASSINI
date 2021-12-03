import numpy as np
import astropy.io.fits as pyfits
import pdb
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from congrid import *

psfs = ['PSF_5x_MASK_NRM_F480M_A0V__ps1_10rms.fits',  'PSF_5x_MASK_NRM_F480M_A0V__ps1_50rms.fits']
disk = pyfits.getdata('rot2_rings_gauss.fits')
conv_psf = np.zeros([2, 400, 400])
for i in range(len(psfs)):
    psf = pyfits.getdata('piston_psfs/'+psfs[i])
    conv_psf[i,:,:] = ndimage.convolve(disk, psf)

pyfits.writeto('conv_psf3_w3.fits', conv_psf, overwrite=True)



pdb.set_trace()