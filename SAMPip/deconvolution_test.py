import numpy as np
import astropy.io.fits as pyfits
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
import skimage.filters as filters
import pdb

scene1 = pyfits.getdata('scene2_cube_sci.fits')
psf = pyfits.getdata('scene2_cube_cal.fits')
reg_im = pyfits.getdata('PSF.fits')

deconv_im = np.zeros_like(scene1)
deconv_im2 = np.zeros_like(scene1)

for i in range(scene1.shape[0]):

    scene11 = scene1[i,:,:]
    psf1 = psf[i,:,:]
    deconvolved_RL, dic = restoration.unsupervised_wiener(scene11, psf1, reg=reg_im, clip=True)
    deconv_im[i,:,:] = deconvolved_RL

    deconvolved_wiener = restoration.wiener(scene11, psf1, 1100, clip=True)
    deconv_im2[i,:,:] = deconvolved_wiener

restored_im1 = filters.gaussian(np.median(deconv_im, axis=0), sigma=1) 
pyfits.writeto('deconvolved_im.fits', np.median(deconv_im, axis=0), overwrite=True)
pyfits.writeto('restored_im.fits', np.median(deconv_im, axis=0), overwrite=True)
pyfits.writeto('deconvolved_im2.fits', np.median(deconv_im2, axis=0), overwrite=True)
restored_im1 = filters.gaussian(np.median(deconv_im, axis=0), sigma=1) 
pyfits.writeto('restored_im2.fits', np.median(deconv_im, axis=0), overwrite=True)




pdb.set_trace()

