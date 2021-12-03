import numpy as np
import pdb
import astropy.io.fits as pyfits
from readcol import *
import cv2
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import scipy.ndimage as ndimage

imsize = 400
im = np.zeros([imsize, imsize])
im2 = np.zeros([imsize, imsize])
ax1=int(np.random.random(1)[0]*8)
ax2=int(np.random.random(1)[0]*8)
pa=np.random.random(1)[0]*360
cv2.ellipse(im, (int(imsize / 2), int(imsize / 2)), (ax1,ax2), \
                pa, 0, 360, (1, 1, 1), -1)
width = np.random.random(1)[0] * 0.5
cv2.ellipse(im2, (int(imsize / 2), int(imsize / 2)), (int(ax1+ax1*width+1), \
                                                          int(ax2+ax2*width+1)), \
                pa, 0, 360, (1, 1, 1), -1)
ells = im2 - im
pyfits.writeto('rings.fits', ells, overwrite=True)

kernel = Gaussian2DKernel(x_stddev=2, x_size=81, y_size=81)
stropy_conv = convolve(ells, kernel)
stropy_conv = stropy_conv / np.sum(stropy_conv)
pyfits.writeto('rings_gauss.fits', stropy_conv, overwrite=True)

# ######
# im = np.zeros([imsize, imsize])
# im2 = np.zeros([imsize, imsize])
# ax1=int(np.random.random(1)[0]*15)
# ax2=int(np.random.random(1)[0]*15)
# pa=np.random.random(1)[0]*360
# cv2.ellipse(im, (int(imsize / 2), int(imsize / 2)), (ax1,ax2), \
#                 pa, 0, 360, (1, 1, 1), -1)
# width = np.random.random(1)[0] * 3.0
# cv2.ellipse(im2, (int(imsize / 2), int(imsize / 2)), (int(ax1+ax1*width+1), \
#                                                           int(ax2+ax2*width+1)), \
#                 pa, 0, 360, (1, 1, 1), -1)
# ells = im2 - im
# pyfits.writeto('rings2.fits', ells, overwrite=True)
#
# kernel = Gaussian2DKernel(x_stddev=5, x_size=81, y_size=81)
# stropy_conv = convolve(ells, kernel)
# stropy_conv = stropy_conv / np.sum(stropy_conv)
# pyfits.writeto('rings_gauss2.fits', stropy_conv, overwrite=True)

#######
# im = np.zeros([imsize, imsize])
# im2 = np.zeros([imsize, imsize])
# ax1=int(np.random.random(1)[0]*20)
# ax2=int(np.random.random(1)[0]*20)
# pa=np.random.random(1)[0]*360
# cv2.ellipse(im, (int(imsize / 2), int(imsize / 2)), (ax1,ax2), \
#                 pa, 0, 360, (1, 1, 1), -1)
# width = np.random.random(1)[0] * 0.5
# cv2.ellipse(im2, (int(imsize / 2), int(imsize / 2)), (int(ax1+ax1*width+1), \
#                                                           int(ax2+ax2*width+1)), \
#                 pa, 0, 360, (1, 1, 1), -1)
# ells = im2 - im
# pyfits.writeto('rings3.fits', ells, overwrite=True)
#
# kernel = Gaussian2DKernel(x_stddev=2, x_size=81, y_size=81)
# stropy_conv = convolve(ells, kernel)
# stropy_conv = stropy_conv / np.sum(stropy_conv)
# pyfits.writeto('rings_gauss3.fits', stropy_conv, overwrite=True)



#psf = pyfits.getdata('piston_psfs/PSF_5x_MASK_NRM_F380M_A0V__ps1_10rms.fits')
#conv_psf = ndimage.convolve(stropy_conv, psf)

#pyfits.writeto('conv_psf.fits', conv_psf, overwrite=True)




pdb.set_trace()