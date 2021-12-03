import numpy as np
import pdb
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import scipy.ndimage.filters as ndimage
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import cv2

no_data = 5000
input_path = ['/Users/donut/Desktop/dataset/']
imsize=71

im = np.zeros([imsize, imsize])
im[:,:] = 1.0/(imsize**2)
pyfits.writeto(input_path[0] + 'background.fits', im, overwrite=True)


pos1 = np.random.uniform(1, 15, size=int(no_data/2))
pos2 = np.random.uniform(10, 60, size=int(no_data/2))
for i in range(int(no_data)):
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
    pyfits.writeto(input_path[0]+'no_conv_rings_gauss'+str(i)+'.fits', ells, overwrite=True)
    kernel = Gaussian2DKernel(x_stddev=2, x_size=71, y_size=71)
    stropy_conv = convolve(ells, kernel)
    stropy_conv = stropy_conv / np.sum(stropy_conv)
    pyfits.writeto(input_path[0]+'rings_gauss'+str(i)+'.fits', stropy_conv, overwrite=True)

pdb.set_trace()

