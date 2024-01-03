import numpy as np
import astropy.io.fits as pyfits
import pdb
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from readcol import *
import cv2


[ims] = readcol('data_new_bpfix_480.txt', twod=False)
##14 for 380
## 16 for 480

for i in range(len(ims)):
    fits_cube = pyfits.open(ims[i])
    cube = fits_cube[1].data
    head = fits_cube[0].header
    head2 = fits_cube[1].header
    head['MJD-AVG'] = head2['MJD-AVG']
    head['ROLL_REF'] = head2['ROLL_REF']
    head['EXTNAME'] = head2['EXTNAME']
    #cube, head = pyfits.getdata(ims[i], header=True)
    c_cube = np.zeros_like(cube)
    
    for j in range(cube.shape[0]):
        
        indx, indy = np.where(cube[j,:,:] == np.max(cube[j,:,:]))
        im_temp = np.zeros([cube.shape[1], cube.shape[2]])
        cv2.circle(im_temp, (indy[0], indx[0]), 16, color=1, thickness=-1)
        pyfits.writeto('mask.fits', im_temp, overwrite=True)
        im = cube[j,:,:] * im_temp
        centroid = nd.center_of_mass(im)
        c_cube[j,:,:] = nd.shift(im, (40-centroid[0], 40 - centroid[1]), order=0)
    pyfits.writeto('corr_'+ims[i], c_cube, header=head, overwrite=True)
    

pdb.set_trace()
