import numpy as np
import pdb
import astropy.io.fits as pyfits
import cv2

im = pyfits.getdata('FT_MEAN_chain4.fits')
aa = np.zeros([70,70])
cv2.circle(aa, (35,35), 15, color=1, thickness=-1)

pyfits.writeto('mask.fits', aa, overwrite=True)

ind = np.where(aa == 0.0)
aa2 = np.zeros([70,70])
aa2[ind] = 1.0

masked = im * aa2

peak = np.max(im)
background = np.mean(masked)
print(peak/background)


pdb.set_trace()