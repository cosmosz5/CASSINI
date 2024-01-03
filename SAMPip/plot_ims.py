import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import matplotlib.cm as cm

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

r = pyfits.getdata('test_480.fits')[64-16:64+17, 64-16:64+17]
g = np.zeros_like(r)
b = pyfits.getdata('test_380.fits')[64-16:64+17, 64-16:64+17]

rgb = np.dstack((r,b)) 
rgb_uint8 = np.dstack((g,r,b))
im = rgb2gray(rgb_uint8)
fig1, ax1 = plt.subplots(1,1)
levs = np.array([0.07, 0.1, 0.3, 0.5, 0.7, 0.9]) * np.max(np.sqrt(b))
ax1.imshow(np.sqrt(r), cmap=cm.gnuplot, vmin=0, vmax=np.max(np.sqrt(r)), extent=[20*16, -20*16, -20*16, 20*16], origin='lower')
ax1.contour(np.sqrt(b), extent=[20*16, -20*16, -20*16, 20*16], levels = levs, colors = 'white', linewidths=1)


ax1.set_xlabel('Milliarcseconds [mas]')
ax1.set_ylabel('Milliarcseconds [mas]')
plt.show()
fig1.savefig('WR137.png', bbox_inches='tight')

pdb.set_trace()
