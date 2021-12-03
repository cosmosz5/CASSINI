import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import *
import scipy.ndimage as ndimage
from PIL import Image
import pdb

input_path = ['/Users/donut/Library/Mobile Documents/com~apple~CloudDocs/JWST_CS/dataset/disks/']
output_path = ['/Users/donut/Library/Mobile Documents/com~apple~CloudDocs/JWST_CS/dataset/disks_clean/']
input_files = readcol(input_path[0] +'data.txt', twod=True)

for i in range(len(input_files)):
    temp = input_path[0]+input_files[i][0]
    im = plt.imread(temp)
    im = np.mean(im, axis=2)
    im2 = ndimage.zoom(im, (128/im.shape[0], 128/im.shape[1]))
    im2 = im2 / np.sum(im2)
    pyfits.writeto(output_path[0]+'c_'+input_files[i][0][:-3]+'fits', im2, overwrite=True)

pdb.set_trace()