import numpy as np
import astropy.io.fits as pyfits
import pdb
import matplotlib.pyplot as plt
from readcol import *

[target] = readcol('dats_0.0001.txt', twod=False)

for i in range(len(target)):
    data, head = pyfits.getdata(target[i], header=True)
    pyfits.writeto('corr_'+target[i], data, header=head, overwrite=True)

pdb.set_trace()
