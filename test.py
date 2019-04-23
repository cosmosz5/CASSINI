import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import pdb
from scipy.linalg import dft

im = pyfits.getdata('2004beauty.fits')

# im = np.zeros([4,4])
# k = 0
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         im[i,j] = k
#         k = k + 1

sz0 = np.shape(im)
pyfits.writeto('im.fits', im, overwrite=True)
im2 = np.swapaxes(im,1,0)
im = im2.reshape(-1)
sz = np.shape(im)
sizes = sz[0]
Tmatrix = dft(sizes)

ft_im2 = Tmatrix.dot(im)
pyfits.writeto('new_ftim_norot.fits', np.abs(ft_im2.reshape(sz0[0],sz0[1])), overwrite=True)

for i in range(int(sizes/sz0[0])):
    index = np.linspace(i*sz0[0], i*sz0[0]+sz0[0], endpoint=False, num=sz0[0], dtype='int')
    print index
    Tmatrix[index, :] = np.roll(Tmatrix[index,:], axis=0, shift=sz0[0]/2)
    print Tmatrix[index, :]
#Tmatrix = np.roll(Tmatrix, axis=0, shift=sizes/2)

ft_im2 = Tmatrix.dot(im)
pyfits.writeto('new_ftim.fits', np.abs(ft_im2.reshape(sz0[0],sz0[1])), overwrite=True)

ft_im = np.abs(np.fft.fft2(im.reshape(sz0[0],sz0[1])))
pyfits.writeto('classic_ftim_norot.fits', ft_im, overwrite=True)
ft_im = np.fft.fftshift(np.abs(np.fft.fft2(im.reshape(sz0[0],sz0[1]))))
pyfits.writeto('classic_ftim.fits', ft_im, overwrite=True)


pdb.set_trace()