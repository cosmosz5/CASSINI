import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import *
import oitools
import pylab
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import Lasso
import sklearn.preprocessing as prepro
import pdb

pylab.ion()
n_nonzero_coefs = 50

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def fista(A, b, err, nvis, nt3phi, l, maxit):
    import time
    from math import sqrt
    import numpy as np
    from scipy import linalg
    import sklearn.preprocessing as prepro

    x = np.zeros(A.shape[1])
    x = x.reshape(-1, 1)

    for ty in range(A.shape[1]):
        A[nvis:,ty] = np.arctan(np.tan(np.deg2rad(A[nvis:,ty])))
    A = prepro.normalize(A, norm='l2', axis=0)

    b[nvis:, 0] = np.arctan(np.tan(np.deg2rad(b[nvis:, 0])))
    b = prepro.normalize(b, norm='l2', axis=0)

    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A, ord=2) ** 2 #### Lipschitz constant
    time0 = time.time()
    for _ in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))
        print(this_pobj)
    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times


### INPUT PARAMETERS ###

oi_data = 'COMB_JWST_SAM_tot.fits'
dict_filename = 'Dict3.npy'
input_path = ['dataset/']
input_files = readcol(input_path[0] +'ring_gauss.txt', twod=True)
scale = 10.0 ## mas
hyperparameter = 0.01
maxit = 10000

output_filename = 'recovered_im_lasso.fits' #Output filename

####### AUTOMATIC FROM HERE #################

### Append, normalize and reshape the data :
obs = oitools.extract_data(oi_data)
v2_data = obs['vis2'].reshape(-1)
phi_data = obs['phi'].reshape(-1)
t3phi_data = obs['t3'].reshape(-1)

v2_data = prepro.scale(v2_data)
phi_data = prepro.scale(np.tan(np.deg2rad(phi_data)))
t3phi_data = prepro.scale(np.tan(np.deg2rad(t3phi_data)))
observables = np.append(np.append(v2_data, phi_data), t3phi_data)

### Open the dictionary
with open(dict_filename, 'rb') as f:
    Dict = np.load(f)

omp = Lasso(hyperparameter, max_iter = 1000)
omp.fit(Dict, observables)

coef = omp.coef_
idx_r, = coef.nonzero()
print(len(idx_r))
[ind] = np.where(np.squeeze(coef) !=0)

im_rec = 0.0
for mm in range(len(ind)):
    temp = input_path[0] + input_files[ind[mm]][0]
    ind2 = np.where(np.isnan(pyfits.getdata(temp)) == True)
    im_rec += coef[ind[mm]]*pyfits.getdata(temp)
pyfits.writeto(output_filename, im_rec, overwrite=True)


pdb.set_trace()

