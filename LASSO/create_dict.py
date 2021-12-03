import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import *
import oitools
import pylab
from skimage.restoration import unwrap_phase
import pdb

def unwrap(signal):
    for i in range(len(signal)-1):
        difference = signal[i+1] - signal[i];
        if difference > np.pi:
            signal[i:] = signal[i:] - 2 * np.pi
        elif difference < -np.pi:
            signal[i:] = signal[i:] + 2 * np.pi

    return signal

oi_data = 'COMB_JWST_SAM_tot.fits'
#input_path = ['/Users/donut/Library/Mobile Documents/com~apple~CloudDocs/JWST_CS/dataset/disks_clean/']
input_path = ['/Users/donut/Desktop/dataset/']
input_files = readcol(input_path[0] +'ring_gauss.txt', twod=True)
scale = 10.0 ## mas

aa = 0
for i in range(len(input_files)):
    temp = input_path[0] + input_files[i][0]
    im_atom = np.squeeze(pyfits.getdata(temp))
    ind = np.where(np.isnan(im_atom) == True)


    if len(ind[0]) > 0:
        if aa == 0:
            bad_files = i
            aa = 1
        else:
            bad_files = np.append(bad_files, i)
            print('Hola')

    if len(ind[0]) == 0:
        print('Caca')
        if i == 0:
            v_model, phi_model, t3phi_model = oitools.compute_obs(oi_data, im_atom, scale)

            v2_model = np.reshape(v_model**2, -1)
            phi_model = np.reshape(phi_model, -1)
            t3phi_model = np.reshape(t3phi_model, -1)

            [ind1] = np.where(v2_model == 0.0)
            v2_model[ind1] = 1e-16

            [ind1] = np.where(phi_model == 0.0)
            phi_model[ind1] = 1e-16

            [ind1] = np.where(t3phi_model == 0.0)
            t3phi_model[ind1] = 1e-16

            v2_model = v2_model - np.mean(v2_model)
            v2_model = v2_model / np.sqrt(np.sum(v2_model**2))
            phi_model = np.tan(np.deg2rad(phi_model))
            phi_model = phi_model - np.mean(phi_model)
            phi_model = phi_model / np.sqrt(np.sum(phi_model**2))
            t3phi_model = np.tan(np.deg2rad(t3phi_model))
            t3phi_model = t3phi_model - np.mean(t3phi_model)
            t3phi_model = t3phi_model / np.sqrt(np.sum(t3phi_model ** 2))
            Dict = np.append(np.append(v2_model, phi_model), t3phi_model)

        else:
            v_model_temp, phi_model_temp, t3phi_model_temp = oitools.compute_obs(oi_data, im_atom, scale)
            v2_model_temp = np.reshape(v_model_temp**2, -1)
            phi_model_temp = np.reshape(phi_model_temp, -1)
            t3phi_model_temp = np.reshape(t3phi_model_temp, -1)

            [ind1] = np.where(v2_model_temp == 0.0)
            v2_model_temp[ind1] = 1e-16

            [ind1] = np.where(phi_model_temp == 0.0)
            phi_model_temp[ind1] = 1e-16

            [ind1] = np.where(t3phi_model_temp == 0.0)
            t3phi_model_temp[ind1] = 1e-16

            v2_model_temp = v2_model_temp - np.mean(v2_model_temp)
            v2_model_temp = v2_model_temp / np.sqrt(np.sum(v2_model_temp ** 2))
            phi_model_temp = np.tan(np.deg2rad(phi_model_temp))
            phi_model_temp = phi_model_temp - np.mean(phi_model_temp)
            phi_model_temp = phi_model_temp / np.sqrt(np.sum(phi_model_temp ** 2))
            t3phi_model_temp = np.tan(np.deg2rad(t3phi_model_temp))
            t3phi_model_temp = t3phi_model_temp - np.mean(t3phi_model_temp)
            t3phi_model_temp = t3phi_model_temp / np.sqrt(np.sum(t3phi_model_temp ** 2))

            Dict_temp = np.append(np.append(v2_model_temp, phi_model_temp), t3phi_model_temp)
            Dict = np.dstack((Dict, Dict_temp))

np.save('Dict2.npy', np.squeeze(Dict))

print(bad_files)

a_file = open(input_path[0] +'ring_gauss.txt', "r")
lines = a_file.readlines()
a_file.close()

del lines[bad_files]
new_file = open("sample.txt", "w+")
for line in lines:
    new_file.write(line)
new_file.close()


pdb.set_trace()

