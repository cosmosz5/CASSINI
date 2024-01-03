####### Import the necessary modules ############
import sim_SAM
import pdb
import calibrate_SAM
import os
import pylab
import matplotlib.pyplot as plt
plt.ion()

#######  The following parameters are necessary to run the SAM pipelie ##################
mask_filename = '7holes_jwst_mask_corr.txt' #A text file with the mask geometry
wave = 4.817e-06 #3.828e-06 #4.817e-6 #3.828e-06 #4.2934e-6 #Central wavelength of the observations (meters)
bandwidth = 0.298e-06 #1.93756431390019e-07 #0.205e-06 #0.298e-06 #0.205e-06 #0.202e-6 #Bandwidth of the observations (meters)
hole_size = 0.82 #in meters
imsize = 80 # in pixels
px_scalex = 65.3249 #in mas
px_scaley = 65.7226
hole_geom = 'HEXAGON' #HEXAGON for the JWST
inst = 'JWST' 
arrname = 'DATA' ## This could be DATA or SIM
rotation_angle = -0.36 ### In case the mask is not propely aligned with the position indicated in the manual
oversample = 1.0 ## Number of times that you want to oversample the data

scale_factor = [0.996, 0.999]
center_factor = [0,0]

data_filename = 'data_WR137_bpfix_480.txt'
source = 'WR137'
sim_SAM.simSAM_PSF (data_filename, mask_filename,  wave, bandwidth, hole_size, px_scalex, px_scaley, imsize, hole_geom, source, inst, \
arrname, rotation_angle, oversample, scale_factor, center_factor, affine=False)

data_filename = 'data_cal_bpfix_480.txt'
source = 'Cal'
sim_SAM.simSAM_PSF (data_filename, mask_filename,  wave, bandwidth, hole_size, px_scalex, px_scaley, imsize, hole_geom, source, inst, \
arrname, rotation_angle, oversample, scale_factor, center_factor, affine=False)


wave = 3.82650365041922e-06 #3.828e-06 #4.817e-6 #3.828e-06 #4.2934e-6 #Central wavelength of the observations (meters)
bandwidth = 0.205e-06 #1.93756431390019e-07 #0.205e-06 #0.298e-06

data_filename = 'data_WR137_bpfix_380.txt'
source = 'WR137'
sim_SAM.simSAM_PSF (data_filename, mask_filename,  wave, bandwidth, hole_size, px_scalex, px_scaley, imsize, hole_geom, source, inst, \
arrname, rotation_angle, oversample, scale_factor, center_factor, affine=False)

data_filename = 'data_cal_bpfix_380.txt'
source = 'Cal'
sim_SAM.simSAM_PSF (data_filename, mask_filename,  wave, bandwidth, hole_size, px_scalex, px_scaley, imsize, hole_geom, source, inst, \
arrname, rotation_angle, oversample, scale_factor, center_factor, affine=False)

