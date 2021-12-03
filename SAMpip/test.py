####### Import the necessary modules ############
import sim_SAMpip
import pdb
import calibrate_SAM
import os
import pylab
pylab.ion()

#######  The tolowing parameters are necessary to run SAMPip ##################
mask_filename = '7holes_jwst_mask.txt' #A text file with the mask geometry
wave = 3.828e-06 #Central wavelength of the observations (meters)
bandwidth = 0.205e-06 #Bandwidth of the observations (meters)
hole_size = 0.82 #Pinhole size (meters)
imsize = 80 # Image size (pixels)
px_scale = 65.6 #Pixel scale (pixels)
hole_geom = 'HEXAGON' #Geometry of the pin-holes (hexagon for the JWST)
inst = 'JWST' #Instrument to be used
arrname = 'SIM' ## This could be DATA or SIM depending if the reduction is on real data or simulated ones
rotation_angle = 0.0 ### In case the non-redundant mask is not propely aligned with the nominal position
oversample = 1.0 ## Number of times to oversample the data (not tested yet)

data_filename = 'sci_psf3_w1.txt' ## Text file with the list of .fits files to be reduced
source = 'disk' ## The source name

#### Automatic from here, the following command produces an oifits file with all the data reduced in an OIFITS file

sim_SAMpip.simSAM_PSF (data_filename, mask_filename,  wave, bandwidth, hole_size, px_scale, imsize, hole_geom, source, inst, \
           arrname, rotation_angle, oversample)

print('PROGRAM FINISHED, TO EXTI THE DEBUGGER TYPE q ON THE TERMINAL')

pdb.set_trace()
