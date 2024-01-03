####### Import the necessary modules ############
import sim_SAM
import pdb
import calibrate_SAM_obo
import readcol
import os
import astropy.io.fits as pyfits

input_sci_path = ''
input_cal_path = ''
calibrator_filename = 'cal_data_bpfix_380.txt'
science_filename = 'sci_data_bpfix_380.txt'
cal_name = 'Cal_new'

[sc_files] = readcol.readcol(input_sci_path+science_filename, twod=False)
[cal_files] = readcol.readcol(input_cal_path+calibrator_filename, twod=False)
for i in range(len(sc_files)):
    sc_dat = sc_files[i]
    cal_dat = cal_files[0]
    delta = 1000.0  ### Hours
    calibrate_SAM_obo.calibrate_SAM(input_sci_path, input_cal_path, sc_dat, cal_dat, cal_name, delta)
    print('done')
