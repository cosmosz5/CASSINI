####### Import the necessary modules ############
import pdb
import calibrate_SAM
import os

## For calibration
sc_files = 'sci_data_480.txt'
cal_files = 'cal_data_480.txt'
delta = 1000.0  ### Hours
calibrate_SAM.calibrate_SAM(sc_files, cal_files, delta)
