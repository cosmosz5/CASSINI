####### Import the necessary modules ############
import sim_SAM_SPHERE
import pdb
import calibrate_SAM
import os

## For calibration
sc_files = 'sci_oi.txt'
cal_files = 'cal_oi.txt'
delta = 0.3  ### Hours
calibrate_SAM.calibrate_SAM(sc_files, cal_files, delta)


#sc_files = 'sci_oi_K2.txt'
#cal_files = 'cal_oi_K2.txt'
#delta = 0.3  ### Hours
#calibrate_SAM.calibrate_SAM(sc_files, cal_files, delta)


sc_files = 'sci_oi_K2.txt'
cal_files = 'cal_oi_K2.txt'
delta = 0.3  ### Hours
calibrate_SAM.calibrate_SAM(sc_files, cal_files, delta)
