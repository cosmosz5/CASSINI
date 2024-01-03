import numpy as np
import os

for j in range(1):
    data = ['CALIB_SIM_DATA_uncalib_scene3_cube_sci.oifits']
    #Pixel size:
    s = 13
    #Image size:
    w= 257
    #Regularizer:
    r_type = ' -tv 50 -l0 0.9 '
    chains = 5
    # Output file
    #initial_im = 'disk+gaussian.fits'
    out_file = ['SQUEEZE_scene3/COMB.fits']


    for i in range(len(data)):
        os.system('squeeze '+data[i]+' -s '+str(s)+' -w '+ str(w) + r_type + \
             ' -not3amp -novis -v2a 0.01 -f_copy 0.2 -f_any 0.1 -e 900 -n 500 -o '+out_file[i]+' -chains '+str(chains))


