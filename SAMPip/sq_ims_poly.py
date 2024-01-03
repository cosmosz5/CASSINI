import numpy as np
import os

for j in range(1):
    data = ['CALIB_SIM_DATA_uncalib_scene1_cube_sci.oifits']
    #Pixel size:
    s = 3
    #Image size:
    w= 1024
    #Regularizer:
    r_type = ' -la 1000 '
    chains = 5
    # Output file
    #initial_im = 'disk+gaussian.fits'
    out_file = ['SQUEEZE_test2/Scene1.fits'] #Output directory and output filename


    for i in range(len(data)):
        os.system('squeeze '+data[i]+' -s '+str(s)+' -w '+ str(w) + r_type + \
             ' -not3amp -novis -f_copy 0.2 -f_any 0.9 -n 1000 -o '+out_file[i]+' -chains '+str(chains))


