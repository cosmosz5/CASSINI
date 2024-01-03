import numpy as np
import astropy.io.fits as pyfits
import pdb
import oitools
import sam_tools
import js_oifits as oifits

observables = oifits.open('CALIB_Cal_new_SCI_uncalib_u_corr_F380M_WR137_BPfix_calints.oifits')

vis = observables.vis2
cp = observables.t3

sam_tools.plot_v2_cp_calibrated('WR137', 'JWST-NIRISS', 21, 35, 3.8e-6, vis.ucoord, vis.vcoord, cp.u1coord, cp.v1coord, cp.u2coord, cp.v2coord, vis.vis2data,  \
               vis.vis2err, cp.t3phi, cp.t3phierr)







pdb.set_trace()
