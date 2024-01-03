import numpy as np
import astropy.io.fits as pyfits
import pdb
import oitools
import matplotlib.pyplot as plt
import js_oifits as oifits


oi_file = pyfits.open('temp.oifits')
im_filename = pyfits.getdata('model_bs.fits')

model_vis2, model_phase, model_t3phi = oitools.compute_obs('temp.oifits', im_filename, 7)


oi_file = oifits.open('temp.oifits')
vis_file = oi_file.vis
vis2_file = oi_file.vis2
t3phi_file = oi_file.t3

calib_oifile = oifits.oifits()
calib_oifile.array = oi_file.array
calib_oifile.target = oi_file.target
calib_oifile.wavelength = oi_file.wavelength

calib_oifile.vis = oifits.OI_VIS(1, vis_file.dateobs, vis_file.arrname, vis_file.insname, vis_file.target_id, vis_file.time, vis_file.mjd, vis_file.int_time, model_vis2, vis_file.visamperr, model_phase, vis_file.visphierr, vis_file.ucoord, vis_file.vcoord, vis_file.sta_index, vis_file.flag)

calib_oifile.vis2 = oifits.OI_VIS2(1, vis2_file.dateobs, vis2_file.arrname, vis2_file.insname, vis2_file.target_id,
                                      vis2_file.time, \
                                      vis2_file.mjd, vis2_file.int_time, model_vis2**2, vis2_file.vis2err, vis2_file.ucoord, \
                                      vis2_file.vcoord, vis2_file.sta_index, vis2_file.flag)

calib_oifile.t3 = oifits.OI_T3(1, t3phi_file.dateobs, t3phi_file.arrname, t3phi_file.insname, t3phi_file.target_id,
                                  t3phi_file.time, \
                                  t3phi_file.mjd, t3phi_file.int_time, t3phi_file.t3amp, t3phi_file.t3amperr,
                                  model_t3phi, \
                                  t3phi_file.t3phierr, t3phi_file.u1coord, t3phi_file.v1coord, t3phi_file.u2coord, \
                                  t3phi_file.v2coord, t3phi_file.sta_index, t3phi_file.flag)


        
calib_oifile.write('model_bowshock2.fits')


plt.show()
pdb.set_trace()
