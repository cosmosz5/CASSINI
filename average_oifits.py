import pdb
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import js_oifits as oifits
import numpy as np
from readcol import *

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

[sc_files] = readcol('CALIBRATED_oifits.txt', twod=False)

for i in range(len(sc_files)):
    if i == 0:
        oidata = pyfits.open(sc_files[i])
        vis_time0 = oidata['OI_VIS'].data['TIME']
        vis_mjd0 = oidata['OI_VIS'].data['MJD']
        vis_visamp0 = oidata['OI_VIS'].data['VISAMP']
        vis_visamperr0 = oidata['OI_VIS'].data['VISAMPERR']
        vis_phi0 = oidata['OI_VIS'].data['VISPHI']
        vis_phierr0 = oidata['OI_VIS'].data['VISPHIERR']
        vis_ucoord0 = oidata['OI_VIS'].data['UCOORD']
        vis_vcoord0 = oidata['OI_VIS'].data['VCOORD']

        vis2_time0 = oidata['OI_VIS2'].data['TIME']
        vis2_mjd0 = oidata['OI_VIS2'].data['MJD']
        vis2_visamp0 = oidata['OI_VIS2'].data['VIS2DATA']
        vis2_visamperr0 = oidata['OI_VIS2'].data['VIS2ERR']
        vis2_ucoord0 = oidata['OI_VIS2'].data['UCOORD']
        vis2_vcoord0 = oidata['OI_VIS2'].data['VCOORD']


        t3_time0 = oidata['OI_T3'].data['TIME']
        t3_mjd0 = oidata['OI_T3'].data['MJD']
        t3_t3phi0 = oidata['OI_T3'].data['T3PHI']
        t3_t3phierr0 = oidata['OI_T3'].data['T3PHIERR']
        t3_u1coord0 = oidata['OI_T3'].data['U1COORD']
        t3_v1coord0 = oidata['OI_T3'].data['V1COORD']
        t3_u2coord0 = oidata['OI_T3'].data['U2COORD']
        t3_v2coord0 = oidata['OI_T3'].data['V2COORD']
    else:
        oidata = pyfits.open(sc_files[i])
        vis_time0 = np.dstack((vis_time0, oidata['OI_VIS'].data['TIME']))
        vis_mjd0 = np.dstack((vis_mjd0, oidata['OI_VIS'].data['MJD']))
        vis_visamp0 = np.dstack((vis_visamp0, oidata['OI_VIS'].data['VISAMP']))
        vis_visamperr0 = np.dstack((vis_visamperr0, oidata['OI_VIS'].data['VISAMPERR']))
        vis_phi0 = np.dstack((vis_phi0, oidata['OI_VIS'].data['VISPHI']))
        vis_phierr0 = np.dstack((vis_phierr0, oidata['OI_VIS'].data['VISPHIERR']))
        vis_ucoord0 = np.dstack((vis_ucoord0, oidata['OI_VIS'].data['UCOORD']))
        vis_vcoord0 = np.dstack((vis_vcoord0, oidata['OI_VIS'].data['VCOORD']))

        vis2_time0 = np.dstack((vis2_time0, oidata['OI_VIS2'].data['TIME']))
        vis2_mjd0 = np.dstack((vis2_mjd0, oidata['OI_VIS2'].data['MJD']))
        vis2_visamp0 = np.dstack((vis2_visamp0, oidata['OI_VIS2'].data['VIS2DATA']))
        vis2_visamperr0 = np.dstack((vis2_visamperr0, oidata['OI_VIS2'].data['VIS2ERR']))
        vis2_ucoord0 = np.dstack((vis2_ucoord0, oidata['OI_VIS2'].data['UCOORD']))
        vis2_vcoord0 = np.dstack((vis2_vcoord0, oidata['OI_VIS2'].data['VCOORD']))

        t3_time0 = np.dstack((t3_time0, oidata['OI_T3'].data['TIME']))
        t3_mjd0 = np.dstack((t3_mjd0, oidata['OI_T3'].data['MJD']))
        t3_t3phi0 = np.dstack((t3_t3phi0, oidata['OI_T3'].data['T3PHI']))
        t3_t3phierr0 = np.dstack((t3_t3phierr0, oidata['OI_T3'].data['T3PHIERR']))
        t3_u1coord0 = np.dstack((t3_u1coord0, oidata['OI_T3'].data['U1COORD']))
        t3_v1coord0 = np.dstack((t3_v1coord0, oidata['OI_T3'].data['V1COORD']))
        t3_u2coord0 = np.dstack((t3_u2coord0, oidata['OI_T3'].data['U2COORD']))
        t3_v2coord0 = np.dstack((t3_v2coord0, oidata['OI_T3'].data['V2COORD']))


vis_time = np.mean(vis_time0[0,:,:], axis=1)
vis_mjd = np.mean(vis_mjd0[0,:,:], axis=1)
vis_visamp = np.zeros([vis_visamp0.shape[1]])
vis_visamperr = np.zeros([vis_visamp0.shape[1]])
vis_visphi = np.zeros([vis_visamp0.shape[1]])
vis_visphierr = np.zeros([vis_visamp0.shape[1]])
for i in range(vis_visamp.shape[0]):
    vis_visamp[i], vis_visamperr[i] = weighted_avg_and_std(vis_visamp0[0,i,:], 1./(vis_visamperr0[0,i,:]))
    vis_visphi[i], vis_visphierr[i] = weighted_avg_and_std(vis_phi0[0, i, :], 1. / (vis_phierr0[0, i, :]))
vis_ucoord = np.mean(vis_ucoord0[0,:,:], axis=1)
vis_vcoord = np.mean(vis_vcoord0[0,:,:], axis=1)

vis2_time = np.mean(vis2_time0[0,:,:], axis=1)
vis2_mjd = np.mean(vis2_mjd0[0,:,:], axis=1)
vis2_visamp = np.zeros([vis2_visamp0.shape[1]])
vis2_visamperr = np.zeros(vis2_visamp0.shape[1])
for i in range(vis2_visamp.shape[0]):
    vis2_visamp[i], vis2_visamperr[i] = weighted_avg_and_std(vis2_visamp0[0,i,:], 1./(vis2_visamperr0[0,i,:]))
vis2_ucoord = np.mean(vis2_ucoord0[0,:,:], axis=1)
vis2_vcoord = np.mean(vis2_vcoord0[0,:,:], axis=1)

t3_time = np.mean(t3_time0[0,:,:], axis=1)
t3_mjd = np.mean(t3_mjd0[0,:,:], axis=1)
t3_t3phi = np.zeros([t3_t3phi0.shape[1]])
t3_t3phierr = np.zeros([t3_t3phi0.shape[1]])
for i in range(t3_t3phi.shape[0]):
    t3_t3phi[i], t3_t3phierr[i] = weighted_avg_and_std(t3_t3phi0[0,i,:], 1. / (t3_t3phierr0[0, i, :]) )
t3_t3amp = np.copy(t3_t3phi) * 0.0 + 1.0
t3_t3amperr = np.copy(t3_t3phi) * 0.0 + 1.0

t3_u1coord = np.mean(t3_u1coord0[0,:,:], axis=1)
t3_v1coord = np.mean(t3_v1coord0[0,:,:], axis=1)
t3_u2coord = np.mean(t3_u2coord0[0,:,:], axis=1)
t3_v2coord = np.mean(t3_v2coord0[0,:,:], axis=1)


##### TO SAVE THE CALIBRATED DATA INTO THE OIFITS FILE #########
oi_file = oifits.open(sc_files[0])
vis_file = oi_file.vis
vis2_file = oi_file.vis2
t3phi_file = oi_file.t3

calib_oifile = oifits.oifits()
calib_oifile.array = oi_file.array
calib_oifile.target = oi_file.target
calib_oifile.wavelength = oi_file.wavelength
target = oi_file.target.target[0]

calib_oifile.vis = oifits.OI_VIS(1, vis_file.dateobs, vis_file.arrname, vis_file.insname, vis_file.target_id, vis_time,
                            vis_mjd, \
                            vis_file.int_time, vis_visamp, vis_visamperr, vis_visphi, vis_visphierr, vis_ucoord, \
                            vis_vcoord, vis_file.sta_index, vis_file.flag)

calib_oifile.vis2 = oifits.OI_VIS2(1, vis2_file.dateobs, vis2_file.arrname, vis2_file.insname, vis2_file.target_id,
                              vis2_time, \
                              vis2_mjd, vis2_file.int_time, vis2_visamp, vis2_visamperr, vis2_ucoord, \
                              vis2_vcoord, vis2_file.sta_index, vis2_file.flag)

calib_oifile.t3 = oifits.OI_T3(1, t3phi_file.dateobs, t3phi_file.arrname, t3phi_file.insname, t3phi_file.target_id,
                          t3_time, \
                          t3_mjd, t3phi_file.int_time, t3_t3amp, t3_t3amperr, t3_t3phi, \
                          t3_t3phierr, t3_u1coord, t3_v1coord, t3_u2coord, \
                          t3_v2coord, t3phi_file.sta_index, t3phi_file.flag)

calib_oifile.write('AVERAGED_FILE_'+ target +'.oifits')




pdb.set_trace()