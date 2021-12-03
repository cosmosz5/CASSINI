import numpy as np
import astropy.io.fits as pyfits
import readcol as readcol
import pandas as pd
import js_oifits as oifits
import pdb

def calibrate_SAM(calibrator_filename, science_filename, delta=2.0):

    [sc_files] = readcol.readcol(calibrator_filename, twod=False)
    [cal_files] = readcol.readcol(science_filename, twod=False)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    for i in range(len(sc_files)):
        print(sc_files[i])
        sc_data = pyfits.open(sc_files[i])
        sc_v = sc_data['OI_VIS'].data['VISAMP']
        sc_v_err = sc_data['OI_VIS'].data['VISAMPERR']
        sc_vphi = sc_data['OI_VIS'].data['VISPHI']
        sc_vphi_err = sc_data['OI_VIS'].data['VISPHIERR']
        sc_v2 = sc_data['OI_VIS2'].data['VIS2DATA']
        sc_v2_err = sc_data['OI_VIS2'].data['VIS2ERR']
        sc_t3amp = sc_data['OI_T3'].data['T3AMP']
        sc_t3amp_err = sc_data['OI_T3'].data['T3AMPERR']
        sc_t3phi = sc_data['OI_T3'].data['T3PHI']
        sc_t3phi_err = sc_data['OI_T3'].data['T3PHIERR']
        sc_mjd = sc_data['OI_VIS2'].data['MJD'][0]

        ### Define the arrays for the calibrators: ####
        cal_v = np.zeros([len(cal_files), len(sc_v)])
        cal_v_err = np.zeros([len(cal_files), len(sc_v)])
        cal_vphi = np.zeros([len(cal_files), len(sc_v)])
        cal_vphi_err = np.zeros([len(cal_files), len(sc_v)])
        cal_v2 = np.zeros([len(cal_files), len(sc_v2)])
        cal_v2_err = np.zeros([len(cal_files), len(sc_v2)])
        cal_t3amp = np.zeros([len(cal_files), len(sc_t3phi)])
        cal_t3amp_err = np.zeros([len(cal_files), len(sc_t3phi)])
        cal_t3phi = np.zeros([len(cal_files), len(sc_t3phi)])
        cal_t3phi_err = np.zeros([len(cal_files), len(sc_t3phi)])
        cal_mjd = np.zeros([len(cal_files)])

        for j in range(len(cal_files)):
            cal_data = pyfits.open(cal_files[j])
            cal_v[j, :] = cal_data['OI_VIS'].data['VISAMP']
            cal_v_err[j, :] = cal_data['OI_VIS'].data['VISAMPERR']
            cal_vphi[j, :] = cal_data['OI_VIS'].data['VISPHI']
            cal_vphi_err[j, :] = cal_data['OI_VIS'].data['VISPHIERR']
            cal_v2[j, :] = cal_data['OI_VIS2'].data['VIS2DATA']
            cal_v2_err[j, :] = cal_data['OI_VIS2'].data['VIS2ERR']
            cal_t3amp[j, :] = cal_data['OI_T3'].data['T3AMP']
            cal_t3amp_err[j, :] = cal_data['OI_T3'].data['T3AMPERR']
            cal_t3phi[j, :] = cal_data['OI_T3'].data['T3PHI']
            cal_t3phi_err[j, :] = cal_data['OI_T3'].data['T3PHIERR']
            cal_mjd[j] = cal_data['OI_VIS2'].data['MJD'][0]

        diff_MJD = (sc_mjd - cal_mjd) * 24.0  # Convert to hours
        [ind_MJD] = np.where(np.abs(diff_MJD) <= delta)

        if len(ind_MJD) == 1:
            calib_v2 = sc_v2 / np.squeeze(cal_v2[ind_MJD, :])
            #calib_v2_err = sc_v2_err
            calib_v2_err = calib_v2 * (np.sqrt(
                (sc_v2_err / sc_v2) ** 2 + (np.squeeze(cal_v2_err[ind_MJD, :]) / np.squeeze(cal_v2[ind_MJD, :])) ** 2))
            calib_v = sc_v / np.squeeze(cal_v[ind_MJD, :])  #### CALIBRATED VIS
            #calib_v_err = sc_v_err
            calib_v_err = calib_v * (np.sqrt(
                (sc_v_err / sc_v) ** 2 + (np.squeeze(cal_v_err[ind_MJD, :]) / np.squeeze(cal_v[ind_MJD, :])) ** 2))
            calib_visphi = sc_vphi - np.squeeze(cal_vphi[ind_MJD, :])  #### CALIBRATED  VISPHI
            #calib_visphi_err = sc_vphi_err
            calib_visphi_err = np.sqrt(
                (sc_vphi_err) ** 2 + (np.squeeze(cal_vphi_err[ind_MJD, :])) ** 2)  ### CALIBRATED VISPHIERR
            calib_t3amp = sc_t3amp / np.squeeze(cal_t3amp[ind_MJD, :])  #### CALIBRATED  T3AMP
            #calib_t3amp_err = sc_t3amp_err
            calib_t3amp_err = calib_t3amp * (np.sqrt((sc_t3amp_err / sc_t3amp) ** 2 + (
                        np.squeeze(cal_t3amp_err[ind_MJD, :]) / np.squeeze(
                    cal_t3amp[ind_MJD, :])) ** 2))  ### CALIBRATED T3AMPERR
            calib_t3phi = sc_t3phi - np.squeeze(cal_t3phi[ind_MJD, :])  #### CALIBRATED  T3PHI
            #calib_t3phi_err = sc_t3phi_err
            calib_t3phi_err = np.sqrt(
                (sc_t3phi_err) ** 2 + (np.squeeze(cal_t3phi_err[ind_MJD, :])) ** 2)  ### CALIBRATED T3PHIERR

        else:
            cal_v = np.mean(cal_v[ind_MJD, :], axis=0)
            cal_v_err = np.std(cal_v, axis=0) / np.sqrt(len(ind_MJD))
            cal_vphi = np.mean(cal_vphi[ind_MJD, :], axis=0)
            cal_vphi_err = np.std(cal_vphi, axis=0) / np.sqrt(len(ind_MJD))
            cal_v2 = np.mean(cal_v2[ind_MJD, :], axis=0)
            cal_v2_err = np.std(cal_v2, axis=0) / np.sqrt(len(ind_MJD))
            cal_t3amp = np.mean(cal_t3amp[ind_MJD, :], axis=0)
            cal_t3amp_err = np.std(cal_t3amp, axis=0) / np.sqrt(len(ind_MJD))
            cal_t3phi = np.mean(cal_t3phi[ind_MJD, :], axis=0)
            cal_t3phi_err = np.std(cal_t3phi, axis=0) / np.sqrt(len(ind_MJD))

            calib_v2 = sc_v2 / np.squeeze(cal_v2)
            #calib_v2_err = sc_v2_err
            calib_v2_err = calib_v2 * (
                np.sqrt((sc_v2_err / sc_v2) ** 2 + (np.squeeze(cal_v2_err) / np.squeeze(cal_v2)) ** 2))
            calib_v = sc_v / np.squeeze(cal_v)  #### CALIBRATED VIS
            #calib_v_err = sc_v_err
            calib_v_err = calib_v * (
                np.sqrt((sc_v_err / sc_v) ** 2 + (np.squeeze(cal_v_err) / np.squeeze(cal_v)) ** 2))
            calib_visphi = sc_vphi - np.squeeze(cal_vphi)  #### CALIBRATED  VISPHI
            #calib_visphi_err = sc_vphi_err
            calib_visphi_err = np.sqrt(
                (sc_vphi_err) ** 2 + (np.squeeze(cal_vphi_err)) ** 2)  ### CALIBRATED VISPHIERR
            calib_t3amp = sc_t3amp / np.squeeze(cal_t3amp)  #### CALIBRATED  T3AMP
            #calib_t3amp_err = sc_t3amp_err
            calib_t3amp_err = calib_t3amp * (np.sqrt((sc_t3amp_err / sc_t3amp) ** 2 + (
                    np.squeeze(cal_t3amp_err) / np.squeeze(cal_t3amp)) ** 2))  ### CALIBRATED T3AMPERR
            calib_t3phi = sc_t3phi - np.squeeze(cal_t3phi)  #### CALIBRATED  T3PHI
            #calib_t3phi_err = sc_t3phi_err
            calib_t3phi_err = np.sqrt(
                (sc_t3phi_err) ** 2 + (np.squeeze(cal_t3phi_err)) ** 2)  ### CALIBRATED T3PHIERR


        ##### TO SAVE THE CALIBRATED DATA INTO THE OIFITS FILE #########
        oi_file = oifits.open(sc_files[i])
        vis_file = oi_file.vis
        vis2_file = oi_file.vis2
        t3phi_file = oi_file.t3

        calib_oifile = oifits.oifits()
        calib_oifile.array = oi_file.array
        calib_oifile.target = oi_file.target
        calib_oifile.wavelength = oi_file.wavelength

        oi_file.vis = oifits.OI_VIS(1, vis_file.dateobs, vis_file.arrname, vis_file.insname, vis_file.target_id, vis_file.time, vis_file.mjd, vis_file.int_time, calib_v.T, calib_v_err.T, calib_visphi.T, calib_visphi_err.T, vis_file.ucoord, vis_file.vcoord, vis_file.sta_index, vis_file.flag)

        oi_file.vis2 = oifits.OI_VIS2(1, vis2_file.dateobs, vis2_file.arrname, vis2_file.insname, vis2_file.target_id,
                                      vis2_file.time, \
                                      vis2_file.mjd, vis2_file.int_time, calib_v2, calib_v2_err, vis2_file.ucoord, \
                                      vis2_file.vcoord, vis2_file.sta_index, vis2_file.flag)

        oi_file.t3 = oifits.OI_T3(1, t3phi_file.dateobs, t3phi_file.arrname, t3phi_file.insname, t3phi_file.target_id,
                                  t3phi_file.time, \
                                  t3phi_file.mjd, t3phi_file.int_time, calib_t3amp, calib_t3amp_err,
                                  calib_t3phi, \
                                  calib_t3phi_err, t3phi_file.u1coord, t3phi_file.v1coord, t3phi_file.u2coord, \
                                  t3phi_file.v2coord, t3phi_file.sta_index, t3phi_file.flag)


        
        oi_file.write('CALIB_' + sc_files[i])

    return 0




