def weighted_avg_and_std(values, weights):
    import numpy as np
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

import numpy as np
to_rd = lambda m, d: m * np.exp(1j * np.deg2rad(d))
to_pd = lambda x: (abs(x), np.rad2deg(np.angle(x)))


def redSAM(data_file, PSF_im, uwindow, bl_x, bl_y, wave, hole_size, imsize, px_scale, nholes, nbl, ncp, \
           baselines, closure_phases, x, y, air, cube_bl, cube_bl_sin, xcoord, ycoord, source, bandwidth, instrument, \
           oversample, arrname = 'SIM'):
    from skimage.restoration import unwrap_phase
    import numpy as np
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    import sam_tools
    import pdb
    import pandas as pd
    import js_oifits as oifits
    from astropy.time import Time
    from datetime import datetime
    from scipy.stats import trim_mean
    import scipy.ndimage as sciim
    import image_registration.fft_tools as shift

    data_cube = pyfits.open(data_file)
    data_header = data_cube[0].header
    data_cube = data_cube[0].data
    sz = np.shape(data_cube)
    data_cube = sciim.zoom(data_cube, (1, oversample, oversample), order=3)

    sz = np.shape(data_cube)
    V2_mod = np.zeros([sz[0],nbl])
    PHASE_mod = np.zeros([sz[0],nbl])
    windowed_data = np.zeros([sz[0], sz[1], sz[2]])
    MODEL_IM = np.zeros([sz[0],imsize, imsize])
    MODEL_IM_TOT = np.zeros([sz[0],imsize, imsize])
    residuals = np.copy(MODEL_IM)
    shifted_data_cube = np.zeros([data_cube.shape[0], data_cube.shape[1], data_cube.shape[2]])

    for i in range(sz[0]):

        windowed_PSF = PSF_im * uwindow
        windowed_data[i,:,:] = data_cube[i,:,:] * uwindow
        ind = np.where(uwindow != 0.0)

        #### Compute the centroid of the data ####
        FT_IM = np.fft.fftshift(np.fft.ifft2(windowed_data[i, :, :]))
        xMid_via_FT = np.angle(FT_IM[int(imsize / 2 + 1), int(imsize / 2)]) / (2 * np.pi) * imsize + 1
        yMid_via_FT = np.angle(FT_IM[int(imsize / 2), int(imsize / 2 + 1)]) / (2 * np.pi) * imsize + 1

        yshift = int(imsize / 2) - (xMid_via_FT - 1)
        xshift = int(imsize / 2) - (yMid_via_FT - 1)

        aa = sciim.measurements.center_of_mass(windowed_data[i, :, :])
        #print(aa)

        #print(xshift, yshift)
        temp_shift = shift.shiftnd(data_cube[i, :, :], [xshift, yshift])

        ### Shift deactivated:

        windowed_data[i, :, :] = data_cube[i, :, :] * uwindow
        shifted_data_cube[i, :, :] = data_cube[i, :, :] * 1.0
        ##windowed_data[i, :, :] = temp_shift * uwindow
        ##shifted_data_cube[i, :, :] = temp_shift

        BB = windowed_data[i, ind[0], ind[1]]
        #### Define the matrix for the P2VM:
        P2VM = np.zeros([BB.shape[0], 2*nbl+2])

        P2VM[:, 0] = air[ind] * nholes
        P2VM[:, -1] = 1.0

        f=0; m = 0
        for k in range(2*nbl):
            if (k %2 == 0):
                P2VM[:, k+1] = air[ind] * 2. *cube_bl[f, ind[0], ind[1]]
                f = f + 1
            elif (k%2 !=0):
                P2VM[:, k + 1] = air[ind] * 2. *cube_bl_sin[m, ind[0], ind[1]]
                m = m + 1

        P2VMt = np.linalg.pinv(P2VM)
        MOD = np.dot(P2VMt,BB)
        MOD_fringes = np.dot(P2VM,MOD)
        MODEL_IM[i,ind[0], ind[1]] = MOD_fringes
        residuals[i, :,:] = MODEL_IM[i,:,:] / windowed_data[i,:,:]
        #pyfits.writeto('MODEL_interferogram_window_'+data_file, MODEL_IM, overwrite=True)

        P2VM = np.zeros([cube_bl.shape[1]**2, 2*nbl+2])
        P2VM[:, 0] = air.reshape(-1) * nholes
        P2VM[:, -1] = 1.0
        
        f=0; m = 0
        for k in range(2*nbl):
            if (k %2 == 0):
                P2VM[:, k+1] = air.reshape(-1) * 2. *np.squeeze(cube_bl[f, :, :]).reshape(-1)
                f = f + 1
            elif (k%2 !=0):
                P2VM[:, k + 1] = air.reshape(-1) * 2. *np.squeeze(cube_bl_sin[m, :, :]).reshape(-1)
                m = m + 1

        MOD_fringes = np.dot(P2VM,MOD)
        MODEL_IM_TOT[i,:, :] = MOD_fringes.reshape(data_cube.shape[1], data_cube.shape[2])

        for ll in range(nbl):
            V2_mod[i, ll] = (MOD[ll*2 +1]**2 + MOD[ll*2+2]**2) / MOD[0]**2
            PHASE_mod[i, ll] = np.arctan2(MOD[ll*2+2], MOD[ll*2+1])

    if arrname == 'SIM':
        mean_parang = -10.0
        bl_xp = bl_x * np.cos(np.deg2rad(mean_parang)) - bl_y * np.sin(np.deg2rad(mean_parang))
        bl_yp = bl_x * np.sin(np.deg2rad(mean_parang)) + bl_y * np.cos(np.deg2rad(mean_parang))
    elif (arrname == 'DATA') and (instrument == 'SPHERE'):
        ##### Transform the baselines according to the rotation of the parallactic angle:
        mean_parang = (-1.0*data_header['PARA_MIN'] - 1.0*data_header['PARA_MAX']) / 2.0
        bl_xp = -(bl_x * np.cos(np.deg2rad(mean_parang)) - bl_y * np.sin(np.deg2rad(mean_parang)))
        bl_yp = (bl_x * np.sin(np.deg2rad(mean_parang)) + bl_y * np.cos(np.deg2rad(mean_parang)))
    elif (arrname == 'DATA') and (instrument == 'NACO'):
        if 'STPOSANG' not in data_header:
            mean_parang = (data_header['HIERARCH ESO ADA POSANG'] + data_header['HIERARCH ESO ADA POSANG END']) / 2.0
            #mean_parang = - (360 + mean_parang)
        else:
            mean_parang = (data_header['STPOSANG'] + data_header['ENPOSANG']) / 2.0
            #mean_parang =  - (360 + mean_parang)
        bl_xp = -(bl_x * np.cos(np.deg2rad(mean_parang)) - bl_y * np.sin(np.deg2rad(mean_parang)))
        bl_yp = bl_x * np.sin(np.deg2rad(mean_parang)) + bl_y * np.cos(np.deg2rad(mean_parang))


    #### Average the Observables ####
    complex_vis = np.zeros([V2_mod.shape[0], V2_mod.shape[1]], dtype='complex')

    data1_visamp = np.zeros([nbl])
    data1_visamperr = np.zeros([nbl])
    data1_visphi = np.zeros([nbl])
    data1_visphierr = np.zeros([nbl])
    data1_v2 = np.zeros([nbl])
    data1_v2err = np.zeros([nbl])
    data1_t3amp = np.zeros([ncp])
    data1_t3phi = np.zeros([ncp])
    data1_t3phierr = np.zeros([ncp])
    data1_t3amperr = np.zeros([ncp])


    for ll in range(V2_mod.shape[0]):
        for mm in range(V2_mod.shape[1]):
            complex_vis[ll, mm] = to_rd(np.sqrt(V2_mod[ll, mm]), np.rad2deg(PHASE_mod[ll, mm]))

    mean_complex_vis = np.mean(complex_vis, axis = 0)
    aver_v_phasor = to_pd(np.mean(complex_vis, axis = 0))
    var_v_real = np.var(complex_vis.real, axis= 0) /V2_mod.shape[0]
    var_v_im = np.var(complex_vis.imag , axis= 0) /V2_mod.shape[0]
    std_v_real = np.std(complex_vis.real, axis= 0) /np.sqrt(V2_mod.shape[0])
    std_v_im = np.std(complex_vis.imag, axis= 0) / np.sqrt(V2_mod.shape[0])

    for mm in range(nbl):
        cov_mat = [[var_v_real[mm], std_v_real[mm] * std_v_im[mm]], [std_v_real[mm] * std_v_im[mm], var_v_im[mm]]]
        c = np.cos(np.deg2rad(aver_v_phasor[1][mm])); s = np.sin(np.deg2rad(aver_v_phasor[1][mm]))
        RR = [[c, -s], [s, c]]
        V_rt = np.linalg.multi_dot([np.transpose(RR), cov_mat, RR])
        data1_visamp[mm] = aver_v_phasor[0][mm]
        data1_visphi[mm] = aver_v_phasor[1][mm]
        data1_visphierr[mm] = np.rad2deg(np.arctan(np.sqrt(V_rt[1, 1]) / aver_v_phasor[0][mm]))
        data1_visamperr[mm] = np.sqrt(V_rt[0, 0])
        data1_v2[mm] = mean_complex_vis[mm].real ** 2 + mean_complex_vis[mm].imag ** 2 - var_v_real[mm] - var_v_im[mm]
        data1_v2err[mm] = 2 * data1_v2[mm] * np.sqrt(V_rt[0, 0])

    index_cp = np.zeros([int(ncp), 3], dtype='int')
    for k in range(ncp):
        [ind1] = np.where(
            (baselines[:, 0] == closure_phases[k, 0]) & (baselines[:, 1] == closure_phases[k, 1]))
        [ind2] = np.where(
            (baselines[:, 0] == closure_phases[k, 1]) & (baselines[:, 1] == closure_phases[k, 2]))
        [ind3] = np.where(
            (baselines[:, 0] == closure_phases[k, 0]) & (baselines[:, 1] == closure_phases[k, 2]))
        index_cp[k, 0] = int(ind1[0])  ##ind1 is a tuple
        index_cp[k, 1] = int(ind2[0])
        index_cp[k, 2] = int(ind3[0])

    t3_model = np.zeros([complex_vis.shape[0], int(ncp)], dtype=complex)
    bis_phase = np.zeros([complex_vis.shape[0], int(ncp)])
    bis_amp = np.zeros([complex_vis.shape[0], int(ncp)])

    for ll in range(t3_model.shape[0]):
        for mm in range(t3_model.shape[1]):
            t3_model[ll, mm] = complex_vis[ll, index_cp[mm, 0]] * complex_vis[ll, index_cp[mm, 1]] * np.conj(
            complex_vis[ll, index_cp[mm, 2]])
            bis_phase[ll,mm] = to_pd(t3_model[ll, mm])[1]
            bis_amp[ll, mm] = to_pd(t3_model[ll, mm])[0]

    #data1_t3phierr = np.std(bis_phase, axis=0) / np.sqrt(t3_model.shape[0])
    #data1_t3amperr = np.std(bis_amp, axis=0) / np.sqrt(t3_model.shape[0])

    aver_bis_phasor = to_pd(np.mean(t3_model, axis=0))
    var_bis_real = np.var(t3_model.real, axis=0) /V2_mod.shape[0]
    var_bis_im = np.var(t3_model.imag, axis=0) / V2_mod.shape[0]
    std_bis_real = np.std(t3_model.real, axis=0) / np.sqrt(V2_mod.shape[0])
    std_bi_im = np.std(t3_model.imag, axis=0) / np.sqrt(V2_mod.shape[0])

    for mm in range(ncp):
        cov_mat = [[var_bis_real[mm], std_bis_real[mm] * std_bi_im[mm]], [std_bis_real[mm] * std_bi_im[mm], var_bis_im[mm]]]
        c = np.cos(np.deg2rad(aver_bis_phasor[1][mm])); s = np.sin(np.deg2rad(aver_bis_phasor[1][mm]))
        RR = [[c, -s], [s, c]]
        V_rt = np.linalg.multi_dot([np.transpose(RR), cov_mat, RR])
        data1_t3amp[mm] = aver_bis_phasor[0][mm]
        data1_t3phi[mm] = aver_bis_phasor[1][mm]
        data1_t3phierr[mm] = np.rad2deg(np.arctan(np.sqrt(V_rt[1, 1]) / aver_bis_phasor[0][mm]))
        data1_t3amperr[mm] = np.sqrt(V_rt[0, 0])
    data1_visphi = np.rad2deg((np.deg2rad(data1_visphi) + np.pi) % (2 * np.pi) - np.pi)
    data1_t3phi = np.rad2deg(((np.deg2rad(data1_t3phi) + np.pi) % (2 * np.pi)) - np.pi)

    #### Compute the u, v coordinates for the closure phases:
    bl_x1_cp = bl_xp[index_cp[:, 0]]
    bl_y1_cp = bl_yp[index_cp[:, 0]]
    bl_x2_cp = bl_xp[index_cp[:, 1]]
    bl_y2_cp = bl_yp[index_cp[:, 1]]

    ###########################################
    ####### Plot the visibilities and closure phases versus spatial frequency

    sam_tools.plot_v2_cp(source, instrument, nbl, ncp, wave, bl_xp, bl_yp, bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp,\
                         data1_v2, data1_v2err, data1_t3phi, data1_t3phierr)

    ####### Save the OIFITS file ##############

    oi_file = oifits.oifits()
    if arrname == 'SIM':
        oi_arrname = instrument+'_'+arrname
        oi_target = source+'_'+arrname
        oi_ra = 0.0
        oi_dec = 0.0
        oi_pmra = 0.0
        oi_pmdec = 0.0
        oi_time = np.zeros([nbl]) + 2000.0
        tjd = Time(datetime.now().strftime('%Y-%m-%d'))
        oi_dateobs = datetime.now().strftime('%Y-%m-%d')
        oi_mjd = np.zeros([nbl]) + tjd.mjd
        oi_inttime = np.zeros([nbl]) + 10.0
        oi_visflag = np.zeros([nbl], dtype='i1')
        oi_v2flag = np.zeros([nbl], dtype='i1')
        oi_targetid = np.zeros([nbl]) + 1
        oi_t3targetid = np.zeros([ncp]) + 1
        oi_t3flag = np.zeros([ncp], dtype='i1')
        oi_t3inttime = np.zeros([ncp]) + 10.0
        oi_t3time = np.zeros([ncp]) + 2000.0
        oi_t3mjd = np.zeros([ncp]) + tjd.mjd
        catg = 'SIM_DATA'
        equinox = 2000.0
        radvel = 0.0
        parallax = 0.0

    else:
        oi_arrname = instrument + '_' + arrname
        oi_target = source + '_' + arrname
        oi_ra = data_header['HIERARCH ESO TEL TARG ALPHA']
        oi_dec = data_header['HIERARCH ESO TEL TARG DELTA']
        oi_pmra = data_header['HIERARCH ESO TEL TARG PMA']
        oi_pmdec = data_header['HIERARCH ESO TEL TARG PMD']
        oi_time = np.zeros([nbl]) + data_header['UTC']
        oi_dateobs = data_header['DATE-OBS']
        oi_datenow = datetime.now().strftime('%Y-%m-%d')
        oi_mjd = np.zeros([nbl]) + data_header['MJD-OBS']
        if instrument == 'NACO':
            oi_inttime = np.zeros([nbl]) + data_header['HIERARCH ESO DET DIT']
        if instrument == 'SPHERE':
            oi_inttime = np.zeros([nbl]) + data_header['HIERARCH ESO DET SEQ1 DIT']
        oi_visflag = np.zeros([nbl], dtype='i1')
        oi_v2flag = np.zeros([nbl], dtype='i1')
        oi_targetid = np.zeros([nbl]) + 1
        oi_t3targetid = np.zeros([ncp]) + 1
        oi_t3flag = np.zeros([ncp], dtype='i1')
        if instrument == 'NACO':
            oi_t3inttime = np.zeros([ncp]) + data_header['HIERARCH ESO DET DIT']
        if instrument == 'SPHERE':
            oi_t3inttime = np.zeros([ncp]) + data_header['HIERARCH ESO DET SEQ1 DIT']
        oi_t3time = np.zeros([ncp]) + data_header['EQUINOX']
        oi_t3mjd = np.zeros([ncp]) + data_header['MJD-OBS']
        catg = data_header['HIERARCH ESO DPR CATG']
        equinox = data_header['HIERARCH ESO TEL TARG EQUINOX']
        radvel = data_header['HIERARCH ESO TEL TARG RADVEL']
        parallax = data_header['HIERARCH ESO TEL TARG PARALLAX']

    oi_telname = np.zeros([nholes], dtype='S16')
    oi_staname = np.zeros([nholes], dtype='S16')
    oi_staindex = np.zeros([nholes], dtype='>i2')
    oi_sta_coord = np.zeros([nholes, 3], dtype='>f8')
    oi_size = np.zeros([nholes], dtype='f4')
    for i in range(nholes):
        oi_telname[i] = 'Hole' + str(i + 1)
        oi_staname[i] = 'P' + str(i + 1)
        oi_staindex[i] = i + 1
        oi_size[i] = hole_size
        oi_sta_coord[i, 0] = xcoord[i, 0]
        oi_sta_coord[i, 1] = ycoord[i, 0]
        oi_sta_coord[i, 2] = 0

    oi_file.array = oifits.OI_ARRAY(1, oi_arrname, 'GEOCENTRIC', 0.,0.,0.,oi_telname, oi_staname, oi_staindex, oi_size, oi_sta_coord)

    oi_file.target = oifits.OI_TARGET(1, np.array([1]), np.array([oi_target]), np.array([oi_ra]), np.array([oi_dec]), \
                                      np.array([equinox]), np.array([0.]), np.array([0.]), np.array([radvel]), \
                                      np.array(['UNKNOWN']), np.array(['OPTICAL']), np.array([oi_pmra]), np.array([oi_pmdec]), \
                                      np.array([0.]), np.array([0.]), np.array([parallax]), \
                                      np.array([0.]), np.array(['UNKNOWN']))
    oi_file.wavelength = oifits.OI_WAVELENGTH(1,instrument, np.array([wave]), np.array([bandwidth]))
    oi_file.vis = oifits.OI_VIS(1, oi_dateobs, oi_arrname, instrument, oi_targetid, oi_time, oi_mjd, oi_inttime, data1_visamp, \
                                data1_visamperr, data1_visphi, data1_visphierr, bl_xp, bl_yp, baselines+1, oi_visflag)

    oi_file.vis2 = oifits.OI_VIS2(1, oi_dateobs, oi_arrname, instrument, oi_targetid, oi_time, oi_mjd, oi_inttime, data1_v2,\
                                  data1_v2err, bl_xp, bl_yp, baselines+1, oi_v2flag)

    oi_file.t3 = oifits.OI_T3(1, oi_dateobs, oi_arrname, instrument, oi_t3targetid, oi_t3time, oi_t3mjd, oi_t3inttime, \
                              data1_t3amp, data1_t3amperr, data1_t3phi, data1_t3phierr, bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp,\
                              closure_phases+1, oi_t3flag)

    oi_file.write(catg+'_uncalib_'+data_file[:-5]+'.oifits')
    #MODEL_IM = sciim.zoom(MODEL_IM, (1, 1/oversample, 1/oversample), order=3)
    pyfits.writeto('MODEL_interferogram_windowed_'+data_file, MODEL_IM, overwrite=True)
    pyfits.writeto('MODEL_interferogram_'+data_file, MODEL_IM_TOT, overwrite=True)
    pyfits.writeto('MODEL_residuals_'+data_file, residuals, overwrite=True)
    pyfits.writeto('CENTERED_' + data_file, shifted_data_cube, overwrite=True)
    #windowed_data = sciim.zoom(windowed_data, (1 / oversample, 1 / oversample), order=3)
    pyfits.writeto('WINDDAT_'+data_file, windowed_data, overwrite=True)
    return
