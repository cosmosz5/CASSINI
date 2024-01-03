import numpy as np
import astropy.io.fits as pyfits
import sam_tools
import reduce_SAM_poly2_backup
import readcol as readcol
from itertools import combinations
import scipy.interpolate as interp
import pdb


def simSAM_PSF(data_filename, mask_filename, wave, bandwidth, hole_size, px_scalex, px_scaley, imsize, hole_geom, source, inst,
               arrname, rotation_angle, oversample, scale_factor = [1.0, 1.0], center_factor=[0,0], affine=False):
    [data_files] = readcol.readcol(data_filename, twod=False)
    [x_coord0, y_coord0] = readcol.readcol(mask_filename, twod=False, skipline=2)
    ###### AUTOMATIC FROM HERE ####y
    ###############################
    px_scalex = float(px_scalex)
    px_scaley = float(px_scaley)
    im = np.zeros([imsize, imsize])
    nholes = x_coord0.shape[0]

    ### Convert coordinates from mm to meters. Notice that 10 mm = 8 m

    if inst == 'JWST':
        x_coord = 1. * x_coord0
        y_coord = 1. * y_coord0
        px_scalex = px_scalex / oversample
        px_scaley = px_scaley / oversample
        imsize = int(imsize * oversample)
        im = np.zeros([imsize, imsize])
    if inst == 'NACO':
        x_coord = x_coord0 * 8. / 10  ### To convert from millimeters to meters
        y_coord = y_coord0 * 8. / 10
        x_coord = -1 * x_coord
        # hole_size = hole_size * 8. /10
    if inst == 'VISIR':
        x_coord = x_coord0 * 0.61 / 1.4  ### To convert from millimeters to meters
        y_coord = y_coord0 * 0.61 / 1.4
        y_coord = -1 * y_coord
        x_coord = -1 * x_coord
    if inst == 'SPHERE':
        x_coord = -1.0 * x_coord0 * 8.1 / 10.5 * 1.03
        y_coord = -1.0 * y_coord0 * 8.1 / 10.5 * 1.03

    ### Rotate and scale the coordinates:
    x_coord = x_coord * scale_factor[0] + center_factor[0]
    y_coord = y_coord * scale_factor[1] + center_factor[1]
    theta1 = np.deg2rad(rotation_angle)  ### Small correction for the mask position
    c, s = np.cos(theta1), np.sin(theta1)
    R_z = np.matrix([[c, -s], [s, c]])
    x_coord0, y_coord0 = np.dot(R_z, np.array([x_coord, y_coord]))
    x_coord = x_coord0.T
    y_coord = y_coord0.T

    #### Select the geometry of the pin-hole (CIRCLE vs HEXAGON):
    ### Define the Airy Ring:
    if hole_geom == 'CIRCLE':
        p0x = 1.22 * (wave / hole_size) / sam_tools.mas2rad(px_scalex)
        p0y = 1.22 * (wave / hole_size) / sam_tools.mas2rad(px_scaley)
        air_temp = sam_tools.airy([1, im.shape[0] / 2, im.shape[1] / 2, p0x, p0y], vheight=False, shape=[im.shape[0], im.shape[1]],
                             fwhm=True)
        pyfits.writeto('airy.fits', air_temp, overwrite=True)
    if hole_geom == 'HEXAGON':
        
        wav_inc = np.linspace(wave - bandwidth/2, wave + bandwidth/2, 5)
        air_temp2 = 0
        for i in range(1):
            air_temp = sam_tools.hexagon(256, hole_size, px_scalex, px_scaley, wave)
            air_temp = air_temp[int(air_temp.shape[0]/2-imsize/2):int(air_temp.shape[0]/2+imsize/2), \
                    int(air_temp.shape[1]/2-imsize/2):int(air_temp.shape[1]/2+imsize/2)]
            air_temp2 = air_temp2 + air_temp
        air = air_temp2/ np.sum(air_temp2)
        pyfits.writeto('hexa.fits', air, overwrite=True)

    if hole_geom == 'HEXAGON_VISIR':
        air = sam_tools.hexagon(imsize, hole_size, px_scalex, px_scaley, wave, visir=True)
        pyfits.writeto('hexa.fits', air, overwrite=True)

    ###### New method using the autocorrelation function of the mask's holes:
    # 1) Define the baselines
    baselines = np.array(list(combinations(range(int(x_coord.shape[0])), 2)))
    closure_phases = np.array(list(combinations(range(int(x_coord.shape[0])), 3)))
    nbl = int((x_coord.shape[0]) * (x_coord.shape[0] - 1) / 2.0)
    ncp = int((x_coord.shape[0]) * (x_coord.shape[0] - 1) * (x_coord.shape[0] - 2) / 6.0)
    bl_x = np.zeros([nbl])
    bl_y = np.zeros([nbl])
    cube_bl_over = np.zeros([nbl, int(imsize*3), int(imsize*3)], dtype='float64')
    cube_bl_sin_over = np.zeros([nbl, int(imsize*3), int(imsize*3)], dtype='float64')
    cube_bl = np.zeros([nbl, imsize, imsize], dtype='float64')
    cube_bl_sin = np.zeros([nbl, imsize, imsize], dtype='float64')
    PSF_im = np.zeros([imsize, imsize])

    sz = [imsize*3, imsize*3]

    if int(imsize *3) % 2 == 0:
        y, x = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2 - 1):sz[1] * 1j,
               -np.floor(sz[0] / 2):np.floor(sz[0] / 2 - 1):sz[0] * 1j]
    else:
        y, x = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2):sz[1] * 1j,
               -np.floor(sz[0] / 2):np.floor(sz[0] / 2):sz[0] * 1j]

    indx = np.where(x == 0)
    indy = np.where(y == 0)
    y[indy] = 1e-12
    x[indx] = 1e-12
    x = x * sam_tools.mas2rad(px_scalex)
    y = y * sam_tools.mas2rad(px_scaley)

    #The exagon is also affected by wavelength smearing

    for i in range(nbl):  # Index for the number of baselines
        bl_x[i] = x_coord[baselines[i, 1]] - x_coord[baselines[i, 0]]
        bl_y[i] = y_coord[baselines[i, 1]] - y_coord[baselines[i, 0]]
        for k in range(cube_bl_over.shape[1]):  # Index for the x coordinate
            for l in range(cube_bl_over.shape[2]):  # Index for the y coordinate
                denom = (bandwidth / wave * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave)
                if denom == 0.0:
                    denom = 1e-12
                cube_bl_over[i, k, l] = np.cos(2. * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                   * np.sin(bandwidth / wave * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                   / denom
                cube_bl_sin_over[i, k, l] = -np.sin(2. * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                       * np.sin(
                    bandwidth / wave * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                       / denom

    sz = np.shape(PSF_im)

    if int(imsize) % 2 == 0:
        yy, xx = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2 - 1):sz[1] * 1j,
               -np.floor(sz[0] / 2):np.floor(sz[0] / 2 - 1):sz[0] * 1j]
    else:
        yy, xx = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2):sz[1] * 1j,
               -np.floor(sz[0] / 2):np.floor(sz[0] / 2):sz[0] * 1j]

    indx = np.where(xx == 0)
    indy = np.where(yy == 0)
    yy[indy] = 1e-12
    xx[indx] = 1e-12
    xx = xx * sam_tools.mas2rad(px_scalex)
    yy = yy * sam_tools.mas2rad(px_scaley)

    for i in range(nbl):
        temp1 = np.squeeze(np.reshape(cube_bl_over[i,:,:], -1))
        temp2 = np.squeeze(np.reshape(cube_bl_sin_over[i,:,:], -1))

        grid1 = interp.griddata((y.reshape(-1), x.reshape(-1)), \
                                temp1, (yy.reshape(-1), xx.reshape(-1)), method='linear')
        grid2= interp.griddata((y.reshape(-1), x.reshape(-1)), \
                                    temp2, (yy.reshape(-1), xx.reshape(-1)), method='linear')

        cube_bl[i,:,:] = np.reshape(grid1, (imsize, imsize))
        cube_bl_sin[i,:,:] = np.reshape(grid2, (imsize, imsize))

    PSF_im = air * nholes + air * np.sum(2 * cube_bl, axis=0)
    MTF_im = np.fft.fftshift(np.fft.ifft2(PSF_im / np.max(PSF_im)))

    pyfits.writeto('PSF.fits', PSF_im, overwrite=True)
    pyfits.writeto('cube_bl.fits', cube_bl, overwrite=True)
    pyfits.writeto('MTF.fits', np.abs(MTF_im), overwrite=True)

    ### To window the PSF:
    if (inst == 'VISIR') | (inst == 'JWST'):
        hole_size = hole_size / np.cos(np.deg2rad(30.))
        dd = 1.0 * wave / (hole_size * sam_tools.mas2rad(px_scalex))
    else:
        dd = 1.0 * wave / (hole_size * sam_tools.mas2rad(px_scalex))
    window = np.fft.fftshift(np.exp(- (sam_tools.dist(imsize) / dd) ** 4))
    ind_window = np.where(window > 0.20)
    uwindow = np.zeros([imsize, imsize])
    uwindow[ind_window] = 1.0


    # To get the observables:
    for i in range(len(data_files)):

        temp_data = pyfits.getdata(data_files[i])
        ind_nan = np.where(np.isnan(temp_data[0,:,:]) == True)
        temp_data[0,ind_nan[0], ind_nan[1]] = 0
        uwindow_temp = uwindow * temp_data[0, :, :]
        ind_window2 = np.where((uwindow_temp >= 0.005 * np.max(uwindow_temp)) | (uwindow_temp < 80e3))
        ind_window3 = np.where((uwindow_temp < 0.005 * np.max(uwindow_temp)) | (uwindow_temp > 80e3))
        uwindow[ind_window2] = 1.0
        uwindow[ind_window3] = 0.0

        # Uniform window over the area to perform the PSF Fitting
        pyfits.writeto('uwindow.fits', uwindow, overwrite=True)

        if arrname == 'DATA':
            hdul= pyfits.open(data_files[i])
            head = hdul[0].header
            source = head['TARGPROP']
            hdul.close()
            
        print(data_files[i])

        if affine == True:
            data_cube = pyfits.open(data_files[i])
            data_header = data_cube[0].header
            data_cube = data_cube[0].data
            median_data = np.median(data_cube, axis=0).reshape(1,imsize, imsize)

            windowed_data = median_data * uwindow
            ind = np.where(uwindow != 0.0)
            BB = windowed_data[0,ind[0], ind[1]]
            #### Define the matrix for the P2VM:
            P2VM = np.zeros([BB.shape[0], 2 * nbl + 2])

            P2VM[:, 0] = air[ind] * nholes
            P2VM[:, -1] = 1.0

            f = 0;
            m = 0
            for k in range(2 * nbl):
                if (k % 2 == 0):
                    P2VM[:, k + 1] = air[ind] * 2. * cube_bl[f, ind[0], ind[1]]
                    f = f + 1
                elif (k % 2 != 0):
                    P2VM[:, k + 1] = air[ind] * 2. * cube_bl_sin[m, ind[0], ind[1]]
                    m = m + 1

            W = np.diag((np.abs(BB) * 0 + 1.0))
            # W = np.diag(np.log(np.abs(BB)))
            Aw = np.dot(W, P2VM)
            Bw = np.dot(BB, W)
            MOD, res, rnk, s = np.linalg.lstsq(Aw, Bw, rcond=-1)

            # P2VMt = np.linalg.pinv(P2VM)
            # MOD = np.dot(P2VMt,BB)
            MOD_fringes = np.dot(P2VM, MOD)
            MODEL_IM = np.zeros([1, imsize, imsize])
            MODEL_IM[i,ind[0], ind[1]] = MOD_fringes
            return MOD_fringes -  windowed_data[0,ind[0], ind[1]]
        else:
            reduce_SAM_poly2_backup.redSAM(data_files[i], PSF_im, uwindow, bl_x, bl_y, wave, hole_size, imsize, px_scalex,
                                       nholes,
                                       nbl, ncp, \
                                       baselines, closure_phases, xx, yy, air, cube_bl, cube_bl_sin, x_coord, y_coord,
                                       source,
                                       bandwidth, inst, oversample, arrname=arrname)

    return 0


