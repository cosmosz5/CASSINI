import numpy as np
import astropy.io.fits as pyfits
import sam_tools
import reduce_SAM_poly2
import readcol as readcol
from itertools import combinations
import pdb


def simSAM_PSF(data_filename, mask_filename, wave, bandwidth, hole_size, px_scale, imsize, hole_geom, source, inst,
               arrname, rotation_angle, oversample):
    [data_files] = readcol.readcol(data_filename, twod=False)
    [x_coord0, y_coord0] = readcol.readcol(mask_filename, twod=False, skipline=2)
    ###### AUTOMATIC FROM HERE ####
    ###############################
    px_scale = float(px_scale)
    im = np.zeros([imsize, imsize])
    nholes = x_coord0.shape[0]

    ### Convert coordinates from mm to meters. Notice that 10 mm = 8 m

    if inst == 'JWST':
        x_coord = -1. * x_coord0
        y_coord = 1. * y_coord0
        px_scale = px_scale / oversample
        imsize = int(imsize * oversample)
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

    ### Rotate coordinates:
    theta1 = np.deg2rad(rotation_angle)  ### Small correction for the mask position
    c, s = np.cos(theta1), np.sin(theta1)
    R_z = np.matrix([[c, -s], [s, c]])
    x_coord0, y_coord0 = np.dot(R_z, np.array([x_coord, y_coord]))
    x_coord = x_coord0.T
    y_coord = y_coord0.T

    #### Select the geometry of the pin-hole (CIRCLE vs HEXAGON):
    ### Define the Airy Ring:
    if hole_geom == 'CIRCLE':
        p0 = 1.22 * (wave / hole_size) / sam_tools.mas2rad(px_scale)
        air = sam_tools.airy([1, im.shape[0] / 2, im.shape[1] / 2, p0], vheight=False, shape=[im.shape[0], im.shape[1]],
                             fwhm=True)
        pyfits.writeto('airy.fits', air, overwrite=True)
    if hole_geom == 'HEXAGON':
        air = sam_tools.hexagon(imsize, hole_size, px_scale, wave)
        pyfits.writeto('hexa.fits', air, overwrite=True)

    if hole_geom == 'HEXAGON_VISIR':
        air = sam_tools.hexagon(imsize, hole_size, px_scale, wave, visir=True)
        pyfits.writeto('hexa.fits', air, overwrite=True)

    ###### New method using the autocorrelation function of the mask's holes:
    # 1) Define the baselines
    baselines = np.array(list(combinations(range(int(x_coord.shape[0])), 2)))
    closure_phases = np.array(list(combinations(range(int(x_coord.shape[0])), 3)))
    nbl = int((x_coord.shape[0]) * (x_coord.shape[0] - 1) / 2.0)
    ncp = int((x_coord.shape[0]) * (x_coord.shape[0] - 1) * (x_coord.shape[0] - 2) / 6.0)
    bl_x = np.zeros([nbl])
    bl_y = np.zeros([nbl])
    cube_bl = np.zeros([nbl, imsize, imsize], dtype='float64')
    cube_bl_sin = np.zeros([nbl, imsize, imsize], dtype='float64')
    PSF_im = np.zeros([imsize, imsize])

    sz = np.shape(PSF_im)

    if imsize % 2 == 0:
        y, x = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2 - 1):sz[1] * 1j,
               -np.floor(sz[0] / 2):np.floor(sz[0] / 2 - 1):sz[0] * 1j]
    else:
        y, x = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2):sz[1] * 1j,
               -np.floor(sz[0] / 2):np.floor(sz[0] / 2):sz[0] * 1j]

    indx = np.where(x == 0)
    indy = np.where(y == 0)
    y[indy] = 1e-12
    x[indx] = 1e-12
    x = x * sam_tools.mas2rad(px_scale)
    y = y * sam_tools.mas2rad(px_scale)

    for i in range(nbl):  # Index for the number of baselines
        bl_x[i] = x_coord[baselines[i, 1]] - x_coord[baselines[i, 0]]
        bl_y[i] = y_coord[baselines[i, 1]] - y_coord[baselines[i, 0]]

        for k in range(cube_bl.shape[1]):  # Index for the x coordinate
            for l in range(cube_bl.shape[2]):  # Index for the y coordinate
                denom = (bandwidth / wave * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave)
                if denom == 0.0:
                    denom = 1e-12
                cube_bl[i, k, l] = np.cos(2. * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                   * np.sin(bandwidth / wave * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                   / denom
                cube_bl_sin[i, k, l] = -np.sin(2. * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                       * np.sin(
                    bandwidth / wave * np.pi * (x[k, l] * bl_x[i] + y[k, l] * bl_y[i]) / wave) \
                                       / denom

    PSF_im = air * nholes + air * np.sum(2 * cube_bl, axis=0)
    MTF_im = np.fft.fftshift(np.fft.ifft2(PSF_im / np.max(PSF_im)))

    pyfits.writeto('PSF.fits', PSF_im, overwrite=True)
    pyfits.writeto('cube_bl.fits', cube_bl, overwrite=True)
    pyfits.writeto('MTF.fits', np.abs(MTF_im), overwrite=True)
    ### To window the PSF:
    if (inst == 'VISIR') | (inst == 'JWST'):
        hole_size = hole_size / np.cos(np.deg2rad(30.))
        dd = 1.0 * wave / (hole_size * sam_tools.mas2rad(px_scale))
    else:
        dd = 1.0 * wave / (hole_size * sam_tools.mas2rad(px_scale))
    window = np.fft.fftshift(np.exp(- (sam_tools.dist(imsize) / dd) ** 4))
    ind_window = np.where(window > 0.20)
    uwindow = np.zeros([imsize, imsize])
    uwindow[ind_window] = 1.0
    uwindow = uwindow * PSF_im
    ind_window2 = np.where(uwindow >= 0.01 * np.max(uwindow))
    ind_window3 = np.where(uwindow < 0.01 * np.max(uwindow))
    uwindow[ind_window2] = 1.0
    uwindow[ind_window3] = 0.0

    # Uniform window over the area to perform the PSF Fitting
    pyfits.writeto('uwindow.fits', uwindow, overwrite=True)

    # To get the observables:
    for i in range(len(data_files)):
        print(data_files[i])
        reduce_SAM_poly2.redSAM(data_files[i], PSF_im, uwindow, bl_x, bl_y, wave, hole_size, imsize, px_scale,
                                       nholes,
                                       nbl, ncp, \
                                       baselines, closure_phases, x, y, air, cube_bl, cube_bl_sin, x_coord, y_coord,
                                       source,
                                       bandwidth, inst, oversample, arrname=arrname)

    return 0



