def mas2rad(x):
    import numpy as np
    y=x/3600/1000*np.pi/180
    y=np.array(y)
    return y

def airy(inpars, circle=True, rotate=False, vheight=True, shape=None, fwhm=False):
    import numpy
    import scipy
    import pdb
    """Returns a 2d Airy *function* of the form:
        x' = numpy.cos(rota) * x - numpy.sin(rota) * y
        y' = numpy.sin(rota) * x + numpy.cos(rota) * y
        (rota should be in degrees)
        radius = sqrt( (x'-xcen)^2 + (y'-ycen)^2 )
        g = b + a * 2.0*BesselJ1( radius ) / radius
            (with a correction for the divide-by-zero in the center)

        inpars = [b,a,center_x,center_y,width_x,width_y,rota]
                 (b is background height, a is peak amplitude)

        where x and y are the input parameters of the returned function,
        and all other parameters are specified by this function

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

        You can choose to ignore / neglect some of the above input parameters
            unumpy.sing the following options:
            circle=1 - default is a circular Airy Disk.  An elliptical Airy is
                possible, but probably not physically motivated (unless it's
                sampled onto a stretched grid?).
            rotate=0 - default allows rotation of the gaussian ellipse.  Can
                remove last parameter by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an
                additive constant for the Airy function.  Can remove first
                parameter by setting this to 0
            shape=None - if shape is set (to a 2-parameter list) then returns
                an image with the gaussian defined by inpars
        fwhm - if set, assumes the Width parameters input are FWHM widths, so
            they'll be converted to "Sigma" widths by s = FWHM/2.0/1.61633
            (http://en.wikipedia.org/wiki/Airy_disk
            and http://home.fnal.gov/~neilsen/notebook/astroPSF/astroPSF.html)
        """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight == 1:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_y, center_x = inpars.pop(0), inpars.pop(0), inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle == 1:
        width = inpars.pop(0)
        width_x = float(width)
        width_y = float(width)
        rotate = 0
    else:
        width_x, width_y = inpars.pop(0), inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate == 1:
        rota = inpars.pop(0)
        rota = numpy.pi / 180. * float(rota)
        rcen_x = center_x * numpy.cos(rota) - center_y * numpy.sin(rota)
        rcen_y = center_x * numpy.sin(rota) + center_y * numpy.cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError("There are still input parameters:" + str(inpars) + \
                         " and you've input: " + str(inpars_old) + \
                         " circle=%d, rotate=%d, vheight=%d" % (circle, rotate, vheight))

    if fwhm:
        width_x /= 2.0 * 1.61633
        width_y /= 2.0 * 1.61633

    def rotairy(x, y):
        if rotate == 1:
            xp = x * numpy.cos(rota) - y * numpy.sin(rota)
            yp = x * numpy.sin(rota) + y * numpy.cos(rota)
        else:
            xp = x
            yp = y
        rr = numpy.sqrt(((rcen_x - xp) / width_x) ** 2 +
                        ((rcen_y - yp) / width_y) ** 2)
        # airy first zero: linspace(3.5,4,10000)[argmin((scipy.special.j1(linspace(3.5,4,10000))/linspace(3.5,4,10000))**2)]
        # or scipy.special.jn_zeros(1,1)
        # http://en.wikipedia.org/wiki/Airy_disk
        airy_func = (2.0 * scipy.special.j1(rr) / rr) ** 2
        airy_func[rr == 0] = 1.0
        airy = height + amplitude * airy_func

        return airy

    if shape is not None:
        return rotairy(*numpy.indices(shape))
    else:
        return rotairy

def hexagon_generator(edge_length, offset):
  """Generator for coordinates in a hexagon."""
  x, y = offset
  for angle in range(0, 360, 60):
      x += np.cos(np.radians(angle)) * edge_length
      y += np.sin(np.radians(angle)) * edge_length
      yield x, y

def hexagon(imsize, hole_size, px_scalex, px_scaley, wave, visir=False):
    import numpy as np
    import pdb
    import astropy.io.fits as pyfits
    from PIL import Image, ImageDraw

    hexa_im = np.zeros([imsize, imsize], dtype=complex)
    hexa_im1 = np.zeros([imsize, imsize], dtype=complex)
    hexa_im2 = np.zeros([imsize, imsize], dtype=complex)

    if imsize % 2 == 0:
        y, x = np.mgrid[-np.floor(imsize / 2):np.floor(imsize / 2 - 1):imsize * 1j,
               -np.floor(imsize / 2):np.floor(imsize / 2 - 1):imsize * 1j]

        y_hex, x_hex = np.mgrid[-np.floor(imsize / 2):np.floor(imsize / 2 - 1):imsize * 1j,
                       -np.floor(imsize / 2):np.floor(imsize / 2 - 1):imsize * 1j]
    else:

        y, x = np.mgrid[-np.floor(imsize / 2):np.floor(imsize / 2):imsize * 1j,
               -np.floor(imsize / 2):np.floor(imsize / 2):imsize * 1j]

        y_hex, x_hex = np.mgrid[-np.floor(imsize / 2):np.floor(imsize / 2):imsize * 1j,
                       -np.floor(imsize / 2):np.floor(imsize / 2):imsize * 1j]


    x = x * mas2rad(px_scalex)
    y = y * mas2rad(px_scaley)

    #   hole_size = hole_size /wave
    #D = (hole_size * np.cos(np.deg2rad(30.0)))
    D = hole_size * 1.027

    ## To create the hexagon shape of the pin-hole
    img = Image.new('L', (imsize, imsize), 0)
    offx = imsize/2 - 64 *np.sin(np.deg2rad(30.))
    offy = imsize/2 - 64 * np.cos(np.deg2rad(30.))
    hexagon = hexagon_generator(64, offset=(offx, offy))
    ImageDraw.Draw(img).polygon(list(hexagon), outline=0, fill=1)
    mask = np.array(img)



    ind_hex = np.where(mask != 0)
    x_hex = D * (x_hex / (128 * np.cos(np.deg2rad(30.))))
    y_hex = D * (y_hex / (128 * np.cos(np.deg2rad(30.))))

    pyfits.writeto('hexagon.fits', mask, overwrite=True)

    if visir == True:

        for i in range(hexa_im.shape[0]):
            for j in range(hexa_im.shape[1]):
                hexa_im[i ,j] = np.sum(np.exp(1j * 2. * np.pi * (y[i, j] * x_hex[ind_hex] + x[i, j] * y_hex[ind_hex]) / wave))

        hexa_FT = (np.abs(hexa_im)) ** 2 / np.max((np.abs(hexa_im)) ** 2)

    else:

        for i in range(hexa_im.shape[0]):
            for j in range(hexa_im.shape[1]):
                hexa_im[i ,j] = np.sum(np.exp(1j * 2. * np.pi * (y[i, j] * y_hex[ind_hex] + x[i, j] * x_hex[ind_hex]) / wave))

        hexa_FT = (np.abs(hexa_im)) ** 2 / np.max((np.abs(hexa_im)) ** 2)

    #
    #
    # for i in range(hexa_im.shape[0]):
    #     for j in range(hexa_im.shape[1]):
    #
    #         print i, j
    #         if ((x[i,j] == 0) & (y[i,j] == 0)) | (((x[i,j] > 0.8 * np.sqrt(3)*y[i,j]) & (x[i,j] < 1.2 * np.sqrt(3)*y[i,j])) & (y[i,j] == 0)) | (((x[i,j] < -0.8 * np.sqrt(3)*y[i,j]) & (x[i,j] > -1.2 * np.sqrt(3)*y[i,j])) & (y[i,j] == 0)):
    #             hexa_im1[i,j] = np.sqrt(3) * hole_size**2 / 4
    #             hexa_im2[i, j] = np.sqrt(3) * hole_size ** 2 / 4
    #         elif ((x[i,j] == 0) & (y[i,j] != 0)) | (((x[i,j] > 0.99 * np.sqrt(3)*y[i,j]) & (x[i,j] < 1.01 * np.sqrt(3)*y[i,j])) & (y[i,j] != 0)) | (((x[i,j] < -0.99 * np.sqrt(3)*y[i,j]) & (x[i,j] > -1.01 * np.sqrt(3)*y[i,j])) & (y[i,j] != 0)):
    #             hexa_im1[i,j] = np.exp(-1j * hole_size *np.pi * y[i,j]) / (2.*np.sqrt(3)*np.pi**2*y[i,j]**2) \
    #                            * (-1. + 1j * hole_size *np.pi *y[i,j] + np.exp(1j*hole_size*np.pi*y[i,j]) - 2.*1j*y[i,j]*np.exp(1j*hole_size*np.pi*y[i,j]))
    #             hexa_im2[i, j] = np.exp(-1j * hole_size * np.pi * (-y[i, j])) / (
    #                     2. * np.sqrt(3) * np.pi ** 2 * (-y[i, j]) ** 2) \
    #                              * (-1. + 1j * hole_size * np.pi * (-y[i, j]) + np.exp(
    #                 1j * hole_size * np.pi * (-y[i, j])) - 2. * 1j * (-y[i, j]) * np.exp(1j * hole_size * np.pi * (-y[i, j])))
    #
    #         else:
    #             hexa_im1[i,j] = (np.exp(-1j*np.pi*hole_size*(2.*x[i,j]/np.sqrt(3)+y[i,j])) / (4* np.pi**2 *(x[i,j]**3 - 3 * x[i,j] * y[i,j]**2))) \
    #                 *((np.sqrt(3)*x[i,j] - np.sqrt(3)*y[i,j]) * (np.exp(1j * np.pi * hole_size * np.sqrt(3) * x[i,j]) \
    #                                                             -(np.exp(1j * np.pi * hole_size * (4/np.sqrt(3)*x[i,j] + y[i,j])))) \
    #                                                             + (np.sqrt(3) *x[i,j] + 3*y[i,j]) * (np.exp(1j*np.pi*hole_size/np.sqrt(3) \
    #                                                                                                     - np.exp(1j*np.pi *hole_size *y[i,j]))))
    #
    #             hexa_im2[i, j] = (np.exp(-1j * np.pi * hole_size * (2. * x[i, j] / np.sqrt(3) + (-y[i, j]))) / (
    #                     4 * np.pi ** 2 * (x[i, j] ** 3 - 3 * x[i, j] * (-y[i, j]) ** 2))) \
    #                             * ((np.sqrt(3) * x[i, j] - np.sqrt(3) * (-y[i, j])) * (
    #                     np.exp(1j * np.pi * hole_size * np.sqrt(3) * x[i, j]) \
    #                     - (np.exp(1j * np.pi * hole_size * (4 / np.sqrt(3) * x[i, j] + (-y[i, j]))))) \
    #                                + (np.sqrt(3) * x[i, j] + 3 * (-y[i, j])) * (np.exp(1j * np.pi * hole_size / np.sqrt(3) \
    #                                                                                 - np.exp(
    #                         1j * np.pi * hole_size * (-y[i, j])))))
    #
    #         hexa_im[j,i] = hexa_im1[i,j] + hexa_im2[i,j]


    return hexa_FT

def dist(value):
    import numpy as np
    import pylab as pl

    val = value * 1.0
    if value % 2 != 0:
        # print 'Entre al 1'
        val_f = np.floor(value / 2) + 1
        val_i = np.floor(value / 2)
    if value % 2 == 0:
        # print 'Entre al 2'
        val_f = np.floor(value / 2) + 1
        val_i = np.floor(value / 2) - 1

    ind1 = np.arange(0, val_f)
    ind2 = np.arange(val_i, 0, -1)
    ind = np.concatenate([ind1, ind2])
    ind = ind ** 2
    a = np.zeros([value, value])

    for i in range(int(val_f)):
        # print i
        y = np.sqrt(ind + i**2)
        # print y
        a[i, :] = y
        if i != 0:
            a[int(val) - i, :] = y
    # ind1=np.arange(0,value/2+1)
    # ind2=np.arange(value/2,0,-1)
    # ind=np.concatenate([ind1, ind2])
    # X, Y=np.ogrid[ind, ind]
    # d=np.hypot(X,Y)
    return a

import numpy as np
to_rd = lambda m, d: m * np.exp(1j * np.deg2rad(d))
to_pd = lambda x: (abs(x), np.rad2deg(np.angle(x)))

def compute_closure_phases(nbl, ncp, sta_index_vis, sta_index_cp, bl_x, bl_y, vis_mod, phi_mod):
    import numpy as np
    import pdb

    vis = np.zeros([vis_mod.shape[0], vis_mod.shape[1]], dtype=complex)
    for i in range(vis.shape[0]):
        vis[i,:] = to_rd(vis_mod[i,:], phi_mod[i,:])

    n_pointings = vis_mod.shape[1] / nbl
    index_cp = np.zeros([ncp, 3], dtype='int')
    for j in range(int(n_pointings)):
        index = range(nbl * j, nbl + nbl * j)
        for k in range(ncp * j, ncp + ncp * j):
            [ind1] = np.where(
                (sta_index_vis[index, 0] == sta_index_cp[k, 0]) & (sta_index_vis[index, 1] == sta_index_cp[k, 1]))
            [ind2] = np.where(
                (sta_index_vis[index, 0] == sta_index_cp[k, 1]) & (sta_index_vis[index, 1] == sta_index_cp[k, 2]))
            [ind3] = np.where(
                (sta_index_vis[index, 0] == sta_index_cp[k, 0]) & (sta_index_vis[index, 1] == sta_index_cp[k, 2]))
            index_cp[k, 0] = index[int(ind1[0])]  ##ind1 is a tuple
            index_cp[k, 1] = index[int(ind2[0])]
            index_cp[k, 2] = index[int(ind3[0])]

    t3_model = np.zeros([vis_mod.shape[0], ncp], dtype=complex)
    if len(phi_mod.shape) > 1:
        t3_model = vis[:, index_cp[:, 0]] * vis[:,index_cp[:, 1]] * np.conj(vis[:,index_cp[:, 2]])
    else:
        t3_model = vis[index_cp[:, 0]] * vis[index_cp[:, 1]]* np.conj(vis[index_cp[:, 2]])
    t3phi_modelA = np.absolute(t3_model)
    t3phi_modelf = np.angle(t3_model, deg=True)
    #ind_wrap1 = np.where(t3phi_modelf > 180.0)
    #t3phi_modelf[ind_wrap1] = t3phi_modelf[ind_wrap1] - 360.0
    #ind_wrap2 = np.where(t3phi_modelf < -180.0)
    #t3phi_modelf[ind_wrap2] = t3phi_modelf[ind_wrap2] + 360.0

    #### Compute the u, v coordinates for the closure phases:
    bl_x1_cp = np.zeros([ncp])
    bl_x2_cp = np.zeros([ncp])
    bl_y1_cp = np.zeros([ncp])
    bl_y2_cp = np.zeros([ncp])

    bl_x1_cp = bl_x[index_cp[:,0]]
    bl_y1_cp = bl_y[index_cp[:,0]]
    bl_x2_cp = bl_x[index_cp[:,1]]
    bl_y2_cp = bl_y[index_cp[:,1]]

    return bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp, t3phi_modelA, t3phi_modelf

def plot_v2_cp(source, instrument, nbl, ncp, wave, bl_x, bl_y, bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp, V2_data,  \
               V2_data_err, CP_data, CP_data_err):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

    uv_cp = np.zeros([bl_x1_cp.shape[0]])
    u_cp = np.zeros([bl_x1_cp.shape[0]])
    v_cp = np.zeros([bl_x1_cp.shape[0]])
    bl_x3_cp = -1.0 * (bl_x1_cp + bl_x2_cp)
    bl_y3_cp = -1.0 * (bl_y1_cp + bl_y2_cp)

    for j in range(bl_x1_cp.shape[0]):
        uv1 = np.sqrt((bl_x1_cp[j]) ** 2 + (bl_y1_cp[j]) ** 2)
        uv2 = np.sqrt((bl_x2_cp[j]) ** 2 + (bl_y2_cp[j]) ** 2)
        uv3 = np.sqrt((bl_x3_cp[j]) ** 2 + (bl_y3_cp[j]) ** 2)
        if uv1 >= uv2 and uv1 >= uv3:
            uv_cp[j] = uv1
            u_cp[j] = bl_x1_cp[j]
            v_cp[j] = bl_y1_cp[j]
        elif uv2 >= uv1 and uv2 >= uv3:
            uv_cp[j] = uv2
            u_cp[j] = bl_x2_cp[j]
            v_cp[j] = bl_y2_cp[j]
        elif uv3 >= uv1 and uv3 >= uv2:
            uv_cp[j] = uv3
            u_cp[j] = bl_x3_cp[j]
            v_cp[j] = bl_y3_cp[j]

    for i in range(nbl):
        uv = np.sqrt(bl_x[i]**2 + bl_y[i]**2) / wave
        if i == 1:
            ax1.errorbar(uv, V2_data[i], yerr=V2_data_err[i], fmt='o', color='blue', label=str(wave/1e-6)+' $\mu$m')
        else:
            ax1.errorbar(uv, V2_data[i], yerr=V2_data_err[i], fmt='o', color='blue')

    for i in range(ncp):
        ax2.errorbar(uv_cp[i] / wave, CP_data[i], yerr=CP_data_err[i], fmt='o', color='blue')

    ax1.legend()
    ax1.set_title(source+'_'+instrument+'_'+'V$^2$')
    ax2.set_title(source+'_'+instrument+'_'+'CPs')
    ax1.set_xlabel('Spatial Frequency [1/rad]')
    ax2.set_xlabel('Spatial Frequency [1/rad]')
    ax1.set_ylabel('Uncalibrated V$^2$')
    ax2.set_ylabel('Uncalibrated CPs [deg]')
    ax1.set_ylim([0,1.2])


    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.savefig(source+'_uncalibrated_observables'+'.pdf', dpi=200, bbox_inches='tight')

    plt.show()

    return

def plot_v2_cp_calibrated(source, instrument, nbl, ncp, wave, bl_x, bl_y, bl_x1_cp, bl_y1_cp, bl_x2_cp, bl_y2_cp, V2_data,  \
               V2_data_err, CP_data, CP_data_err):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))

    uv_cp = np.zeros([bl_x1_cp.shape[0]])
    u_cp = np.zeros([bl_x1_cp.shape[0]])
    v_cp = np.zeros([bl_x1_cp.shape[0]])
    bl_x3_cp = -1.0 * (bl_x1_cp + bl_x2_cp)
    bl_y3_cp = -1.0 * (bl_y1_cp + bl_y2_cp)

    for j in range(bl_x1_cp.shape[0]):
        uv1 = np.sqrt((bl_x1_cp[j]) ** 2 + (bl_y1_cp[j]) ** 2)
        uv2 = np.sqrt((bl_x2_cp[j]) ** 2 + (bl_y2_cp[j]) ** 2)
        uv3 = np.sqrt((bl_x3_cp[j]) ** 2 + (bl_y3_cp[j]) ** 2)
        if uv1 >= uv2 and uv1 >= uv3:
            uv_cp[j] = uv1
            u_cp[j] = bl_x1_cp[j]
            v_cp[j] = bl_y1_cp[j]
        elif uv2 >= uv1 and uv2 >= uv3:
            uv_cp[j] = uv2
            u_cp[j] = bl_x2_cp[j]
            v_cp[j] = bl_y2_cp[j]
        elif uv3 >= uv1 and uv3 >= uv2:
            uv_cp[j] = uv3
            u_cp[j] = bl_x3_cp[j]
            v_cp[j] = bl_y3_cp[j]

    for i in range(nbl):
        uv = np.sqrt(bl_x[i]**2 + bl_y[i]**2) / wave
        if i == 1:
            ax1.errorbar(uv, V2_data[i], yerr=V2_data_err[i], fmt='o', color='blue', label=str(wave/1e-6)+' $\mu$m')
        else:
            ax1.errorbar(uv, V2_data[i], yerr=V2_data_err[i], fmt='o', color='blue')

    for i in range(ncp):
        ax2.errorbar(uv_cp[i] / wave, CP_data[i], yerr=CP_data_err[i], fmt='o', color='blue')

    ax1.legend()
    ax1.set_title(source+'_'+instrument+'_'+'V$^2$')
    ax2.set_title(source+'_'+instrument+'_'+'CPs')
    ax1.set_xlabel('Spatial Frequency [1/rad]')
    ax2.set_xlabel('Spatial Frequency [1/rad]')
    ax1.set_ylabel('Uncalibrated V$^2$')
    ax2.set_ylabel('Uncalibrated CPs [deg]')
    ax1.set_ylim([0,1.2])


    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.savefig(source+'_CALIBRATED_observables'+'.pdf', dpi=200, bbox_inches='tight')

    plt.show()

    return
