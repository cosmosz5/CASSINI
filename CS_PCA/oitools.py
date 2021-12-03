import numpy as np
from pynufft import NUFFT

to_rd = lambda m, d: m * np.exp(1j * np.deg2rad(d))
to_pd = lambda x: (abs(x), np.rad2deg(np.angle(x)))


def mas2rad(x):
    import numpy as np
    y=x/3600.0/1000.0*np.pi/180.0
    y=np.array(y)
    return y

def compute_vis_matrix(im, u, v, scale):
    import numpy as np
    import math
    import pdb
    sz = np.shape(im)
    if sz[0] % 2 == 0:
        x, y = np.mgrid[-np.floor(sz[1] / 2 - 1):np.floor(sz[1] / 2):sz[1] * 1j,
               np.floor(sz[0] / 2 - 1):-np.floor(sz[0] / 2):sz[0] * 1j]
    else:
        x, y = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2):sz[1] * 1j,
               np.floor(sz[0] / 2):-np.floor(sz[0] / 2):sz[0] * 1j]

    x = x * scale
    y = y * scale
    xx = x.reshape(-1)
    yy = y.reshape(-1)
    im = im.reshape(-1)
    arg = -2.0 * np.pi * (u * yy + v * xx)
    reales = im.dot(np.cos(arg))
    imaginarios = im.dot(np.sin(arg))
    visib = np.absolute(reales + 1j*imaginarios)
    phase = np.angle(reales + 1j*imaginarios, deg=True)

    #visib = np.linalg.norm([reales, imaginarios])
    #phase_temp = np.arctan2(imaginarios, reales)
    #phase = np.rad2deg((phase_temp + np.pi) % (2 * np.pi) - np.pi)
    return visib, phase

def compute_closure_phases(nbl, ncp, sta_index_vis, sta_index_cp,  vis_mod, phi_mod):
    import numpy as np
    import pdb
    vis = np.zeros([vis_mod.shape[0], vis_mod.shape[1]], dtype=complex)
    for i in range(vis.shape[0]):
        vis[i, :] = to_rd(vis_mod[i, :], phi_mod[i, :])

    n_pointings = vis_mod.shape[0] / nbl
    index_cp = np.zeros([int(ncp * n_pointings), 3], dtype='int')
    
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

    t3_model = np.zeros([int(ncp * n_pointings), vis_mod.shape[1]], dtype=complex)
    t3phi_modelA = np.zeros([int(ncp * n_pointings), vis_mod.shape[1]])
    t3phi_modelf = np.zeros([int(ncp * n_pointings), vis_mod.shape[1]])

    for r in range(t3_model.shape[1]):
        t3_model[:,r] = vis[index_cp[:, 0], r] * vis[index_cp[:, 1], r] * np.conj(vis[index_cp[:, 2],r])
        t3phi_modelA[:,r] = np.absolute(t3_model[:,r])
        t3phi_modelf[:,r] = np.angle(t3_model[:,r])
        t3phi_modelf[:,r] = np.rad2deg((t3phi_modelf[:,r] + np.pi) % (2 * np.pi) - np.pi)

    return t3phi_modelf

def extract_data(oidata_filename):
    import numpy as np
    import astropy.io.fits as pyfits
    import pdb
    oidata = pyfits.open(oidata_filename)
    oi_wave = oidata['OI_WAVELENGTH'].data
    oi_vis = oidata['OI_VIS'].data
    #oi_vis2 = oidata['OI_VIS2'].data
    oi_t3 = oidata['OI_T3'].data
    waves = oi_wave['EFF_WAVE']
    u = oi_vis['UCOORD']
    v = oi_vis['VCOORD']
    sta_index_vis = oi_vis['STA_INDEX']
    vis = oi_vis['VISAMP']
    vis_err = oi_vis['VISAMPERR']
    #vis2 = oi_vis2['VIS2DATA']
    #vis2_err = oi_vis2['VIS2ERR']
    phi = oi_vis['VISPHI']
    phi_err = oi_vis['VISPHIERR']
    flag_vis =  oi_vis['FLAG']
    t3 = oi_t3['T3PHI']
    t3_err = oi_t3['T3PHIERR']
    flag_t3 = oi_t3['FLAG']

    sta_index_cp = oi_t3['STA_INDEX']
    u1 = oi_t3['U1COORD']
    u2 = oi_t3['U2COORD']
    v1 = oi_t3['V1COORD']
    v2 = oi_t3['V2COORD']

    ##########
    # To compute the UV coordinates of the closure phases:
    uv_cp = np.zeros([u1.shape[0]])
    u_cp = np.zeros([u1.shape[0]])
    v_cp = np.zeros([u1.shape[0]])
    u3 = -1.0 * (u1 + u2)
    v3 = -1.0 * (v1 + v2)

    for j in range(u1.shape[0]):
        uv1 = np.sqrt((u1[j]) ** 2 + (v1[j]) ** 2)
        uv2 = np.sqrt((u2[j]) ** 2 + (v2[j]) ** 2)
        uv3 = np.sqrt((u3[j]) ** 2 + (v3[j]) ** 2)
        if uv1 >= uv2 and uv1 >= uv3:
            uv_cp[j] = uv1
            u_cp[j] = u1[j]
            v_cp[j] = v1[j]
        elif uv2 >= uv1 and uv2 >= uv3:
            uv_cp[j] = uv2
            u_cp[j] = u2[j]
            v_cp[j] = v2[j]
        elif uv3 >= uv1 and uv3 >= uv2:
            uv_cp[j] = uv3
            u_cp[j] = u3[j]
            v_cp[j] = v3[j]

    uv = np.zeros([u.shape[0], waves.shape[0]])
    uv_cpt  = np.zeros([uv_cp.shape[0], waves.shape[0]])
    for i in range(uv.shape[0]):
        uv[i,:] = np.sqrt(u[i]**2 + v[i]**2)/waves[:]
    
    for i in range(uv_cp.shape[0]):
        uv_cpt[i,:] = uv_cp[i]/waves[:]

    #observables = {'uv': uv, 'u':u, 'v':v, 'waves':waves, 'vis2':vis2, 'vis2_err':vis2_err, 'phi':phi, 'phi_err':phi_err, 'uv_cp':uv_cpt, 't3':t3, 't3_err':t3_err, 'flag_vis2':flag_vis2,\
    # 'flag_t3':flag_t3, 'sta_vis':sta_index_vis, 'sta_cp':sta_index_cp}

    observables = {'uv': uv, 'u': u, 'v': v, 'waves': waves, 'vis': vis, 'vis_err': vis_err, 'phi':phi, 'phi_err':phi_err, 'uv_cp': uv_cpt, 't3': t3, 't3_err': t3_err, 'flag_vis': flag_vis, \
        'flag_t3':flag_t3, 'sta_vis':sta_index_vis, 'sta_cp':sta_index_cp}

    return observables

def compute_obs(oidata_filename, im_filename, scale, nbl, ncp):
    import numpy as np
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    from skimage.restoration import unwrap_phase
    import pdb
    oidata = pyfits.open(oidata_filename)
    oi_wave = oidata['OI_WAVELENGTH'].data
    oi_vis2 = oidata['OI_VIS2'].data
    oi_t3 = oidata['OI_T3'].data
    waves = oi_wave['EFF_WAVE']
    u = oi_vis2['UCOORD']
    v = oi_vis2['VCOORD']
    sta_index_vis = oi_vis2['STA_INDEX']

    vis2 = oi_vis2['VIS2DATA']
    t3 = oi_t3['T3PHI']
    sta_index_cp = oi_t3['STA_INDEX']
    u1 = oi_t3['U1COORD']
    u2 = oi_t3['U2COORD']
    v1 = oi_t3['V1COORD']
    v2 = oi_t3['V2COORD']

    ##########
    # To compute the UV coordinates of the closure phases:
    uv_cp = np.zeros([u1.shape[0]])
    u_cp = np.zeros([u1.shape[0]])
    v_cp = np.zeros([u1.shape[0]])
    u3 = -1.0 * (u1 + u2)
    v3 = -1.0 * (v1 + v2)

    for j in range(u1.shape[0]):
        uv1 = np.sqrt((u1[j]) ** 2 + (v1[j]) ** 2)
        uv2 = np.sqrt((u2[j]) ** 2 + (v2[j]) ** 2)
        uv3 = np.sqrt((u3[j]) ** 2 + (v3[j]) ** 2)
        if uv1 >= uv2 and uv1 >= uv3:
            uv_cp[j] = uv1
            u_cp[j] = u1[j]
            v_cp[j] = v1[j]
        elif uv2 >= uv1 and uv2 >= uv3:
            uv_cp[j] = uv2
            u_cp[j] = u2[j]
            v_cp[j] = v2[j]
        elif uv3 >= uv1 and uv3 >= uv2:
            uv_cp[j] = uv3
            u_cp[j] = u3[j]
            v_cp[j] = v3[j]

    if len(vis2.shape) < 2:
        model_vis = np.zeros([vis2.shape[0], 1])
        model_phase = np.zeros([vis2.shape[0], 1])
        model_t3phi = np.zeros([t3.shape[0], 1])
    else:
        model_vis = np.zeros([vis2.shape[0], vis2.shape[1]])
        model_phase = np.zeros([vis2.shape[0], vis2.shape[1]])
        model_t3phi = np.zeros([t3.shape[0], t3.shape[1]])

    #model_vis = np.zeros([vis2.shape[0]])
    #model_phase = np.zeros([vis2.shape[0]])
    #model_t3phi = np.zeros([t3.shape[0]])

    for j in range(model_vis.shape[1]):  # Wavelengths
        for k in range(model_vis.shape[0]):  # Baselines
            u_points = u[k] / waves[j]
            v_points = v[k] / waves[j]
            model_vis[k, j], model_phase[k, j] = compute_vis_matrix(im_filename, u_points, v_points, mas2rad(scale))
    #pdb.set_trace()
    #model_vis = model_vis.reshape(-1,1)
    #model_phase = model_phase.reshape(-1,1)
    #model_t3phi = model_t3phi.reshape(-1,1)
    model_t3phi = compute_closure_phases(nbl, ncp, sta_index_vis, sta_index_cp, model_vis, model_phase)

    return model_vis, model_phase, model_t3phi


def compute_obs_wave(oidata_filename, im_filename, scale, nbl, ncp, wave):
    import numpy as np
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    from skimage.restoration import unwrap_phase
    import pdb
    oidata = pyfits.open(oidata_filename)
    oi_wave = oidata['OI_WAVELENGTH'].data
    oi_vis = oidata['OI_VIS'].data
    oi_t3 = oidata['OI_T3'].data
    waves = oi_wave['EFF_WAVE']
    u = oi_vis['UCOORD']
    v = oi_vis['VCOORD']
    sta_index_vis = oi_vis['STA_INDEX']

    vis = oi_vis['VISAMP']
    t3 = oi_t3['T3PHI']
    sta_index_cp = oi_t3['STA_INDEX']
    u1 = oi_t3['U1COORD']
    u2 = oi_t3['U2COORD']
    v1 = oi_t3['V1COORD']
    v2 = oi_t3['V2COORD']

    ##########
    # To compute the UV coordinates of the closure phases:
    uv_cp = np.zeros([u1.shape[0]])
    u_cp = np.zeros([u1.shape[0]])
    v_cp = np.zeros([u1.shape[0]])
    u3 = -1.0 * (u1 + u2)
    v3 = -1.0 * (v1 + v2)

    for j in range(u1.shape[0]):
        uv1 = np.sqrt((u1[j]) ** 2 + (v1[j]) ** 2)
        uv2 = np.sqrt((u2[j]) ** 2 + (v2[j]) ** 2)
        uv3 = np.sqrt((u3[j]) ** 2 + (v3[j]) ** 2)
        if uv1 >= uv2 and uv1 >= uv3:
            uv_cp[j] = uv1
            u_cp[j] = u1[j]
            v_cp[j] = v1[j]
        elif uv2 >= uv1 and uv2 >= uv3:
            uv_cp[j] = uv2
            u_cp[j] = u2[j]
            v_cp[j] = v2[j]
        elif uv3 >= uv1 and uv3 >= uv2:
            uv_cp[j] = uv3
            u_cp[j] = u3[j]
            v_cp[j] = v3[j]


    model_vis = np.zeros([vis.shape[0]])
    model_phase = np.zeros([vis.shape[0]])
    model_t3phi = np.zeros([t3.shape[0]])

    #for j in range(model_vis.shape[1]):  # Wavelengths
    for k in range(model_vis.shape[0]):  # Baselines
        u_points = u[k] / wave
        v_points = v[k] / wave
        model_vis[k], model_phase[k] = compute_vis_matrix(im_filename, u_points, v_points, mas2rad(scale))
    #pdb.set_trace()
    model_vis = model_vis.reshape(-1,1)
    model_phase = model_phase.reshape(-1,1)
    model_t3phi = model_t3phi.reshape(-1,1)
    model_t3phi = compute_closure_phases(nbl, ncp, sta_index_vis, sta_index_cp, model_vis, model_phase)

    return model_vis, model_phase, model_t3phi





def compute_obs_fft(oidata_filename, im_filename, scale, nbl, ncp):
    import numpy as np
    import astropy.io.fits as pyfits
    import matplotlib.pyplot as plt
    from skimage.restoration import unwrap_phase
    import pdb
    oidata = pyfits.open(oidata_filename)
    oi_wave = oidata['OI_WAVELENGTH'].data
    oi_vis2 = oidata['OI_VIS2'].data
    oi_t3 = oidata['OI_T3'].data
    waves = oi_wave['EFF_WAVE']
    u = oi_vis2['UCOORD']
    v = oi_vis2['VCOORD']
    sta_index_vis = oi_vis2['STA_INDEX']

    vis2 = oi_vis2['VIS2DATA']
    t3 = oi_t3['T3PHI']
    sta_index_cp = oi_t3['STA_INDEX']
    u1 = oi_t3['U1COORD']
    u2 = oi_t3['U2COORD']
    v1 = oi_t3['V1COORD']
    v2 = oi_t3['V2COORD']

    ##########
    # To compute the UV coordinates of the closure phases:
    uv_cp = np.zeros([u1.shape[0]])
    u_cp = np.zeros([u1.shape[0]])
    v_cp = np.zeros([u1.shape[0]])
    u3 = -1.0 * (u1 + u2)
    v3 = -1.0 * (v1 + v2)

    for j in range(u1.shape[0]):
        uv1 = np.sqrt((u1[j]) ** 2 + (v1[j]) ** 2)
        uv2 = np.sqrt((u2[j]) ** 2 + (v2[j]) ** 2)
        uv3 = np.sqrt((u3[j]) ** 2 + (v3[j]) ** 2)
        if uv1 >= uv2 and uv1 >= uv3:
            uv_cp[j] = uv1
            u_cp[j] = u1[j]
            v_cp[j] = v1[j]
        elif uv2 >= uv1 and uv2 >= uv3:
            uv_cp[j] = uv2
            u_cp[j] = u2[j]
            v_cp[j] = v2[j]
        elif uv3 >= uv1 and uv3 >= uv2:
            uv_cp[j] = uv3
            u_cp[j] = u3[j]
            v_cp[j] = v3[j]

    #if len(vis2.shape) < 2:
        #model_vis = np.zeros([vis2.shape[0], 1])
        #model_phase = np.zeros([vis2.shape[0], 1])
        #model_t3phi = np.zeros([t3.shape[0], 1])
    #else:
    #    model_vis = np.zeros([vis2.shape[0], vis2.shape[1]])
    #    model_phase = np.zeros([vis2.shape[0], vis2.shape[1]])
    #    model_t3phi = np.zeros([t3.shape[0], t3.shape[1]])
    u_points = np.zeros([vis2.shape[0], vis2.shape[1]])
    v_points = np.zeros([vis2.shape[0], vis2.shape[1]])

    for j in range(vis2.shape[1]):  # Wavelengths
        for k in range(vis2.shape[0]):  # Baselines
            u_points[k, j] = u[k] / waves[j]
            v_points[k, j] = v[k] / waves[j]

    sz = np.shape(im_filename)
    nfft = NUFFT()
    Nd = (sz[0], sz[1])
    Kd = (2*sz[0], 2*sz[1])
    Jd = (6, 6)
    u_points = u_points.reshape(-1)
    v_points = v_points.reshape(-1)
    om = np.squeeze(np.dstack((2 * np.pi * mas2rad(scale) * u_points,
                                       -2 * np.pi * mas2rad(scale) * v_points)))
    nfft.plan(om, Nd, Kd, Jd)
    y = nfft.forward(im_filename)
    vis = np.absolute(y)
    phase = np.angle(y, deg=True)

    model_vis = vis.reshape(-1, vis2.shape[1])
    model_phase = phase.reshape(-1, vis2.shape[1])
    model_t3phi = compute_closure_phases(nbl, ncp, sta_index_vis, sta_index_cp, model_vis, model_phase)

    return model_vis, model_phase, model_t3phi
