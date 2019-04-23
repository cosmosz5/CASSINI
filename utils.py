#### Compressed sensing utilities #####

def dct2(x):
    import scipy.fftpack as spfft
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    import scipy.fftpack as spfft
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def mas2rad(x):
    import numpy as np
    y=x/3600.0/1000.0*np.pi/180.0
    y=np.array(y)
    return y

def compute_vis(im, u, v, scale):
    import numpy as np
    import math
    sz= np.shape(im)
    if sz[0] % 2 == 0:
        x, y = np.mgrid[-np.floor(sz[1]/2-1):np.floor(sz[1]/2):sz[1]*1j, np.floor(sz[0]/2-1):-np.floor(sz[0]/2):sz[0]* 1j]
    else:
        x, y = np.mgrid[-np.floor(sz[1]/2):np.floor(sz[1]/2):sz[1]*1j, np.floor(sz[0]/2):-np.floor(sz[0]/2):sz[0]* 1j]
    x=x*scale
    y=y*scale
    arg=-2.0*math.pi*(u*y+v*x)
    visib=np.linalg.norm([np.sum(im*np.cos(arg)),np.sum(im*np.sin(arg))])
    reales = np.sum(im*np.cos(arg))
    imaginarios = np.sum(im*np.sin(arg))
    phase=np.degrees(np.arctan2(imaginarios, reales))
    if phase > 180.0:
        phase = phase - 360.0
    if phase < -180.0:
        phase = phase + 360.0

    return visib, phase

def compute_vis_matrix(im, u, v, scale):
    import numpy as np
    import math
    sz = np.shape(im)
    if sz[0] % 2 == 0:
        x, y = np.mgrid[-np.floor(sz[1] / 2 - 1):np.floor(sz[1] / 2):sz[1] * 1j,
               np.floor(sz[0] / 2 - 1):-np.floor(sz[0] / 2):sz[0] * 1j]
    else:
        x, y = np.mgrid[-np.floor(sz[1] / 2):np.floor(sz[1] / 2):sz[1] * 1j,
               np.floor(sz[0] / 2):-np.floor(sz[0] / 2):sz[0] * 1j]
    x = x * scale
    y = y * scale
    x = x.reshape(-1)
    y = y.reshape(-1)
    im = im.reshape(-1)
    arg = -2.0 * math.pi * (u * y + v * x)
    reales =


