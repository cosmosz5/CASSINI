import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from readcol import *
import oitools
import pylab
from pylbfgs import owlqn
import pdb
pylab.ion()

# def evaluate(x, g, step):
#     """An in-memory evaluation callback."""
#
#     # we want to return two things:
#     # (1) the norm squared of the residuals, sum((Ax-b).^2), and
#     # (2) the gradient 2*A'(Ax-b)
#
#     x = x.reshape(-1, 1)
#
#     Ax = Dict.dot(x)
#
#     Ax[nvis:, 0] = np.rad2deg(np.arctan(np.tan(np.deg2rad(Ax[nvis:, 0]))))
#     b[nvis:, 0] = np.rad2deg(np.arctan(np.tan(np.deg2rad(b[nvis:, 0]))))
#     Axb = (Ax - b)
#     Axb[0:nvis, 0] = Axb[0:nvis, 0] / (nvis * err[0:nvis,0])
#     Axb[nvis:, 0] = Axb[nvis:, 0] / (nt3phi * err[nvis:, 0])
#
#     fx = np.sum(np.power(Axb, 2))
#     #Axb2 = np.reshape(Axb.T.flat, (-1, 1))
#     AtAxb = 2 * (Dict.T.dot(Axb))
#
#     np.copyto(g, AtAxb)
#
#     return fx
#
# def progress(x, g, fx, xnorm, gnorm, step, k, ls):
#     # Print variables to screen or file or whatever. Return zero to
#     # continue algorithm; non-zero will halt execution.
#     print(fx, xnorm, gnorm, step, k, ls)
#
#     if step <= 0.05:
#         return 1
#     else:
#         return 0

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def fista(A, b, err, nvis, nt3phi, l, maxit):
    import time
    from math import sqrt
    import numpy as np
    from scipy import linalg
    import sklearn.preprocessing as prepro

    x = np.zeros(A.shape[1])
    x = x.reshape(-1, 1)

    for ty in range(A.shape[1]):
        A[nvis:,ty] = np.arctan(np.tan(np.deg2rad(A[nvis:,ty])))
    A = prepro.normalize(A, norm='l2', axis=0)

    b[nvis:, 0] = np.arctan(np.tan(np.deg2rad(b[nvis:, 0])))
    b = prepro.normalize(b, norm='l2', axis=0)

    pobj = []
    t = 1
    z = x.copy()
    L = linalg.norm(A, ord=2) ** 2 #### Lipschitz constant
    time0 = time.time()
    for _ in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))
        print(this_pobj)
    times, pobj = map(np.array, zip(*pobj))
    return x, pobj, times



def lasso(A, b, err, nvis, nt3phi, l, maxit):
    from pyunlocbox import functions
    from pyunlocbox import solvers
    import sklearn.preprocessing as prepro

    x = np.zeros(A.shape[1])
    x = x.reshape(-1, 1)

    for ty in range(A.shape[1]):
        A[nvis:, ty] = np.arctan(np.tan(np.deg2rad(A[nvis:, ty])))
    A = prepro.normalize(A, norm='l2', axis=0)

    b[nvis:, 0] = np.arctan(np.tan(np.deg2rad(b[nvis:, 0])))
    b = prepro.normalize(b, norm='l2', axis=0)

    f1 = functions.norm_l1(lambda_=l)
    f2 = functions.norm_l2(y=b, A=A)
    step = 0.5 / np.linalg.norm(A, ord=2) ** 2
    solver = solvers.forward_backward(step=step)
    ret = solvers.solve([f1, f2], x, solver, rtol=1e-6, maxit=maxit)
    return ret



oi_data = 'CALIB_SIM_DATA_uncalib_t_disk_small2_0__PSF_MASK_NRM_F430M_x11_0.82_ref__00.oifits'
#input_path = ['/Users/donut/Library/Mobile Documents/com~apple~CloudDocs/JWST_CS/dataset/disks_clean/']
input_path = ['/Users/donut/Library/Mobile Documents/com~apple~CloudDocs/JWST_CS/dataset/tot_forms/']
input_files = readcol(input_path[0] +'ring_gauss.txt', twod=True)
scale = 10.0 ## mas
hyperparameter = 0.15

maxit = 10000

for i in range(len(input_files)):
    temp = input_path[0] + input_files[i][0]
    im_atom = pyfits.getdata(temp)
    if i == 0:
        v_model, phi_model, t3phi_model = oitools.compute_obs(oi_data, im_atom, scale)
    else:
        v_model_temp, phi_model_temp, t3phi_model_temp = oitools.compute_obs(oi_data, im_atom, scale)
        v_model = np.dstack((v_model, v_model_temp))
        phi_model = np.dstack((phi_model, phi_model_temp))
        t3phi_model = np.dstack((t3phi_model, t3phi_model_temp))

oidata = pyfits.open(oi_data)
oi_wave = oidata['OI_WAVELENGTH'].data
oi_vis = oidata['OI_VIS'].data
oi_vis2 = oidata['OI_VIS2'].data
oi_t3 = oidata['OI_T3'].data
waves = oi_wave['EFF_WAVE']

vis = oi_vis['VISAMP']
vis_err = oi_vis['VISAMPERR']
phase = oi_vis['VISPHI']
phase_err = oi_vis['VISPHIERR']
vis2 = oi_vis2['VIS2DATA']
vis2_err = oi_vis2['VIS2ERR']
t3 = oi_t3['T3PHI']
t3_err = oi_t3['T3PHIERR']

u = oi_vis2['UCOORD']
v = oi_vis2['VCOORD']
u1 = oi_t3['U1COORD']
u2 = oi_t3['U2COORD']
v1 = oi_t3['V1COORD']
v2 = oi_t3['V2COORD']

nvis = vis2.shape[0]
nt3phi = t3.shape[0]
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

for j in range(len(waves)): ### For each wavelength in the data
    v2_mod = np.squeeze(v_model[:,j,:])
    phi_mod = np.squeeze(phi_model[:,j,:])
    t3phi_mod = np.squeeze(t3phi_model[:,j,:])
    Dict = np.vstack((v2_mod, phi_mod))

    if len(vis2.shape) < 2:
        b = np.hstack((vis, phase)).reshape(-1,1)
        err = np.hstack((vis_err, phase_err)).reshape(-1,1)
    else:
        b = np.vstack((np.squeeze(vis[:, j]), np.squeeze(phase[:, j]))).reshape(-1,1)

    Xat2, pobj_fista, times_fista = fista(Dict, b, err, nvis, nt3phi, hyperparameter, maxit)
    #ret = lasso(Dict, b, err, nvis, nt3phi, hyperparameter, maxit)
    #Xat2 = ret['sol']
    #Xat2 = owlqn(Dict.shape[1], evaluate, None, 0.01)
    print(Xat2)

    [ind] = np.where(np.squeeze(Xat2) !=0)

    im_rec = 0.0
    for mm in range(len(ind)):
        temp = input_path[0] + input_files[ind[mm]][0]
        im_rec += Xat2[ind[mm]]*pyfits.getdata(temp)

    vis_synth, phi_synth, t3phi_synth = oitools.compute_obs(oi_data, im_rec, scale)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
    fig2, axx1 = plt.subplots(1, 1)

    uv_range = np.sqrt(u ** 2 + v ** 2) / waves[j]
    ax1.errorbar(uv_range, vis, yerr=vis_err, fmt='o', color='black')
    ax1.plot(uv_range, vis_synth, 'or', zorder=500, alpha=0.8)
    ax2.errorbar(uv_range, phase, yerr=phase_err, fmt='o', color='black')
    ax2.plot(uv_range, phi_synth, 'or', zorder=500, alpha=0.8)

    ax3.plot(uv_range, (vis.reshape(-1,1) - vis_synth), 'o', color='black')
    ax4.plot(uv_range, (np.rad2deg(np.arctan(np.tan(np.deg2rad(phase.reshape(-1,1))))) - np.rad2deg(
            np.arctan(np.tan(np.deg2rad(phi_synth))))), 'o', color='black')

    ax1.set_ylabel('V$')
    ax3.set_ylabel('Residuals$^2$')
    ax3.set_xlabel('Spatial Frequency [1/rad]')

    ax2.set_ylabel('Fourier Phases')
    ax4.set_ylabel('Residuals$^2$ [deg]')
    ax4.set_xlabel('Spatial Frequency [1/rad]')
    fig.subplots_adjust(hspace=0.0)

    # objective = np.array(ret['objective'])
    # axx1.semilogy(objective[:, 0], label='L1-norm objective')
    # axx1.semilogy(objective[:, 1], label='L2-norm objective')
    # axx1.semilogy(np.sum(objective, axis=1), label='Global objective')
    # axx1.grid(True)
    # axx1.set_title('Convergence')
    # axx1.legend()
    # axx1.set_xlabel('Iteration number')
    # axx1.set_ylabel('Objective function value')
    print('DID IT')
    fig.savefig('observables.png', bbox_tight='true')

    pyfits.writeto('reconvered_im.fits', im_rec, overwrite=True)


pdb.set_trace()






pdb.set_trace()
