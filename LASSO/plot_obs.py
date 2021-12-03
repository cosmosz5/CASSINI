import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.io.fits as pyfits
import oitools


oi_data = 'COMB_JWST_SAM_tot.fits'
im_rec = np.squeeze(pyfits.getdata('reconvered_im_lasso_0.001.fits'))
wave_range = [3.0e-6, 5.0e-6]
scale = 10.0 ## mas

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

[ind] = np.where((waves >= wave_range[0]) & (waves <= wave_range[1]))

vis_synth, phi_synth, t3phi_synth = oitools.compute_obs(oi_data, im_rec, scale)
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

for i in range(len(ind)):
    uv_range = np.sqrt(u ** 2 + v ** 2) / waves[ind[i]]
    ax1.errorbar(uv_range, vis2[:,ind[i]], yerr=vis2_err[:,ind[i]], fmt='o', color='black')
    ax1.plot(uv_range, (vis_synth[:,ind[i]])**2, 'or', zorder=500, alpha=0.8)
    ax2.errorbar(uv_cp/waves[ind[i]], t3[:,ind[i]], yerr=t3_err[:,ind[i]], fmt='o', color='black' )
    ax2.plot(uv_cp/waves[ind[i]], t3phi_synth[:,ind[i]], 'or', zorder=500, alpha=0.8)

ax1.set_ylabel('V$^2$')
ax1.set_xlabel('Spatial Frequency [1/rad]')
ax2.set_ylabel('Closure Phases [deg]')
ax2.set_xlabel('Spatial Frequency [1/rad]')
plt.show()
fig.savefig('observables.png', bbox_tight='true')

pdb.set_trace()
