import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.io.fits as pyfits
import oitools
from readcol import *
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import gridspec

oi_data = 's0_a200_COMB_pa_2018_HD163296.fits'

[ims] = readcol('new_mean_ims_ring_2018.txt', twod=False)
tit = 'Observables from BSMEM Images [Ring Model 2019]'
year= 2019

#wave_range = [2.30e-6, 2.40e-6]
scale = 0.1 ## mas

oidata = pyfits.open(oi_data)
oi_wave = oidata['OI_WAVELENGTH'].data
#oi_vis = oidata['OI_VIS'].data
oi_vis2 = oidata['OI_VIS2'].data
oi_t3 = oidata['OI_T3'].data
waves = oi_wave['EFF_WAVE']
#vis = oi_vis['VISAMP']
#vis_err = oi_vis['VISAMPERR']
#phase = oi_vis['VISPHI']
#phase_err = oi_vis['VISPHIERR']
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

values = range(len(waves))
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
fig2 = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])


for i in range(len(ims)):
    im_rec = np.squeeze(pyfits.getdata(ims[i]))
    wave_min = waves[i] - 0.01e-6
    wave_max = waves[i] + 0.01e-6

    [ind] = np.where((waves > wave_min) & (waves < wave_max))
    print(ind)
    vis_synth, phi_synth, t3phi_synth = oitools.compute_obs(oi_data, im_rec, scale, 6, 4)

    colorVal = scalarMap.to_rgba(values[i])
    uv_range = np.sqrt(u ** 2 + v ** 2) / waves[ind]
    ax1.errorbar(uv_range, vis2[:, ind], yerr=5*vis2_err[:, ind], fmt='o', color='black')
    ax1.plot(uv_range, (vis_synth[:, ind]) ** 2, 'o', color=colorVal, zorder=500, alpha=0.8,  label = str(np.round(waves[ind][0]/1e-6, 3))+' $\mu$m')
    ax2.errorbar(uv_cp / waves[ind], t3[:, ind], yerr=0.3, fmt='o', color='black')
    ax2.plot(uv_cp / waves[ind], t3phi_synth[:, ind], 'o', color=colorVal, zorder=500, alpha=0.8)
    ax3.plot(uv_range, (vis2[:,ind] - (vis_synth[:,ind] ** 2)), 'o', color=colorVal)
    ax4.plot(uv_cp / waves[ind], (t3[:,ind] - t3phi_synth[:,ind]), 'o', color=colorVal)
    mean_res_vis = np.mean((vis2[:,ind] - (vis_synth[:,ind] ** 2)))
    mean_res_cp = np.mean((t3[:,ind] - t3phi_synth[:,ind]))
    std_res_vis = np.std((vis2[:,ind] - (vis_synth[:,ind] ** 2)))
    std_res_cp = np.std((t3[:,ind] - t3phi_synth[:,ind]))
    print('Residual V2:', mean_res_vis, std_res_vis)
    print('Residual CP:', mean_res_cp, std_res_cp)

ax3.set_ylabel('M-D')
ax4.set_ylabel('M-D')
ax1.set_ylim([0, 1.0])
ax2.set_ylim([-20, 20])
ax3.set_ylim([-0.05, 0.05])
ax4.set_ylim([-5, 5])
ax3.set_xlabel('Spatial Freq. [1/rad]')
ax1.set_ylabel('Squared Visibility')
ax4.set_xlabel('Spatial Freq. [1/rad]')
ax2.set_ylabel('Closure Phases [deg]')
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
yticks = ax4.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
fig2.legend()
fig2.suptitle(tit)
fig2.subplots_adjust(hspace=0.0)
#fig2.savefig('model_azimuth_inter'+str(year)+'.pdf', bbox_tight='true')


ax1.set_ylabel('V$^2$')
ax1.set_xlabel('Spatial Frequency [1/rad]')
ax2.set_ylabel('Closure Phases [deg]')
ax2.set_xlabel('Spatial Frequency [1/rad]')
plt.show()
#fig2.savefig('ims_ring_inter'+str(year)+'.pdf', bbox_tight='true')

pdb.set_trace()
