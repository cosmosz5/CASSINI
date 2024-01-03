import amical
import pdb
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
from nrm_analysis import InstrumentData, nrm_core
import matplotlib.colors as colors
import matplotlib.cm as cm
#from nrm_analysis.misctools.implane2oifits import calibrate_oifits
import oitools
from readcol import *

[files] = readcol('corr_data.txt', twod=False)

for i in range(len(files)):
    nrm_file = files[i]
    clean_param = {
        "isz": 80,  # final cropped image size [pix]
        "r1": 16,  # Inner radius to compute sky [pix]
        "dr": 2,  # Outer radius (r2 = r1 + dr)
        "apod": True,  # If True, apply windowing
        "window": 16,  # FWHM of the super-gaussian (for windowing)
    }

    # Firsly, check if the input parameters are valid
    #amical.show_clean_params(nrm_file, **clean_param)
    cube_cleaned = amical.select_clean_data(nrm_file, **clean_param, clip=False)
    params_ami = {
        "peakmethod": "fft",
        "maskname": "g7",  # 7 holes mask of NIRISS
        "targetname": "binary",
        "filtname": "F380M",
        "instrum": "NIRISS"
    }
    bs = amical.extract_bs(cube_cleaned, nrm_file, **params_ami)

    cp_matrix = bs.matrix.cp_arr
    v2_matrix = bs.matrix.v2_arr
    u = bs.u
    v = bs.v

    values = range(21)
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    fig2, axs = plt.subplots(3, 7, figsize=(24, 8))
    r = 0;
    s = 0
    for i in range(v2_matrix.shape[1]):
        colorVal = scalarMap.to_rgba(values[i])
        n, bins, patches = axs[r, s].hist(v2_matrix[:, i], bins='auto', density=True, color=colorVal)
        axs[r, s].annotate('BL:' + str(np.round(np.sqrt(u[i] ** 2 + v[i] ** 2), 2)), xy=(0.63, 0.85),
                           xycoords='axes fraction')
        axs[r, s].annotate('PA:' + str(np.round(np.rad2deg(np.arctan2(u[i], v[i])), 2)), xy=(0.63, 0.75),
                           xycoords='axes fraction')

        # best fit of data
        (mu, sigma) = norm.fit(v2_matrix[:,i])
        # add a 'best fit' line
        best_fit_line = norm.pdf(bins, mu, sigma)
        axs[r, s].plot(bins, best_fit_line, '--', linewidth=2, color='red')
        axs[r,s].set_title('$\mu=$'+str(np.round(mu, 2))+' $\sigma$='+str(np.round(sigma,4)))
        axs[r,s].set_xlabel('V$^2$')
        axs[r,s].set_ylabel('# Frames')
        axs[r, s].xaxis.set_major_locator(plt.MaxNLocator(4))

        if s == 6:
            s = 0
            r = r + 1
        else:
            s = s + 1
    fig2.suptitle('v2_amical_' + nrm_file[:-5])
    fig2.savefig('v2_amical_' + nrm_file[:-5] + '.png', bbox_inches='tight')

    

    values = range(35)
    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    fig2, axs = plt.subplots(5, 7, figsize=(24, 12))
    r = 0;
    s = 0
    for i in range(cp_matrix.shape[1]):
        colorVal = scalarMap.to_rgba(values[i])
        n, bins, patches= axs[r, s].hist(cp_matrix[:, i], bins='auto', density=True, color=colorVal)
        # axs[r, s].annotate('BL:' + str(np.round(np.sqrt(bl_xp[i] ** 2 + bl_yp[i] ** 2), 2)), xy=(0.63, 0.85),
        #                   xycoords='axes fraction')
        # axs[r, s].annotate('PA:' + str(np.round(np.rad2deg(np.arctan2(bl_xp[i], bl_yp[i])), 2)), xy=(0.63, 0.75),
        #                   xycoords='axes fraction')

        # best fit of data
        (mu, sigma) = norm.fit(cp_matrix[:,i])
        # add a 'best fit' line
        best_fit_line = norm.pdf(bins, mu, sigma)
        axs[r, s].plot(bins, best_fit_line, '--', linewidth=2, color='red')
        axs[r,s].set_title('$\mu=$'+str(np.round(mu, 4))+' $\sigma$='+str(np.round(sigma,4)), fontsize=8)
        axs[r,s].set_xlabel('CP [deg]')
        axs[r,s].set_ylabel('# Frames')
        axs[r, s].xaxis.set_major_locator(plt.MaxNLocator(4))

        if s == 6:
            s = 0
            r = r + 1
        else:
            s = s + 1
    fig2.suptitle('cp_amical_' + nrm_file[:-5])
    fig2.savefig('cp_amical_' + nrm_file[:-5] + '.png', bbox_inches='tight')

    np.savez(data_file[:-5] + '_amical.npz', V2 = v2_matrix, CP = cp_matrix)

    plt.close('all')
    
pdb.set_trace()


















pdb.set_trace()


nrm_file = "corr_c_pa90_sep200_F380M_sky_81px_x11__F380M_81_flat_x11_read_jitt_flat__00_mir.fits"
clean_param = {
    "isz": 80,  # final cropped image size [pix]
    "r1": 16,  # Inner radius to compute sky [pix]
    "dr": 2,  # Outer radius (r2 = r1 + dr)
    "apod": True,  # If True, apply windowing
    "window": 16,  # FWHM of the super-gaussian (for windowing)
}

# Firsly, check if the input parameters are valid
amical.show_clean_params(nrm_file, **clean_param)
cube_cleaned = amical.select_clean_data(nrm_file, **clean_param, clip=False)
params_ami = {
    "peakmethod": "fft",
    "maskname": "g7",  # 7 holes mask of NIRISS
    "targetname": "binary",
    "filtname": "F380M",
    "instrum": "NIRISS"
}
bs_c = amical.extract_bs(cube_cleaned, nrm_file, **params_ami)

cal = amical.calibrate(bs, bs_c)

amical.show(cal, cmax=1, vmin=0.998, vmax=1.002)


pdb.set_trace()




fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))

vis2 = cal.vis2
vis2_err = cal.e_vis2
cp = cal.cp
cp_err = cal.e_cp
ucoord = cal.u
vcoord = cal.v
wave = cal.wl


# set up instrument-specfic part
nirissdata = InstrumentData.NIRISS(filt="F380M", objname="binary")
targfiles =[nrm_file]
ff =  nrm_core.FringeFitter(nirissdata, oversample = 1, savedir="targ", npix=81)
ff.fit_fringes(targfiles)
calfiles =["c_pa0_sep200_F380M_sky_81px_x11__F380M_81_flat_x11__00.fits"]
ff2 =  nrm_core.FringeFitter(nirissdata, oversample = 1, savedir="cal", npix=81, save_txt_only=False)
ff2.fit_fringes(calfiles)

targdir = "targ/t_pa0_sep200_F380M_sky_81px_x11__F380M_81_flat_x11__00/"
caldir = "cal/c_pa0_sep200_F380M_sky_81px_x11__F380M_81_flat_x11__00/"
calib = nrm_core.Calibrate([targdir, caldir], nirissdata, savedir = "my_calibrated/")
#calib.save_to_oifits("niriss_implaneia.oifits")
cp_impia = calib.cp_calibrated_deg[0,:]
cp_err_impia = calib.cp_calibrated_deg[0,:]
vis2_impia = calib.v2_calibrated[0,:]
vis2_err_impia = calib.v2_err_calibrated[0,:]


oi_filename = 'CALIB_SIM_DATA_uncalib_t_pa0_sep200_F380M_sky_81px_x11__F380M_81_flat_x11__00.oifits'
oidata = pyfits.open(oi_filename)
vis2_fft = oidata['OI_VIS2'].data['VIS2DATA']
vis2_fft_err = oidata['OI_VIS2'].data['VIS2ERR']
cp_fft = oidata['OI_T3'].data['T3PHI']
cp_fft_err = oidata['OI_T3'].data['T3PHIERR']
ucoord_fft =  oidata['OI_VIS2'].data['UCOORD']
vcoord_fft = oidata['OI_VIS2'].data['VCOORD']
wave_fft = oidata['OI_WAVELENGTH'].data['EFF_WAVE']

u1 = oidata['OI_T3'].data['U1COORD']
u2 = oidata['OI_T3'].data['U2COORD']
v1 = oidata['OI_T3'].data['V1COORD']
v2 = oidata['OI_T3'].data['V2COORD']

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


ax1.errorbar(np.sqrt(ucoord**2 + vcoord**2)/wave, vis2, yerr=vis2_err, fmt='o', color='red', label='amical')
ax1.errorbar(np.sqrt(ucoord_fft**2 + vcoord_fft**2)/wave_fft, vis2_fft, yerr=vis2_fft_err, fmt='o', color='blue', label='jsb')
ax1.errorbar(np.sqrt(ucoord_fft**2 + vcoord_fft**2)/wave_fft, vis2_impia, yerr=vis2_err_impia, fmt='o', color='green', label='implaneia')
#ax1.errorbar(np.sqrt(ucoord_fft**2 + vcoord_fft**2)/wave_fft, vis2_fft, yerr=vis2_fft_err, fmt='o', color='blue', label='implaneia')
ax2.errorbar(bs.bl_cp/wave, cp, yerr=cp_err, fmt='o', color='red')
ax2.errorbar(uv_cp/wave, cp_fft, yerr=cp_fft_err, fmt='o', color='blue')
ax2.errorbar(bs.bl_cp/wave, cp_impia, yerr=cp_err_impia, fmt='o', color='green')

ax1.set_ylim([0.97, 1.1])
ax1.set_ylabel('V$^2$')
ax1.set_xlabel('Spatial Frequency [1/rad]')
ax2.set_ylabel('Closure Phases [degrees]')
ax2.set_xlabel('Spatial Frequency [1/rad]')
ax1.legend()

plt.show()
pdb.set_trace()
