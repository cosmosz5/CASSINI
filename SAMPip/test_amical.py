import amical
import pdb
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.stats import norm
#from nrm_analysis.misctools.implane2oifits import calibrate_oifits
import oitools
from readcol import *

[files] = readcol('corr_data_t.txt', twod=False)

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

    np.savez(nrm_file[:-5] + '_amical.npz', V2 = v2_matrix, CP = cp_matrix)

    plt.close('all')
    
pdb.set_trace()

