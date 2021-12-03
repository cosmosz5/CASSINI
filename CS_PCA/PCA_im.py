import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import pdb
import oitools
import cvxpy as cp
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.cm as cm
from skimage.restoration import unwrap_phase
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso
import pylops.optimization.sparsity as sparsity
import pylops
import pylab
pylab.ion()


def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def plot_regularization_path(lambd_values, beta_values):
    num_coeffs = len(beta_values[0])
    fig3, ax2 = plt.subplots(1,1)
    for i in range(num_coeffs):
        ax2.plot(lambd_values, [wi[i] for wi in beta_values])
    ax2.set_xlabel(r"$\lambda$", fontsize=16)
    ax2.set_ylabel(r"$\alpha\,\,\mathrm{values}$", fontsize=16)
    ax2.set_xscale("log")
    ax2.set_title("Regularization Path")
    fig3.savefig('regularization_path_cvxpy.png', bbox_inches='tight')
    return 0
    #plt.show()


#### MODIFY THE FOLLOWING INPUT PARAMETERS ######
cube_filename = 'cube_example.fits' ## The data cube with the reconstructed images
oifits_filename = 'oifits_example.fits' ## The oifits file with the interferometric data to be modelled

### For fitting 1 #####
n_comp = 50 ###Number of components to be extracted from the image, it has to be less or equal than the number of frames in the cube
scale = 0.1 ### Pixel scale used in the reconstruction of the images (mas)
n_comp_compiled = 10 ## Number of components to be added and stored into a .fits file for visualization purposes (it has to be <= n_comp)
display = True ## Bolean variable, it is used to display a plot of the 10 principal components in the image (in n_comp has to be equal or larger than 10)
lambd_values1 = np.logspace(-2, 3, 5)

### Secondary visibility reconstructions with sklearn and pylops
sklearn_opt = True
lambd_values2 = np.logspace(-5, -2, 5) ##Lambda values for CS solver using scikit-learn
pylops_opt = True
lambd_values3 = np.logspace(-5, -2, 5) ##Lambda values for CS solver using pylops

### Automatic from here #####
data = pyfits.getdata(cube_filename)
data_temp = data * 1.0
data = data.reshape(data.shape[0], -1)
imsize = data_temp.shape[1]
pca = PCA(n_comp)
pca.fit(data)
converted_data = pca.fit_transform(data)
data_inverted = pca.inverse_transform(converted_data)
data_inverted = data_inverted.reshape(data_temp.shape[0], data_temp.shape[1], data_temp.shape[2])
pyfits.writeto('restored_cube.fits', data_inverted[0,:,:], overwrite=True)

a_tot=0
for i in range(int(n_comp_compiled)):
    a = pca.components_[i].reshape(data_temp.shape[1], data_temp.shape[2]) * converted_data[0,i]
    a_tot = a + a_tot

pyfits.writeto('pca'+str(n_comp_compiled)+'comp.fits', a_tot, overwrite=True)
pyfits.writeto('pca'+str(n_comp_compiled)+'comp+mean.fits', a_tot+pca.mean_.reshape(imsize, imsize), overwrite=True)

if display == True:
    im_temp = pca.mean_.reshape(imsize, imsize)
    indx, indy = np.where(im_temp == np.max(im_temp))
    fig, axs = plt.subplots(2,5, figsize=(18.5, 7))
    r =0; s=0
    aa = 0
    for i in range(10):
        axs[r, s].imshow(pca.components_[i].reshape(imsize, imsize)[int(indx)-64:int(indx)+65,int(indy)-64:int(indy)+65], origin='lower', cmap=cm.jet, extent=[64*scale, -64*scale, -64*scale, 64*scale  ])
        axs[r, s].set_xlabel('Milliarcseconds (mas)')
        axs[r, s].set_ylabel('Milliarcseconds (mas)')
        axs[r, s].annotate('Component No. '+str(i),  xy=(0.2, 0.8),  xycoords='axes fraction')
        if s == 4:
            s = 0
            r = r+1
        else:
            s = s +1

    fig.savefig('principal_components.png', bbox_inches='tight')
 
comp_cube = np.zeros([n_comp+1, imsize, imsize])
comp_cube2 = np.zeros([imsize, imsize, n_comp+1])

oifile = oifits_filename
oidata = pyfits.open(oifile)
oi_wave = oidata['OI_WAVELENGTH'].data
oi_vis = oidata['OI_VIS'].data
waves = oi_wave['EFF_WAVE']
vis = oi_vis['VISAMP']
vis_err = oi_vis['VISAMPERR']
phase = oi_vis['VISPHI']
phase_err = oi_vis['VISPHIERR']
flag = oi_vis['FLAG']
u = oi_vis['UCOORD']
v = oi_vis['VCOORD']

[ind] = np.where(flag == False)
model_vis = np.zeros([ind.shape[0]])
model_phase = np.zeros([ind.shape[0]])

mod_vis_tot = 0
phase_tot = 0
fig1, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 12))
u_points = u[ind] / waves
v_points = v[ind] / waves
ax1.errorbar(np.sqrt(u_points ** 2 + v_points ** 2), vis[ind], yerr=vis_err[ind], fmt='o', color='black', label='data')
ax2.errorbar(np.sqrt(u_points ** 2 + v_points ** 2), phase[ind], yerr=phase_err[ind], fmt='o', color='black')
for i in range(n_comp+1):
    if i == 0:
        comp_cube[i, :, :] = pca.mean_.reshape(imsize, imsize)
    else:
        comp_cube[i, :, :] = pca.mean_.reshape(imsize, imsize)
        for m in range(i):
            temp = pca.components_[m-1].reshape(imsize, imsize) * converted_data[1, m-1]
            comp_cube[i, :, :] = comp_cube[i, :, :] + temp

    for k in range(ind.shape[0]):
        u_points = u[ind[k]] / waves
        v_points = v[ind[k]] / waves
        model_vis[k], model_phase[k] = oitools.compute_vis_matrix(comp_cube[i, :, :], u_points, v_points,\
                                                                  oitools.mas2rad(scale))
    u_points = u[ind] / waves
    v_points = v[ind] / waves
    if i == 0:
        ax1.plot(np.sqrt(u_points ** 2 + v_points ** 2), model_vis, 'ob', zorder=500, alpha = 0.5, label='PCA model')
    else:
        ax1.plot(np.sqrt(u_points ** 2 + v_points ** 2), model_vis, 'ob', zorder=500, alpha = 0.5)
    ax2.plot(np.sqrt(u_points ** 2 + v_points ** 2), model_phase, 'ob', zorder=500, alpha=0.5)

ax1.set_xlabel('Spatial Freq. [1/rad]')
ax1.set_ylabel('Visibilities')
ax2.set_xlabel('Spatial Freq. [1/rad]')
ax2.set_ylabel('Phases')
ax1.set_ylim([0, 1])
ax2.set_ylim([-180, 180])
ax1.grid()
ax1.legend()
ax2.grid()
fig1.savefig('PCA_visibilities_from_im.png', bbox_inches='tight')

comp_cube = np.zeros([n_comp+1, imsize, imsize])
comp_cube2 = np.zeros([imsize, imsize, n_comp+1])
############################################
for i in range(n_comp+1):
    if i == 0:
        comp_cube[i, :, :] = pca.mean_.reshape(imsize, imsize)
        comp_cube2[:, :, i] = pca.mean_.reshape(imsize, imsize)
    else:
        comp_cube[i, :, :] = pca.components_[i-1].reshape(imsize, imsize) * converted_data[0, i-1]
        comp_cube2[:, :, i] = pca.components_[i-1].reshape(imsize, imsize) * converted_data[0, i-1]
    for k in range(ind.shape[0]):
        u_points = u[ind[k]] / waves
        v_points = v[ind[k]] / waves
        model_vis[k], model_phase[k] = oitools.compute_vis_matrix(comp_cube[i, :, :], u_points, v_points,\
                                                                  oitools.mas2rad(scale))

    if i == 0:
        mod_vis_tot = model_vis * 1.0
        phase_tot = model_phase * 1.0
    else:
        mod_vis_tot = model_vis + mod_vis_tot
        phase_tot = model_phase + phase_tot

    if i == 0:
        dict_0 = np.append(model_vis, np.deg2rad(model_phase))
    else:
        rr =  np.append(model_vis, np.deg2rad(model_phase))
        dict_0 = np.dstack((dict_0,rr))

dict_0 = np.squeeze(dict_0)
with open('dict_PCA.npy', 'wb') as f:
    np.save(f, dict_0)

######

m = u[ind].shape[0]
n = n_comp
Y = np.append(vis[ind], np.deg2rad(phase[ind]))
Y_err = np.append(vis_err[ind], np.deg2rad(phase_err[ind]))

beta = cp.Variable(n+1)
lambd = cp.Parameter(nonneg=True)
problem = cp.Problem(cp.Minimize(objective_fn(dict_0, Y, beta, lambd)))

train_errors = []
test_errors = []
beta_values = []

fig4, (ax21, ax22) = plt.subplots(2,1, figsize=(8, 12))
ii = 0

values = range(len(lambd_values1))
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

for m in lambd_values1:
    colorVal = scalarMap.to_rgba(values[ii])
    lambd.value = m
    problem.solve()
    train_errors.append(mse(dict_0, Y, beta))
    test_errors.append(mse(dict_0, Y, beta))
    beta_values.append(beta.value)
    rec = np.dot(dict_0, beta.value)
    u_points = u[ind] / waves
    v_points = v[ind] / waves
    ax21.errorbar(np.sqrt(u_points**2 + v_points**2), vis[ind], yerr=vis_err[ind], fmt='o', color='black')
    ax22.errorbar(np.sqrt(u_points**2 + v_points**2), phase[ind], yerr=phase_err[ind], fmt='o', color='black')
    ax21.plot(np.sqrt(u_points ** 2 + v_points ** 2), rec[0:len(ind)], 'o', label = str(np.round(m, 3)), color=colorVal, zorder=500, alpha = 0.5)
    ax22.plot(np.sqrt(u_points ** 2 + v_points ** 2), np.rad2deg(rec[len(ind):]), 'o', color=colorVal, zorder=500)

    a_tot = 0
    rr = np.argsort(beta.value)
    for j in range(n_comp_compiled):
        aa = comp_cube[rr[int(n_comp-j)],:,:] * beta.value[rr[int(n_comp-j)]]
        a_tot = a_tot + aa
        pyfits.writeto('image_from_visibilities_pca_lambda_'+str(np.round(m, 3))+'.fits', a_tot, overwrite=True)
    ii = ii +1


fig4.suptitle('Visbilities from principal components (cvxpy solver)')
ax21.set_xlabel('Spatial Freq. [1/rad]')
ax21.set_ylabel('Visibilities')
ax22.set_xlabel('Spatial Freq. [1/rad]')
ax22.set_ylabel('Phases')
ax21.set_ylim([0, 1])
ax22.set_ylim([-180, 180])
ax21.legend()
ax21.grid()
ax22.grid()

fig4.savefig('visibilities_cvxpy.png', bbox_inches='tight')
plot_regularization_path(lambd_values1, beta_values)

if sklearn_opt == True:
    ###### OTHER LASSO IMPLEMENTATION #########
    fig10, (ax4, ax5) = plt.subplots(2,1, figsize=(8, 12))
    ax4.errorbar(np.sqrt(u_points**2 + v_points**2), vis[ind], yerr=vis_err[ind], fmt='o', color='black')
    ax5.errorbar(np.sqrt(u_points**2 + v_points**2), phase[ind], yerr=phase_err[ind], fmt='o', color='black')

    n_predictores = []
    ii = 0
    values = range(len(lambd_values2))
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

    for alpha in lambd_values2:
        colorVal = scalarMap.to_rgba(values[ii])
        modelo = LassoLars(alpha=alpha, max_iter=100, positive=False)
        modelo.fit(X = dict_0, y = Y)
        coef_no_cero = np.sum(modelo.coef_.flatten() != 0)
        n_predictores.append(coef_no_cero)

        cof = modelo.coef_.flatten()
        a_tot = 0
        for i in range(comp_cube.shape[0]):
            aa = comp_cube[i,:,:] * cof[i]
            a_tot = a_tot + aa
        pyfits.writeto('image_from_visibilities_pca2_lambda_'+str(np.round(alpha, 3))+'.fits', a_tot, overwrite=True)

        rec = np.dot(dict_0, cof)
        u_points = u[ind] / waves
        v_points = v[ind] / waves
    
        ax4.plot(np.sqrt(u_points ** 2 + v_points ** 2), rec[0:len(ind)], 'o', label = str(np.round(alpha, 3)), color=colorVal, zorder=500, alpha = 0.5)
        ax5.plot(np.sqrt(u_points ** 2 + v_points ** 2), np.rad2deg(rec[len(ind):]), 'o', color=colorVal, zorder=500)
        ii = ii + 1

    ax4.set_xlabel('Spatial Freq. [1/rad]')
    ax4.set_ylabel('Visibilities')
    ax5.set_xlabel('Spatial Freq. [1/rad]')
    ax5.set_ylabel('Phases')
    ax4.set_ylim([0, 1])
    ax5.set_ylim([-180, 180])
    ax4.legend()
    ax4.grid()
    ax5.grid()
    fig10.savefig('visibilities_sklearn.png', bbox_inches='tight')

    fig7, ax = plt.subplots(figsize=(7, 3.84))
    ax.plot(lambd_values2, n_predictores)
    ax.set_xscale('log')
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('Number of non-zero coef.')
    fig7.savefig('non_zero_coefficients.png', bbox_inches='tight')

if pylops_opt == True:
######################################################
    fig11, (ax6, ax7) = plt.subplots(2,1, figsize=(8, 12))
    ax6.errorbar(np.sqrt(u_points**2 + v_points**2), vis[ind], yerr=vis_err[ind], fmt='o', color='black')
    ax7.errorbar(np.sqrt(u_points**2 + v_points**2), phase[ind], yerr=phase_err[ind], fmt='o', color='black')

    values = range(len(lambd_values3))
    jet = plt.get_cmap('jet')
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    ii = 0
    for m in lambd_values3:
        colorVal = scalarMap.to_rgba(values[ii])
        sigma = m
        Aop = pylops.MatrixMult(dict_0)
        X_mp = sparsity.OMP(Aop, Y,  niter_outer=1000, niter_inner=1000, sigma=sigma)
        rec = np.dot(dict_0, X_mp[0])
        u_points = u[ind] / waves
        v_points = v[ind] / waves
        ax6.plot(np.sqrt(u_points ** 2 + v_points ** 2), rec[0:len(ind)], 'o', label = str(np.round(alpha, 3)), zorder=500, alpha=0.5, color=colorVal)
        ax7.plot(np.sqrt(u_points ** 2 + v_points ** 2), np.rad2deg(rec[len(ind):]), 'o', zorder=500, alpha=0.5, color=colorVal)

        a_tot = 0
        aa = 0
        for i in range(comp_cube.shape[0]):
            aa = comp_cube[i,:,:] * X_mp[0][i]
            a_tot = a_tot + aa
        pyfits.writeto('image_from_visibilities_pca3_lambda_'+str(np.round(m, 3))+'.fits', a_tot, overwrite=True)
        ii = ii + 1

    ax6.set_xlabel('Spatial Freq. [1/rad]')
    ax6.set_ylabel('Visibilities')
    ax7.set_xlabel('Spatial Freq. [1/rad]')
    ax6.set_ylabel('Phases')
    ax6.set_ylim([0, 1])
    ax7.set_ylim([-180, 180])
    ax6.legend()
    ax6.grid()
    ax7.grid()
    fig11.savefig('visibilities_pylops.png', bbox_inches='tight')

plt.show()
pdb.set_trace()
        

