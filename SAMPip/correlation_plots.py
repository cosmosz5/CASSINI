import pygtc
import numpy as np
import matplotlib.pyplot as plt
import pdb
from readcol import *
import matplotlib as mpl

fig, axes = plt.subplots(1,2, figsize=(10, 4))

data = np.load('corr_jw01093016001_03104_00001_nis_calints.npz')
data2 = np.load('corr_jw01093015001_03104_00001_nis_calints.npz')

v2_cal1 = data['V2']
v2_cal2 = data2['V2']
cp_cal1 = data['CP']
cp_cal2 = data2['CP']

v2_cal1mean, v2_cal1std, v2_cal1cov = np.mean(v2_cal1,axis=0), np.std(v2_cal1,axis=0), np.cov(v2_cal1.T)
cp_cal1mean, cp_cal1std, cp_cal1cov = np.mean(cp_cal1,axis=0), np.std(cp_cal1, axis=0), np.cov(cp_cal1.T)

v2_cal2mean, v2_cal2std, v2_cal2cov = np.mean(v2_cal2,axis=0), np.std(v2_cal2,axis=0), np.cov(v2_cal2.T)
cp_cal2mean, cp_cal2std, cp_cal2cov = np.mean(cp_cal2,axis=0), np.std(cp_cal2, axis=0), np.cov(cp_cal2.T)

v2_cov2, cp_cov2 = np.cov((v2_cal2).T), np.cov((cp_cal2).T)
v2_cov1, cp_cov1 = np.cov((v2_cal1).T), np.cov((cp_cal1).T)
v2_cov = v2_cov2/v2_cov1
cp_cov = cp_cov2/cp_cov1

vmin_v2, vmax_v2 = np.min(v2_cov),np.max(v2_cov)
vmin_cp, vmax_cp = np.min(cp_cov),np.max(cp_cov)
vmin_cp, vmax_cp = -np.max([np.abs(vmin_cp),np.abs(vmax_cp)]), np.max([np.abs(vmin_cp),np.abs(vmax_cp)])

## imshow v2
img_v2_cal2cov = axes[0].imshow(v2_cov,vmin=-5,vmax=5)
ax = axes[0]
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(img_v2_cal2cov,cax=cax)


img_cp_cal2cov = axes[1].imshow(cp_cov,vmin=-5, vmax=5, cmap=mpl.cm.seismic_r)
ax = axes[1]
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(img_cp_cal2cov,cax=cax)


axes[0].set_title('SAMPip V2 Covariance',fontsize=16)
axes[1].set_title('SAMPip CP Covariance',fontsize=16)

plt.show()

pdb.set_trace()

names = ['BL1','BL2','BL3','BL4','BL5','BL6','BL7','BL8','BL9','BL10',\
                 'BL111','BL12','BL13','BL14','BL15','BL16','BL17','BL18',\
                 'BL19','BL20','BL21']
chainLabels = ["Calibrator_exp1", "Calibrator_exp2"]
GTC = pygtc.plotGTC(chains=[data['V2'].T, data2['V2'].T])
GTC.suptitle('V2_'+filename[:-5])
GTC.savefig('V2_'+filename[:-5]+'.pdf', bbox_inches='tight')

names = ['CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CP7', 'CP8', 'CP9', 'CP10', \
             'CP11', 'CP12', 'CP13', 'CP14', 'CP15', 'CP16', 'CP17', 'CP18', 'CP19', 'CP20',\
             'CP21', 'CP22', 'CP23', 'CP24', 'CP25', 'CP26', 'CP27', 'CP28', 'CP29', 'CP30',\
             'CP31', 'CP32', 'CP33', 'CP34', 'CP35']

chainLabels = ["Calibrator_exp1", "Calibrator_exp2"]
GTC = pygtc.plotGTC(chains=[data['CP'].T, data2['CP'].T], paramNames=names, chainLabels=chainLabels)
GTC.suptitle('CP_'+filename[:-5])
GTC.savefig('CP_'+filename[:-5]+'.pdf', bbox_inches='tight')

plt.close('all')
