import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.io.fits as pyfits

oidata = pyfits.open('CALIB_SCI_uncalib_u_corr_jw01349006001_03103_00001_nis_calints.oifits')

waves = oidata['OI_WAVELENGTH'].data['EFF_WAVE']
ucoord = oidata['OI_VIS2'].data['UCOORD']
vcoord = oidata['OI_VIS2'].data['VCOORD']

fig, ax1 = plt.subplots(1,1)

ax1.plot(ucoord/waves, vcoord/waves, 'o')
ax1.plot(-ucoord/waves, -vcoord/waves, 'o')

ax1.set_title('JWST/SAM u-v coverage')
ax1.set_xlabel('Spatial Frequency [1/rad]')
ax1.set_ylabel('Spatial Frequency [1/rad]')
ax1.set_aspect('equal')
ax1.set_xlim([1.5e6, -1.5e6])
ax1.set_ylim([-1.5e6, 1.5e6])
plt.show()
fig.savefig('uv_coverage.png', bbox_inches='tight')
pdb.set_trace()
