import numpy as np
import matplotlib.pyplot as plt
import pdb
import astropy.io.fits as pyfits

oidata = pyfits.open('CALIB_SIM_DATA_uncalib_t_disk_small2_0__PSF_MASK_NRM_F430M_x11_0.82_ref__00.oifits')

waves = oidata['OI_WAVELENGTH'].data['EFF_WAVE']
ucoord = oidata['OI_VIS2'].data['UCOORD']
vcoord = oidata['OI_VIS2'].data['VCOORD']

fig, ax1 = plt.subplots(1,1)

ax1.plot(ucoord, vcoord, 'o')
ax1.plot(-ucoord, -vcoord, 'o')

ax1.set_title('JWST/SAM u-v coverage')
ax1.set_xlabel('Meters')
ax1.set_ylabel('Meters')
plt.show()
fig.savefig('uv_coverage.png', bbox_tight='true')
pdb.set_trace()
