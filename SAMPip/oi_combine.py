#my_new_list = np.core.defchararray.add(tab_dist['prefix'][i], files)
from readcol import *

cd = True  ### Must always be FALSE when combined for the reconstruction (use false for CDIFF)
if cd == True:
    import oi_merge as oi_merge
else:
    import oi_merge2 as oi_merges#
import importlib
importlib.reload(oi_merge)

data = 'calib_data_380.txt'
[files] = readcol(data, twod=False)
merged = oi_merge.oi_merge(files)
merged.write('COMB_JWST_SAM_WR137_F380.fits')
