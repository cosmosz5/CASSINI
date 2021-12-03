from readcol import *

cd = True 
if cd == True:
    import oi_merge as oi_merge
else:
    import oi_merge2 as oi_merges#
import importlib
importlib.reload(oi_merge)

data = 'combine_data_tot.txt'
[files] = readcol(data, twod=False)
merged = oi_merge.oi_merge(files)
merged.write('COMB_JWST_SAM_tot.fits')
