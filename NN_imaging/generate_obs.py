import numpy as np
import astropy.io.fits as pyfits
import oitools
import pdb

load_data = np.load('bs64_flux.npz')
train_images = load_data['arr_0']
X_train = train_images.reshape(4397, 64, 64, 1)

for i in range(X_train.shape[0]):
    image = np.reshape(X_train[i,:,:,:], (64,64))
    #flux = np.random.uniform(0.1, 1.0)
    #pyfits.writeto('bs_models64_flux/model_'+str(i)+'.fits', flux*image, overwrite=True)
    #pyfits.writeto('bs_models64/model_'+str(i)+'.fits', image, overwrite=True)
    vis, phase, t3phi = oitools.compute_obs('MERGED_IRS1W.oifits', image, 10, 36, 84)
    #data_temp = np.append((vis.reshape(-1))**2, np.sin(np.deg2rad(t3phi.reshape(-1))))
    #data = np.append(data_temp, np.cos(np.deg2rad(t3phi.reshape(-1))))
    data = np.append((vis.reshape(-1))**2, np.deg2rad(t3phi.reshape(-1)))
    print(len(vis.reshape(-1)))
    print(len(data))
    if i == 0:
        xnew = data * 1.0
    else:
        xnew = np.dstack((xnew, data))
    xnew2 = np.squeeze(xnew)
    #train_images[i,:,:] = flux * train_images[i,:,:]
#np.savez('bs64_flux.npz', train_images)
np.savez('xnew_OBTot.npz', data=xnew2)

pdb.set_trace()
