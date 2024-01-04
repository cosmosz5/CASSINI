import os
import pdb
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.models import Sequential
from keras.layers import Dense
import keras.layers as kl
import keras.backend as kb
from keras.engine.topology import Layer
from keras.layers import Reshape, Dropout, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects
import astropy.io.fits as pyfits
import numpy as np
import argparse
import math
import oitools

class Atanh(Layer):

    def __init__(self, **kwargs):
        super(Atanh, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._amp = self.add_weight(name='amp',
                                    shape=(input_shape[1],),
                                    initializer='random_normal',
                                    trainable=True)
        self._slope = self.add_weight(name='slope',
                                 shape=(input_shape[1],),
                                 initializer='random_normal',
                                 trainable=True)
        super(Atanh, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self._amp * ((kb.exp(2*self._slope*x) -1) / (kb.exp(2*self._slope*x) + 1))

    def compute_output_shape(self, input_shape):
        return input_shape

class AReLU(Layer):

    def __init__(self, **kwargs):
        super(AReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._const = self.add_weight(name='const',
                                    shape=(1,),
                                    initializer='random_normal',
                                    trainable=True)

        super(AReLU, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return kb.maximum(self._const * x, x)

    def compute_output_shape(self, input_shape):
        return input_shape


# def atanh(x):
#     amplitude = Layer.add_weight(shape = x.shape[-1],
#                    initializer='zeros', trainable=True)
#     slope = Layer.add_weight(shape = x.get_shape[-1],
#                    initializer='zeros', trainable=True)
#     pdb.set_trace()
#     return amplitude * ((kb.exp(2*slope*x) -1) / (kb.exp(2*slope*x) + 1))

#def custom_activation(x):
#    const = kb.variable(value=0, dtype='float64')
#    return kb.maximum(const,x)

def myprint(s):
    with open('modelsummary.txt','w') as f:
        print(s, file=f)
    
def automap():
    #get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    #get_custom_objects().update({'atanh': Activation(atanh)})
    model = Sequential()
    model.add(Dense(32 * 32, use_bias = False, input_shape = (2040,), activation='sigmoid'))
    model.add(Dense(64 * 64, use_bias = False ))
    model.add(Atanh())
    model.add(Dense(64 * 64, use_bias=  False ))
    #model.add(Activation(atanh, name='SpecialActivation2'))
    model.add(Atanh())
    model.add(Reshape((64, 64, 1)))
    model.add(Conv2D(64, kernel_size=5, strides=(1, 1), padding='same', use_bias=  False))
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(Conv2D(64, kernel_size=5, strides=(1, 1), padding='same', use_bias=  False))
    model.add(PReLU())
    model.add(Dropout(0.1))
    model.add(UpSampling2D((1, 1)))
    model.add(Conv2D(64, kernel_size=7, strides=(1, 1), padding='same', use_bias=  False))
    model.add(Conv2D(1, kernel_size=64, strides=(1, 1), padding='same', use_bias=  False))
    model.add(AReLU())
    #model.summary()
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def train(BATCH_SIZE, EPOCH):
    load_data = np.load('bs64_flux.npz')
    train_images = load_data['arr_0']
    X_train = train_images.reshape(4397, 64, 64, 1)
    load_oidata = np.load('xnew_OBTot.npz')
    train_oidata = load_oidata['data'].T
    m = automap()
    #m_optim = Adam(0.00009)
    m_optim = Adam(0.009)
    m.compile(loss='mean_squared_error', metrics = ['mse'], optimizer=m_optim)
    #m.load_weights('model64_cos_acc_final')

    fig2, ax11 = plt.subplots(1, 1, figsize=(9, 6))
    for epoch in range(EPOCH):
        m_losses= np.zeros([int(X_train.shape[0]/BATCH_SIZE)])
        #epoch = epoch + 200
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            ### Define the input values from the FT of the images ****
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            oidata_batch = train_oidata[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            #### Generate images from the initial values
            m_loss = m.train_on_batch(oidata_batch, image_batch)
            print("batch %d model_loss : %e" % (index, m_loss[0]))

            if index % 10 == 0:
                generated_images = m.predict(oidata_batch, verbose=0)
                image = combine_images(generated_images)
                #image = image*127.5+127.5
                fig, ax1 = plt.subplots(1,1)
                ax1.imshow(image, cmap='jet', origin='lower')
                fig.savefig(str(epoch)+"_OBTot_"+str(index)+".png")
                pyfits.writeto(str(epoch)+"_OBTot_"+str(index)+".fits", image, overwrite=True)
                plt.close(fig)

            m_losses[index] = m_loss[1]
        if epoch % 50 == 0:
            m.save_weights('model64_OBTot_'+str(epoch), True)

        if epoch == 0:
            metrics = np.mean(m_losses)
        else:
            metrics = np.append(metrics, np.mean(m_losses))

        ax11.errorbar(epoch, np.mean(m_losses), yerr= np.std(m_losses), fmt='o', color='red')
        ax11.set_xlabel('Epoch')
        ax11.set_ylabel('Model MSE')
        fig2.savefig('losses64_OBTot.png', bbox_inches='tight')
        plt.close(fig2)
    np.savez('metrics_OBTot.npz', metrics = metrics)
    m.save_weights('model64_OBTot_final', True)
    return

def generate(oifilename):

    obs = oitools.extract_data(oifilename)

    generated_images = np.zeros([64, 64, 100])
    for i in range(100):

        vis2 = np.random.normal(obs['vis2'].reshape(-1), np.abs(obs['vis2_err'].reshape(-1)))
        t3phi = np.random.normal(np.deg2rad(obs['t3'].reshape(-1)), np.deg2rad(obs['t3_err'].reshape(-1)))
        t3phi = (t3phi + np.pi) % (2 * np.pi) - np.pi
        
        OBSERVABLES = np.append(vis2, t3phi)
        #OBSERVABLES = np.append(temp, t3phi_cos)
        OBSERVABLES = np.reshape(OBSERVABLES, (1, OBSERVABLES.shape[0]))
        #load_oidata = np.load('xnew_cos_sin.npz')
        #train_oidata = load_oidata['data'].T
        #pdb.set_trace()
        m = automap()
        m_optim = Adam(0.009)
        m.compile(loss='mse', metrics=['mse'], optimizer=m_optim)
        m.load_weights('model64_OBTot_final')
        m.trainable = False
        generated_image = m.predict(OBSERVABLES, verbose=1)
        #generated_image = m.predict(train_oidata[20,:].reshape(1,-1), verbose=1)
        generated_images[:,:,i] = np.squeeze(generated_image)

    pyfits.writeto('test.fits', np.mean(generated_images, axis=2), overwrite=True)

    
    vis, phase, t3phi = oitools.compute_obs(oifilename, np.mean(generated_images, axis=2), 10, 36, 84)
    fig2, (ax2, ax3) =plt.subplots(1,2, figsize=(12, 4))
    sz = obs['vis2'].shape[0]
    ax2.errorbar(obs['uv'], obs['vis2'].reshape(-1), yerr=np.abs(obs['vis2_err'].reshape(-1)), fmt='o', color='black', alpha=0.5)
    ax2.plot(obs['uv'], vis**2, 'or', ms = 3, zorder= 1000)
    ax3.errorbar(obs['uv_cp'], obs['t3'].reshape(-1), yerr=np.abs(obs['t3_err'].reshape(-1)), fmt='o', color='black', alpha=0.5)
    ax3.plot(obs['uv_cp'], t3phi, 'or', ms = 3, zorder= 1000)
    ax2.set_xlabel('Spatial Frequencies [1/rad]')
    ax2.set_ylabel('Squared visibilities')
    ax3.set_xlabel('Spatial Frequencies [1/rad]')
    ax3.set_ylabel('Closure Phases [deg.]')
    fig2.savefig("observables_OBTot.png")
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(np.mean(generated_images, axis=2), cmap='jet', origin='lower')
    fig.savefig("generated_image_OBTot.png")
    pyfits.writeto("generated_image_0BTot.fits", np.mean(generated_images, axis=2), overwrite=True)

    return

def validate(oifilename):

    obs = oitools.extract_data(oifilename)

    load_oidata = np.load('xnew_OBTot.npz')
    train_oidata = load_oidata['data'].T
    train_oidata = train_oidata[0:2000, :]
    #generated_images = np.zeros([2000, 64, 64])

    m = automap()
    m_optim = Adam(0.009)
    m.compile(loss='mse', metrics=['mse'], optimizer=m_optim)
    m.load_weights('model64_OBTot_final')
    m.trainable = False
    generated_images = m.predict(train_oidata, verbose=1)
    generated_images = np.squeeze(generated_images)
    
    pyfits.writeto("validate_generated_image_0BTot.fits", generated_images, overwrite=True)

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--oifilename", type=str)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, EPOCH=args.epoch)
    elif args.mode == "generate":
        generate(oifilename=args.oifilename)
    elif args.mode == "validate":
        validate(oifilename=args.oifilename)
    
