import candid
import numpy
import matplotlib.pyplot as plt
import pdb
import pylab


if __name__ == '__main__':
    candid.CONFIG['Ncores'] = 16
    candid.CONFIG['long exec warning'] = 20063

    o = candid.Open('COMB_JWST_SAM_abdor.fits')
    o.observables = ['v2', 'cp']
    fig = plt.figure(1)
    o.fitMap(fig=1, rmin = 20, rmax = 300.0, step = 20.0, doNotFit=['diam*'])#, addParam={'fres': 0.1})
    aa = plt.figure(1)
    aa.savefig('test_candid.png')
    pdb.set_trace()

    #o.fitBoot(fig=2, doNotFit=['diam*'])

