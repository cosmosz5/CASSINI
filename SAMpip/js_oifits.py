import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table
import datetime
import pdb

__version__ ='1.0v'

class OI_ARRAY:

    def __init__(self, oirevn, arrname, frame, arrayx, arrayy, arrayz, tel_name, sta_name, sta_index, diameter, staxyz):
        self.oirevn=oirevn
        self.arrname=arrname
        self.frame=frame
        self.arrayx=arrayx
        self.arrayy=arrayy
        self.arrayz=arrayz
        self.tel_name=tel_name
        self.sta_name=sta_name
        self.sta_index=sta_index
        self.diameter=diameter
        self.staxyz=staxyz

class OI_TARGET:

    def __init__(self, oirevn, target_id, target, raep0, decep0, equinox, ra_err, dec_err, sysvel, veltyp, veldef,\
                 pmra, pmdec, pmra_err, pmdec_err, parallax, para_err, spectyp):
        self.oirevn=oirevn
        self.target_id=target_id
        self.target=target
        self.raep0=raep0
        self.decep0=decep0
        self.equinox=equinox
        self.ra_err=ra_err
        self.dec_err=dec_err
        self.sysvel=sysvel
        self.veltyp=veltyp
        self.veldef=veldef
        self.pmra=pmra
        self.pmdec=pmdec
        self.pmra_err=pmra_err
        self.pmdec_err=pmdec
        self.parallax=parallax
        self.para_err=para_err
        self.spectyp=spectyp

class OI_WAVELENGTH:

    def __init__(self, oirevn, insname, eff_wave, eff_band):
        self.oirevn=oirevn
        self.insname=insname
        self.eff_wave=eff_wave
        self.eff_band=eff_band

class OI_VIS:

    def __init__(self, oirevn, dateobs, arrname, insname, target_id, time, mjd, int_time, visamp, visamperr, visphi, visphierr, \
                 ucoord, vcoord, sta_index, flag):
        self.oirevn=oirevn
        self.dateobs=dateobs
        self.arrname=arrname
        self.insname=insname
        self.target_id=target_id
        self.time=time
        self.mjd=mjd
        self.int_time=int_time
        self.visamp=visamp
        self.visamperr=visamperr
        self.visphi=visphi
        self.visphierr=visphierr
        self.ucoord=ucoord
        self.vcoord=vcoord
        self.sta_index=sta_index
        self.flag=flag

class OI_VIS2:

    def __init__(self, oirevn, dateobs, arrname, insname, target_id, time, mjd, int_time, vis2data, vis2err, ucoord, vcoord, \
                 sta_index, flag):
        self.oirevn=oirevn
        self.dateobs=dateobs
        self.arrname=arrname
        self.insname=insname
        self.target_id=target_id
        self.time=time
        self.mjd=mjd
        self.int_time=int_time
        self.vis2data=vis2data
        self.vis2err=vis2err
        self.ucoord=ucoord
        self.vcoord=vcoord
        self.sta_index=sta_index
        self.flag=flag

class OI_T3:

    def __init__(self, oirevn, dateobs, arrname, insname, target_id, time, mjd, int_time, t3amp, t3amperr, t3phi, t3phierr, \
                 u1coord, v1coord, u2coord, v2coord, sta_index, flag):
        self.oirevn=oirevn
        self.dateobs=dateobs
        self.arrname=arrname
        self.insname=insname
        self.target_id=target_id
        self.time=time
        self.mjd=mjd
        self.int_time=int_time
        self.t3amp=t3amp
        self.t3amperr=t3amperr
        self.t3phi=t3phi
        self.t3phierr=t3phierr
        self.u1coord=u1coord
        self.v1coord=v1coord
        self.u2coord=u2coord
        self.v2coord=v2coord
        self.sta_index=sta_index
        self.flag=flag

class OI_FLUX:

    def __init__(self, oirevn, dateobs, arrname, insname, calstat, target_id, mjd, int_time, flux, fluxerr, sta_index, flag):
        self.oirevn=oirevn
        self.dateobs=dateobs
        self.arrname=arrname
        self.insname=insname
        self.calstat=calstat
        self.target_id=target_id
        self.mjd=mjd
        self.int_time=int_time
        self.flux=flux
        self.fluxerr=fluxerr
        self.sta_index=sta_index
        self.flag=flag

class oifits:

    def __init__(self):
        self.wavelength = {}
        self.target = np.empty(0)
        self.array = {}
        self.flux = np.empty(0)
        self.vis = np.empty(0)
        self.vis2 = np.empty(0)
        self.t3 = np.empty(0)

    def write(self, filename, head=False):

        print ('Writing data in a new OIFITS file...')
        hdulist = pyfits.HDUList()
        if head != False:
            prihdr = pyfits.Header.copy(head)
            hdu= pyfits.PrimaryHDU(header=prihdr)
            hdulist.append(hdu)
        else:
            hdu= pyfits.PrimaryHDU()
        #pdb.set_trace()
        #hdu.header['DATE']=(datetime.datetime.now().strftime(format='%F'), 'Creation date')
        #hdu.header['CREATED']=('Written by JS_OIFITS Python module version %s'%__version__)
        if self.array:
            print ('Writing OI_ARRAY table...')
            data= self.array
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='TEL_NAME', format='16A', array=data.tel_name),\
                                                   pyfits.Column(name='STA_NAME', format='16A', array=data.sta_name),\
                                                   pyfits.Column(name='STA_INDEX', format='1I', array=data.sta_index),\
                                                   pyfits.Column(name='DIAMETER', unit='METERS', format='1E', array=data.diameter),\
                                                   pyfits.Column(name='STAXYZ', unit='METERS', format='3D', array=data.staxyz),\
                                                   )))
            hdu.header['EXTNAME']=('OI_ARRAY')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdu.header['ARRNAME']=(data.arrname, 'Array name, for cross-referencing')
            hdu.header['FRAME']=(data.frame, 'Coordinate frame')
            hdu.header['ARRAYX']=(data.arrayx, 'ARRAY center x coordinate (m)')
            hdu.header['ARRAYY']=(data.arrayy, 'ARRAY center y coordinate (m)')
            hdu.header['ARRAYZ']=(data.arrayz, 'ARRAY center z coordinate (m)')
            hdulist.append(hdu)

        if self.target:
            print ('Writing OI_TARGET table...')
            data= self.target
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='TARGET_ID', format='1I', array=data.target_id),\
                                                   pyfits.Column(name='TARGET', format='16A', array=data.target),\
                                                   pyfits.Column(name='RAEP0', unit='DEGREES', format='1D', array=data.raep0),\
                                                   pyfits.Column(name='DECEP0', unit='DEGREES', format='1D', array=data.decep0),\
                                                   pyfits.Column(name='EQUINOX', unit='YEARS', format='1E', array=data.equinox),\
                                                   pyfits.Column(name='RA_ERR', unit='DEGREES', format='1D', array=data.ra_err),\
                                                   pyfits.Column(name='DEC_ERR', unit='DEGREES', format='1D', array=data.dec_err),\
                                                   pyfits.Column(name='SYSVEL', unit='M/S', format='1D', array=data.sysvel),\
                                                   pyfits.Column(name='VELTYP', format='8A', array=data.veltyp),\
                                                   pyfits.Column(name='VELDEF', format='8A', array=data.veldef),\
                                                   pyfits.Column(name='PMRA', unit='DEG/YR', format='1D', array=data.pmra),\
                                                   pyfits.Column(name='PMDEC', unit='DEG/YR', format='1D', array=data.pmdec),\
                                                   pyfits.Column(name='PMRA_ERR', unit='DEG/YR', format='1D', array=data.pmra_err),\
                                                   pyfits.Column(name='PMDEC_ERR', unit='DEG/YR', format='1D', array=data.pmdec_err),\
                                                   pyfits.Column(name='PARALLAX', unit='DEGREES', format='1E', array=data.parallax),\
                                                   pyfits.Column(name='PARA_ERR', unit='DEGREES', format='1E', array=data.para_err),\
                                                   pyfits.Column(name='SPECTYP', format='16A', array=data.spectyp),\
                                                   )))
            hdu.header['EXTNAME']=('OI_TARGET')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdulist.append(hdu)

        if self.wavelength:
            print ('Writing OI_WAVELENGTH table...')
            data = self.wavelength
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='EFF_WAVE', unit= 'METERS', format='1E', array=data.eff_wave),\
                                                   pyfits.Column(name='EFF_BAND', unit='METERS',  format='1E', array=data.eff_band),\
                                                   )))
            hdu.header['EXTNAME']=('OI_WAVELENGTH')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdu.header['INSNAME']=(data.insname, 'Name of detector, for cross-referencing')
            hdulist.append(hdu)

        if self.vis:
            print ('Writing OI_VIS table...')
            data= self.vis
            nwave = len((self.wavelength).eff_wave)
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='TARGET_ID', format='1I', array=data.target_id),\
                                                   pyfits.Column(name='TIME', unit='SECONDS', format='1D', array=data.time),\
                                                   pyfits.Column(name='MJD', unit='DAY', format='1D', array=data.mjd),\
                                                   pyfits.Column(name='INT_TIME', unit='SECONDS', format='1D', array=data.int_time),\
                                                   pyfits.Column(name='VISAMP', format='%dD'%nwave, array=data.visamp),\
                                                   pyfits.Column(name='VISAMPERR', format='%dD'%nwave, array=data.visamperr),\
                                                   pyfits.Column(name='VISPHI', unit='DEGREES', format='%dD'%nwave, array=data.visphi),\
                                                   pyfits.Column(name='VISPHIERR', unit='DEGREES', format='%dD'%nwave, array=data.visphierr),\
                                                   pyfits.Column(name='UCOORD', unit='METERS', format='1D', array=data.ucoord),\
                                                   pyfits.Column(name='VCOORD', unit='METERS', format='1D', array=data.vcoord),\
                                                   pyfits.Column(name='STA_INDEX', format='2I', array=data.sta_index),\
                                                   pyfits.Column(name='FLAG', format='%dL'%nwave, array=data.flag),\
                                                   )))
            hdu.header['EXTNAME']=('OI_VIS')
            hdu.header['DATE-OBS']=(data.dateobs, 'UTC start date of observations')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdu.header['ARRNAME']=(data.arrname, 'Name of the corresponding array')
            hdu.header['INSNAME']=(data.insname, 'Name of the corresponding detector')
            hdulist.append(hdu)

        if self.vis2:
            print ('Writing OI_VIS2 table...')
            data= self.vis2
            nwave = len((self.wavelength).eff_wave)
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='TARGET_ID', format='1I', array=data.target_id),\
                                                   pyfits.Column(name='TIME', unit='SECONDS', format='1D', array=data.time),\
                                                   pyfits.Column(name='MJD', unit='DAY', format='1D', array=data.mjd),\
                                                   pyfits.Column(name='INT_TIME', unit='SECONDS', format='1D', array=data.int_time),\
                                                   pyfits.Column(name='VIS2DATA', format='%dD'%nwave, array=data.vis2data),\
                                                   pyfits.Column(name='VIS2ERR', format='%dD'%nwave, array=data.vis2err),\
                                                   pyfits.Column(name='UCOORD', unit='METERS', format='1D', array=data.ucoord),\
                                                   pyfits.Column(name='VCOORD', unit='METERS', format='1D', array=data.vcoord),\
                                                   pyfits.Column(name='STA_INDEX', format='2I', array=data.sta_index),\
                                                   pyfits.Column(name='FLAG', format='%dL'%nwave, array=data.flag),\
                                                   )))
            hdu.header['EXTNAME']=('OI_VIS2')
            hdu.header['DATE-OBS']=(data.dateobs, 'UTC start date of observations')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdu.header['ARRNAME']=(data.arrname, 'Name of the corresponding array')
            hdu.header['INSNAME']=(data.insname, 'Name of the corresponding detector')
            hdulist.append(hdu)

        if self.t3:
            print ('Writing OI_T3 table...')
            data=self.t3
            nwave = len((self.wavelength).eff_wave)
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='TARGET_ID', format='1I', array=data.target_id),\
                                                   pyfits.Column(name='TIME', unit='SECONDS', format='1D', array=data.time),\
                                                   pyfits.Column(name='MJD', unit='DAY', format='1D', array=data.mjd),\
                                                   pyfits.Column(name='INT_TIME', unit='SECONDS', format='1D', array=data.int_time),\
                                                   pyfits.Column(name='T3AMP', format='%dD'%nwave, array=data.t3amp),\
                                                   pyfits.Column(name='T3AMPERR', format='%dD'%nwave, array=data.t3amperr),\
                                                   pyfits.Column(name='T3PHI', unit='DEGREES', format='%dD'%nwave, array=data.t3phi),\
                                                   pyfits.Column(name='T3PHIERR', unit='DEGREES', format='%dD'%nwave, array=data.t3phierr),\
                                                   pyfits.Column(name='U1COORD', unit='METERS', format='1D', array=data.u1coord),\
                                                   pyfits.Column(name='V1COORD', unit='METERS', format='1D', array=data.v1coord),\
                                                   pyfits.Column(name='U2COORD', unit='METERS', format='1D', array=data.u2coord),\
                                                   pyfits.Column(name='V2COORD', unit='METERS', format='1D', array=data.v2coord),\
                                                   pyfits.Column(name='STA_INDEX', format='3I', array=data.sta_index),\
                                                   pyfits.Column(name='FLAG', format='%dL'%nwave, array=data.flag),\
                                                   )))
            hdu.header['EXTNAME']=('OI_T3')
            hdu.header['DATE-OBS']=(data.dateobs, 'UTC start date of observations')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdu.header['ARRNAME']=(data.arrname, 'Name of the corresponding array')
            hdu.header['INSNAME']=(data.insname, 'Name of the corresponding detector')
            hdulist.append(hdu)
            #pdb.set_trace()
            
        if self.flux:
            print ('Writing OI_FLUX table...')
            data= self.flux
            nwave = len((self.wavelength).eff_wave)
            hdu = pyfits.BinTableHDU.from_columns(pyfits.ColDefs((pyfits.Column(name='TARGET_ID', format='1I', array=data.target_id),\
                                                   pyfits.Column(name='MJD', unit='DAY', format='1D', array=data.mjd),\
                                                   pyfits.Column(name='INT_TIME', unit='SECONDS', format='1D', array=data.int_time),\
                                                   pyfits.Column(name='FLUX', format='%dD'%nwave, array=data.flux),\
                                                   pyfits.Column(name='FLUXERR', format='%dD'%nwave, array=data.fluxerr),\
                                                   pyfits.Column(name='STA_INDEX', format='1I', array=data.sta_index),\
                                                   pyfits.Column(name='FLAG', format='%dL'%nwave, array=data.flag),\
                                                   )))
            hdu.header['EXTNAME']=('OI_FLUX')
            hdu.header['DATE-OBS']=(data.dateobs, 'UTC start date of observations')
            hdu.header['OI_REVN']=(data.oirevn, 'Revision number of the table definition')
            hdu.header['ARRNAME']=(data.arrname, 'Name of the corresponding array')
            hdu.header['INSNAME']=(data.insname, 'Name of the corresponding detector')
            hdu.header['CALSTAT']=(data.calstat, 'C: Spectrum calibrated, U: uncalibrated')
            hdulist.append(hdu)
            
        hdulist.writeto(filename, overwrite=True)

def open(filename):
    newobj = oifits()
    hdulist=pyfits.open(filename)
    #print hdulist.info()
    for tables in hdulist:
        if tables.name == 'OI_ARRAY':
            print ('Reading OI_ARRAY table...')
            header=tables.header
            data=tables.data
            if newobj.array == None: newobj.array = {}
            oirevn=header['OI_REVN']
            arrname=header['ARRNAME']
            frame=header['FRAME']
            arrayx=header['ARRAYX']
            arrayy=header['ARRAYY']
            arrayz=header['ARRAYZ']
            tel_name=data['TEL_NAME']
            sta_name=data['STA_NAME']
            sta_index=data['STA_INDEX']
            diameter=data['DIAMETER']
            staxyz=data['STAXYZ']
            newobj.array = OI_ARRAY(oirevn, arrname, frame, arrayx, arrayy, arrayz, tel_name, \
                    sta_name, sta_index, diameter, staxyz)

        if tables.name == 'OI_TARGET':
            print ('Reading OI_TARGET table...')
            header=tables.header
            data=tables.data
            oirevn=header['OI_REVN']
            newobj.target=OI_TARGET(oirevn, data['TARGET_ID'], data['TARGET'], data['RAEP0'], data['DECEP0'], data['EQUINOX'], data['RA_ERR'], data['DEC_ERR'],\
                                    data['SYSVEL'], data['VELTYP'], data['VELDEF'], data['PMRA'], data['PMDEC'], data['PMRA_ERR'], data['PMDEC_ERR'], data['PARALLAX'], data['PARA_ERR'], data['SPECTYP'])

        if tables.name == 'OI_WAVELENGTH':
            print ('Reading OI_WAVELENGTH table...')
            header=tables.header
            data=tables.data
            if newobj.wavelength == None: newobj.wavelength = {}
            oirevn=header['OI_REVN']
            arrname=header['INSNAME']
            eff_wave=data['EFF_WAVE']
            eff_band=data['EFF_BAND']
            newobj.wavelength=OI_WAVELENGTH(oirevn, arrname, eff_wave, eff_band)

        if tables.name == 'OI_FLUX':
            print ('Reading OI_FLUX table...')
            header=tables.header
            data=tables.data
            oirevn= 1 #header['OI_REVN']
            date_obs=header['DATE-OBS']
            arrname=header['ARRNAME']
            insname=header['INSNAME']
            calstat=header['CALSTAT']
            newobj.flux=OI_FLUX(oirevn, date_obs, arrname, insname, calstat, data['TARGET_ID'], data['MJD'], data['INT_TIME'], data['FLUX'], data['FLUXERR'], data['STA_INDEX'], data['FLAG'])

        if tables.name == 'OI_VIS':
            print ('Reading OI_VIS table...')
            header=tables.header
            data=tables.data
            oirevn=header['OI_REVN']
            date_obs=header['DATE-OBS']
            arrname=header['ARRNAME']
            insname=header['INSNAME']
            newobj.vis=OI_VIS(oirevn, date_obs, arrname, insname, data['TARGET_ID'], data['TIME'], data['MJD'], data['INT_TIME'], data['VISAMP'], data['VISAMPERR'], data['VISPHI'], data['VISPHIERR'], data['UCOORD'], \
                               data['VCOORD'], data['STA_INDEX'], data['FLAG'])

        if tables.name == 'OI_VIS2':
            print ('Reading OI_VIS2 table...')
            header=tables.header
            data=tables.data
            oirevn=header['OI_REVN']
            date_obs=header['DATE-OBS']
            arrname=header['ARRNAME']
            insname=header['INSNAME']
            newobj.vis2=OI_VIS2(oirevn, date_obs, arrname, insname, data['TARGET_ID'], data['TIME'], data['MJD'], data['INT_TIME'], data['VIS2DATA'], data['VIS2ERR'], data['UCOORD'], data['VCOORD'], data['STA_INDEX'], \
                                data['FLAG'])

        if tables.name == 'OI_T3':
            print ('Reading OI_T3 table...')
            header=tables.header
            data=tables.data
            oirevn=header['OI_REVN']
            date_obs=header['DATE-OBS']
            arrname=header['ARRNAME']
            insname=header['INSNAME']
            newobj.t3=OI_T3(oirevn, date_obs, arrname, insname, data['TARGET_ID'], data['TIME'], data['MJD'], data['INT_TIME'], data['T3AMP'], data['T3AMPERR'], data['T3PHI'], data['T3PHIERR'], data['U1COORD'], \
                                data['V1COORD'], data['U2COORD'], data['V2COORD'], data['STA_INDEX'], data['FLAG'])
            #pdb.set_trace()
    return newobj


#data='2004contest1.oifits'
#datos=open(data)

#Sving the data...
#new=oifits()
#new.array=datos.array
#new.target=datos.target
#new.wavelength=datos.wavelength
#new.vis2=datos.vis2
#new.t3=datos.t3
#new.write('Valemadres.fits')
