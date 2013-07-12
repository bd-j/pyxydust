##some file reading utilities
import numpy as np
import pyfits
import os

def load_image_cube(imnamelist = None,errnamelist = None,fudge_err= None, **kwargs):
    if imnamelist is None : raise ValueError('No image names specified!!!')

    hdr = pyfits.getheader(imnamelist[0])
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']
    data_mag = np.zeros([nx,ny,len(imnamelist)])
    data_magerr = np.zeros([nx,ny,len(imnamelist)])

    for i,imname in enumerate(imnamelist):
        im = pyfits.open(imname)
        data_mag[:,:,i] = 0-2.5*np.log10(im[0].data/3631.0)
        if os.path.isfile(errnamelist[i]) :
            err = pyfits.open(errnamelist[i])
            data_magerr[:,:,i] = 1.086*( err[0].data/im[0].data )
            err.close()
        else:
            data_magerr[:,:,i] = 1.086*(fudge_err[i]) #fudge errors when they don't exist
        im.close()
    
    return data_mag, data_magerr, hdr
