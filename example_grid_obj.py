#pyxydust: 
#  1) Fit each pixel in an image with DL07 dust models
#  2) Generate images of percentiles of the resulting
#     marginalized posterior probability for each model
#     parameter.  Also, plot some joint distributions for
#     selected pixels.

import numpy as np
import os, time
import pyxydust

############### USER INPUTS ###########

## user inputs are stored in a dictionary, here 'rp', which is passed to the fitter
## on initialization.  

rp = {'outname': 'results/test/NGC6822', 'dist': 0.490, 'ngrid': 5e4,
      'wave_min':30e4, 'wave_max': 1e7, #AA, range for determination of LDUST
      'percentiles' : np.array([0.025,0.5,0.975]) #output percentiles of the cdf for each parameter
      }

## these are the parameters to return as images in addition to LDUST and MDUST
rp['outparnames'] = ['UMIN', 'UMAX', 'UBAR', 'GAMMA','QPAH'] 
rp['return_residuals'] = True # False or True #return best fit residuals in each band

### Filter Info ###
## list the filters you want to use (in the same order
## as the images below).  These are based on k_correct names
rp['fnamelist'] = ['spitzer_mips_24','herschel_pacs_70','herschel_pacs_100',
             'herschel_pacs_160','herschel_spire_250']

### Image File Names ###
## images should be convolved to a common resolution, pixel matched,
## and in units of Jy/pixel. Otherwise modify utils.load_image_cube
## or add -2.5*log(conversion) to the magnitude arrays.
## If the error image can't be found the fudge values are used. The error
## images should include any systematics, etc.

## note that images of size 1 x N_obj can be used to fit the SEDs of an arbitrary
## set of objects, where each object is a 'pixel'
imnamelist = ['mips_24.6arcsec.Jypix','pacs_70.6arcsec.Jypix',
              'pacs_100.6arcsec.Jypix','pacs_160.6arcsec.Jypix',
              'spire_250.6arcsec.Jypix']
errnamelist = ['x','pacs_70.6arcsec.sig','pacs_100.6arcsec.sig',
               'pacs_160.6arcsec.sig','x']

impath = os.getenv('pyxydust')+'/imdata/older_withSPIRE/NGC6822' #not an empath, that would be cool tho
rp['imnamelist'] = ['{0}_conv250_{1}.fits'.format(impath,name) for name in imnamelist]
rp['errnamelist'] = ['{0}_{1}.fits'.format(impath,name) for name in errnamelist]
rp['fudge_err'] = [0.1,0.1,0.1,0.1,0.15] #magnitude errors to use when real error images not present

####### INITIALIZE AND RUN THE FITTER ##########

## Intitalize the grid-based fitter with the rp dictionary
fitter = pyxydust.PyxydustGrid(rp)
## read in the image cube given filenames in rp
fitter.load_data() 

## Change the definition of a good pixel.  Goodpix is a
## tuple of y and x indices of the good pixels
#fitter.goodpix = np.where(fitter.data_mag CONDITIONS & fitter.data_magerr CONDITONS)

## Change the allowed range of a given parameter from the default
#print(fitter.params.keys()) #show the available model input parameters
#fitter.params['QPAH']['min'] = some_number.  #make sure to use log10 if params['PAR']['type'] == 'log'
#fitter.params['GAMMA']['max'] =  #allow GAMMA to go up to one
#fitter.params['GAMMA'] = {'min':-4, 'max':0, 'type':'log'} #make the prior uniform in the log (from 1e-4 to 1)
fitter.params['UMAX']['min'] = 6 #require UMax = 1e6 always ('UMAX'['max'] is 1e6 also)
fitter.params['UMAX']['min'] = 5

#generate the model grid with the given prior distributions
#params can be a dictionary of input parameter descriptor dictionaries
fitter.initialize_grid(params = None)

## Set up for predicting the emission in any band
## Do this after running fitter.initialize_grid() but before
## running fitter.fit_image()
#import observate
#predfilt = observate.load_filters(your_prediction_filternamelist)
#prediction_sed, tmp1, tmp2 = fitter.dl07.generateSEDs(fitter.dustgrid.pars,pred_filt,wave_min = 8e4, wave_max = 1e7)
#for i in len(pred_filt) : fitter.dustgrid.add_par(prediction_sed[:,i],pred_filt[i].name())
#fitter.rp['outparnames']+= [pred_filt[i].name() for i in len(pred_filt)]

## Cry Havoc!  actually fit each pixel
fitter.fit_image()

## Force the fit of a given single pixel.  
#fitter.fit_pixel(ix,iy) 

## Change the output filename root
#fitter.rp['outname']=
## Change the parameters you want output.  Must be a subset
## of the original rp['outparnames'] plus 'LDUST' and 'MDUST'
#fitter.outparnames = ['LDUST','UBAR']
fitter.write_output()
