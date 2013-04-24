#pyxydust: 
#  1) Fit each pixel in an image with DL07 dust models
#  2) Return images of percentiles of the resulting
#     marginalized posterior probability for each model
#     parameter.  Also, plot some joint distributions for
#     selected pixels.

#import numpy as np
import numpy as np
import os, time
import pyfits

import observate
import sedmodels
import datacube
import statutils

import matplotlib.pyplot as pl
#from dustplot import *

############### USER INPUTS ###########

outname = 'NGC6822'
ngrid = 5e4
percentiles = np.array([0.025,0.5,0.975]) #output percentiles

######## Filter info ##################
#list the filters you want to use (in the same order
#as the images below).  These are based on k_correct names

wave_min, wave_max = 15e4, 1e7 #AA, range for determination of L_TIR
fnamelist = ['spitzer_mips_24','herschel_pacs_70','herschel_pacs_100',
             'herschel_pacs_160','herschel_spire_250']
nfilt = len(fnamelist)

######## Image file names ##############
#images should be convolved to a common resolution, pixel matched,
#and in units of Jy/pixel. Otherwise modify utils.loadImageCube
#or add -2.5*log(conversion) to the magnitude arrays

dist = 0.490 #Mpc
imnamelist = ['mips_24.6arcsec.Jypix','pacs_70.6arcsec.Jypix',
              'pacs_100.6arcsec.Jypix','pacs_160.6arcsec.Jypix',
              'spire_250.6arcsec.Jypix']
errnamelist = ['x','pacs_70.6arcsec.sig','pacs_100.6arcsec.sig',
               'pacs_160.6arcsec.sig','x']
fudge_err = [0.1,0.1,0.1,0.1,0.15]

path = os.getenv('pyxydust')+'/imdata/NGC6822'
imnamelist = ['%s_conv250_%s.fits' % (path,name) for name in imnamelist]
errnamelist = ['%s_%s.fits' % (path,name) for name in errnamelist]

############### END USER INPUT ##########

############### MAIN ####################

######## read the images and errors ######
data_mag, data_magerr, header = datacube.loadImageCube(imnamelist,errnamelist,fudge_err)
dm = 5.0*np.log10(dist)+25
data_mag = np.where(data_mag != 0.0, data_mag-dm, 0.0)
nx, ny = data_mag.shape[0], data_mag.shape[1]

#### Load the filters and the DL07 model grid ####
filterlist = observate.loadFilters(fnamelist)
dl07 = sedmodels.DraineLi()

#### Set parameter ranges for priors ####
par_names = ['UMIN','UMAX','GAMMA','QPAH']
par_range = np.array([[0.1,25],dl07.parRange(['UMAX'],inds=dl07.pdr_inds)[0],
                      [0.00,0.5],
                      dl07.parRange(['QPAH'])[0]])
par_range[1,:] = np.log10(par_range[1,:]) #make the UMAX prior uniform in log.
npar = len(par_names)

######## initialize grid ################
theta = np.zeros([ngrid,npar])
for j in xrange(npar) :
    theta[:,j] = np.random.uniform(par_range[j,0],par_range[j,1],ngrid)
theta[:,1]=10**theta[:,1] #deal with uniform log prior in UMAX

start = time.time()
dustgrid = sedmodels.ModelGrid()
dustgrid.setPars(theta,par_names)
dustgrid.generateSEDs(dl07,filterlist,wave_min=wave_min,wave_max=wave_max)
duration=time.time()-start
print('Model Grid built in ',duration,' seconds')

ubar = dl07.ubar(dustgrid.pars['UMIN'],dustgrid.pars['UMAX'],dustgrid.pars['GAMMA'])

######## set up output ###################
outparnames=['LDUST','MDUST','UBAR']+par_names
noutpar=len(outparnames) #the ndim + mdust, ldust, ubar
parval = np.zeros([nx,ny,noutpar,4]) 
delta_best = np.zeros([nx,ny])-99
max_lnprob = np.zeros([nx,ny])-99

######## Loop over pixels ################
#prefilter bad pixels
g = np.where(data_mag < 0,1,0)
g = np.where(np.isfinite(data_mag),g,0)
goodpix = np.where(g.sum(2) == len(imnamelist)) #restrict to pixels with at least 4 bands

start = time.time()

for ipix in xrange(goodpix[0].shape[0]):

    iy, ix  = goodpix[0][ipix], goodpix[1][ipix]
    obs = data_mag[iy,ix,:]
    err = data_magerr[iy,ix,:]
    mask = np.where(np.logical_and( (obs < 0), np.isfinite(obs) ), 1, 0)

    lnprob , ltir, dustm, delta_mag = statutils.lnprob_grid(dustgrid, obs, err, mask)
    ind_isnum=np.isfinite(lnprob)
    lnprob_isnum=lnprob[ind_isnum]

    #output
    max_lnprob[iy,ix] = np.max(lnprob_isnum)
    allpars = np.vstack([ltir, dustm, ubar, dustgrid.pars['UMIN'], dustgrid.pars['UMAX'],
                       dustgrid.pars['GAMMA'], dustgrid.pars['QPAH']])

    #get the percentiles of the 1d marginalized posterior distribution
    for ipar in xrange(len(outparnames)):
        par = np.squeeze(allpars[ipar,:])
        par= par[ind_isnum]
        order = np.argsort(par)
        cdf = np.cumsum(np.exp(lnprob_isnum[order])) / np.sum(np.exp(lnprob_isnum))
        ind_ptiles= np.searchsorted(cdf,percentiles) 
        ind_max=np.argmax(lnprob_isnum)
        parval[iy,ix,ipar,:-1] = (par[order[ind_ptiles-1]] +
                                  par[order[ind_ptiles]])/2.0 #almost right.  should interpolate between ind and ind-1
        parval[iy,ix,ipar,-1] = par[ind_max]

    if ipix%100 == 0 :
        #    do stuff for every 100th pixel
        pass

#write out the parval images
for i in xrange(len(outparnames)):
    #    header.set('BUNIT',unit[i])
    for j in xrange(3):
        outfile= 'results/%s_%s_p%5.3f.fits' % (outname,outparnames[i],percentiles[j]) 
        pyfits.writeto(outfile,parval[:,:,i,j],header=header,clobber=True)
    outfile= 'results/%s_%s_bestfit.fits' % (outname,outparnames[i]) 
    pyfits.writeto(outfile,parval[:,:,i,-1],header=header,clobber=True)
outfile= 'results/%s_CHIBEST.fits' % outname
pyfits.writeto(outfile,max_lnprob*(-2),header=header,clobber=True)
duration =  time.time()-start
print('Done in',duration,' seconds' )

print(delta_mag.shape)
