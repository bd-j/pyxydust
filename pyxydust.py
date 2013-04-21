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
import dustmodel
import modelgrid
import utils

import matplotlib.pyplot as pl
#from dustplot import *

############### USER INPUTS ###########

##### Filter info 
#list the filters you want to use (in the same order
#as the images below).  These are based on k_correct

wave_min, wave_max = 15e4, 1e7 #AA, range for determination of L_TIR
fnamelist = ['spitzer_mips_24','herschel_pacs_70','herschel_pacs_100',
             'herschel_pacs_160','herschel_spire_250']

##### Image file names
#images should be convolved to a common resolution, pixel matched,
#and in units of Jy/pixel. Otherwise modify utils.loadImageCube
#or simply add -2.5*log(conversion) to the magnitude arrays

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

#####
##### function to obtain likelihood ####
#####
#def lnprob(theta,obs_maggies,obs_ivar,mask):

def lnprob_grid(grid, obs, err, mask):
    #linearize fluxes.  
    inds = np.where(mask > 0)
    mod_maggies = 10**(0-grid.sed[...,inds]/2.5)
    obs_maggies = 10**(0-obs[inds]/2.5)
    obs_ivar = (obs_maggies*err[inds]/1.086)**(-2)

    #best scale for these parameters
    mdust = np.squeeze(( (mod_maggies*obs_maggies*obs_ivar).sum(axis=-1) ) /
                       ( ( (mod_maggies**2.0)*obs_ivar ).sum(axis=-1) ))
    lbol = grid.lbol*mdust
        
    #probability with dimensional juggling to get the broadcasting right
    chi2 =  (( (mdust*mod_maggies.T).T - obs_maggies)**2)*obs_ivar
    #print(ascale,chi2.sum())
    lnprob = -0.5*chi2.sum(axis=-1)
    #delta_mag = ( sed-2.5*np.log10(ascale) ) - obs
        
    return lnprob, lbol, mdust

############### MAIN ####################

##### Load the filters and the DL07 model grid ########
filterlist = observate.loadFilters(fnamelist)
dl07 = dustmodel.DraineLi()

##### Set parameter ranges for priors #########
par_names = ['UMIN','UMAX','GAMMA','QPAH']
par_range = np.array([dl07.parRange('UMIN'),
                      (100.0,dl07.parRange('UMAX')[1]),
                      (0.00,0.5),
                      dl07.parRange('QPAH')]
    )
#par_range[0,:] = np.log10(par_range[0,:])
par_range[1,:] = np.log10(par_range[1,:])
npar = len(par_names)

###### initialize grid #############
ngrid=1e3
theta = np.zeros([ngrid,npar])
for j in xrange(npar) :
    theta[:,j] = np.random.uniform(par_range[j,0],par_range[j,1],ngrid)

dustgrid=modelgrid.ModelGrid()
dustgrid.setParameters(theta,par_names)
dustgrid.generateSEDs(dl07,filterlist,wave_min=15e4,wave_max=1e7)

#raise ValueError('debug')

##### read the images and errors

data_mag, data_magerr, header = utils.loadImageCube(imnamelist,errnamelist,fudge_err)
dm = 5.0*np.log10(dist)+25
data_mag = np.where(data_mag != 0.0, data_mag-dm, 0.0)
nx, ny = data_mag.shape[0], data_mag.shape[1]

##### set up output
percentiles = np.array([0.16,0.5,0.84]) #output percentiles
noutpar=npar+3 #the ndim + mdust, ldust, ubar
parval = np.zeros([nx,ny,noutpar,3]) 
delta_best = np.zeros([nx,ny])-99
max_lnprob = np.zeros([nx,ny])-99

####### Loop over pixels #######
#should change the loop to prefilter bad pixels
g = np.where(data_mag < 0,1,0)
g = np.where(np.isfinite(data_mag),g,0)
goodpix = np.where(g.sum(2) == len(imnamelist)) #restrict to pixels with at least 5 bands

for ipix in xrange(goodpix[0].shape[0]):
    start = time.time()
    iy, ix  = goodpix[0][ipix], goodpix[1][ipix]

    obs = data_mag[iy,ix,:]
    err = data_magerr[iy,ix,:]
    mask = np.where(np.logical_and( (obs < 0), np.isfinite(obs) ), 1, 0)

    lnprob , ltir, dustm = lnprob_grid(dustgrid, obs, err, mask)

    #output
    ubar = dl07.ubar(dustgrid.pars['UMIN'],dustgrid.pars['UMAX'],dustgrid.pars['GAMMA'])
    max_lnprob[ix,iy] = np.max(lnprob)
    allpars = np.vstack([ltir,dustm,ubar,dustgrid.pars['UMIN'],dustgrid.pars['UMAX'],
                       dustgrid.pars['GAMMA'],dustgrid.pars['QPAH']])
    outparnames=['LDUST','MDUST','UBAR']+par_names

    #get the percentiles of the 1d marginalized posterior distribution
    for ipar in xrange(len(outparnames)):
        par = np.sort(allpars[ipar,:])
        order = np.argsort(par)
        cdf = np.cumsum(np.exp(lnprob[order])) / np.sum(np.exp(lnprob))
        ind_ptiles= np.searchsorted(cdf,percentiles)
        parval[ix,iy,ipar,:] = par[order[ind_ptiles]]

    if ipix % 1 == 0:
        pass
        #plot and try to output the full sampler
                #pass
    if ipix == 20:
        pass
        #raise ValueError("Stop at ipix %s" %ipix)

#write out the parval images
for i in xrange(len(outparnames)):
    #    header.set('BUNIT',unit[i])
    for j in xrange(3):
        outfile= 'results/%s_p%4.2f.fits' % (outparnames[i],percentiles[j]) 
        pyfits.writeto(outfile,parval[:,:,i,j],header=header,clobber=True)


