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
import emcee

import observate
import dustmodel
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

#plotting of posterior samples
def plot_all_samples2d(sourcename):
    pl.figure(1,(12.0,12.0),dpi=600)
    for i in range(npar):
        for j in range(i,npar):
            pl.subplot(npar,npar,1+j*npar+i)
                
            if i != j:
                pl.plot( allpars[i,:], allpars[j,:],
                         linestyle='none', marker='o', color='red', mec='red',
                        alpha=.3, label='Posterior', zorder=-99, ms=1.0)
                if par_log[i] == 1 : pl.xscale('log')
                if par_log[j] == 1 : pl.yscale('log')
                if i == 0 :
                    locs,labs = pl.yticks()
                    labs=[]
                    for k in range(locs.shape[0]) : labs.append(str(locs[k]))
                    pl.yticks(locs,labs,fontsize=6)
                    pl.ylabel(par_names[j],fontsize=7)                
                else:
                    pl.yticks([])
                if j == (npar-1) :
                    locs,labs=pl.xticks()
                    labs=[]
                    for k in range(locs.shape[0]) : labs.append(str(locs[k]))
                    pl.xlabel(par_names[i],fontsize=7)
                    pl.xticks(locs,labs,fontsize=6)
                else:
                    pl.xticks([])

            else:
                pl.hist(allpars[i,:],30)
                ax = pl.gca()
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                if par_log[i] == 1 : pl.xscale('log')
                if par_log[j] == 1 : pl.yscale('log')
                locs, labs = pl.xticks()
                labs=[]
                for k in range(locs.shape[0]) : labs.append(str(locs[k]))
                #pl.ylabel('',fontsize=7)
                pl.xticks(locs,labs,fontsize=6)
                pl.yticks([])
                pl.xlabel(par_names[i],fontsize=7)
                
            #tight_layout(0.1)
    pl.savefig('%s_param_dist.png' %sourcename)
    pl.close(1)
    

#####
##### function to obtain model SED #####
#####
def model(umin,umax,gamma,qpah,mdust=1):
    alpha = 2.0 #model library does not have other alphas
    spec = (dl07.generateSpectrum(umin,umax,gamma,qpah,alpha,mdust))[:,0]
    sed = observate.getSED(dl07.wavelength,spec,filterlist)

    #Units are L_sun/M_sun, i.e. this is U_bar*P_o.
    lbol = dl07.convert_to_lsun*observate.Lbol(dl07.wavelength,spec,wave_min=wave_min,wave_max=wave_max)

    return sed, lbol

#####
##### function to obtain likelihood ####
#####
#def lnprob(theta,obs_maggies,obs_ivar,mask):
#@profile
def lnprob(theta, obs, err, mask):

    #prior bounds check
    pcheck=[]
    for i,par in enumerate(theta):
        pcheck.append(par >= par_range[i,0])
        pcheck.append(par <= par_range[i,1])
    pcheck.append(10**theta[1] >= theta[0]) #require Umax >= Umin
    
    if not (False in pcheck):
        #model sed (in AB absolute mag) for these parameters
        sed, lbol = model(theta[0],10**theta[1],theta[2],theta[3])
        
        #linearize fluxes.  could move obs out of the loop
        inds = np.where(mask > 0)
        sed_maggies = 10**(0-sed[inds]/2.5)
        obs_maggies = 10**(0-obs[inds]/2.5)
        obs_ivar = (obs_maggies*err[inds]/1.086)**(-2)

        #best scale for these parameters
        mdust = ( (sed_maggies*obs_maggies*obs_ivar).sum() ) / ( ( (sed_maggies**2.0)*obs_ivar ).sum() )
        lbol = lbol*mdust
        
        #probability
        chi2 = ( (mdust*sed_maggies - obs_maggies)**2 )*obs_ivar*mask
        #print(ascale,chi2.sum())
        lnprob = -0.5*chi2.sum()
        #delta_mag = ( sed-2.5*np.log10(ascale) ) - obs
        
    else:
        #set lnp to -infty if parameters out of prior bounds
        lnprob = -np.infty
        lbol, mdust = -1, -1

    return lnprob, [lbol, mdust]

############### MAIN ####################

##### load the filters and the DL07 model grid
filterlist=observate.loadFilters(fnamelist)
dl07=dustmodel.DraineLi()

##### parameter ranges for priors
par_names = ['Umin','Umax','gamma','Qpah']
par_range = np.array([[0.1,25],
                      [100.0,dl07.model_lib['UMAX'].max()],
                      [0.00,0.5],
            [dl07.model_lib['QPAH'].min(),dl07.model_lib['QPAH'].max()]])

    #par_range[0,:] = np.log10(par_range[0,:])
par_range[1,:] = np.log10(par_range[1,:])

par_names = ['Ldust','Mdust','Ubar']+par_names
#par_units = 
par_log=[0,0,0,0,1,0,0]

##### read the images and errors
data_mag, data_magerr, header = utils.loadImageCube(imnamelist,errnamelist,fudge_err)
dm = 5.0*np.log10(dist)+25
data_mag = np.where(data_mag != 0.0, data_mag-dm, 0.0)
nx, ny = data_mag.shape[0], data_mag.shape[1]

##### set up output
percentiles = np.array([0.16,0.5,0.84]) #output percentiles
npar = len(par_names) #the ndim + mdust, ldust, ubar
parval = np.zeros([nx,ny,npar,3]) 
accepted = np.zeros([nx,ny])-99
max_lnprob = np.zeros([nx,ny])-99

############################
#### cry 'havoc'! ##########
############################
#this is insane and slow when loping over pixels  Should just use a precomputed model grid

##### Sampler properties
ndim = 4
nwalkers = 12
nsteps = 30
nburn = 15
nthreads = 2

####### Loop over pixels #######
#should change the loop to prefilter bad pixels
g = np.where(data_mag < 0,1,0)
g = np.where(np.isfinite(data_mag),g,0)
goodpix = np.where(g.sum(2) == len(imnamelist)) #restrict to pixels with at least 5 bands

#raise ValueError("debug")

for ipix in xrange(goodpix[0].shape[0]):
    start = time.time()
    ix, iy  = goodpix[0][ipix*10+2815], goodpix[1][ipix*10+2815]

    obs = data_mag[ix,iy,:]
    err = data_magerr[ix,iy,:]
    mask=np.where(np.logical_and( (obs < 0), np.isfinite(obs) ), 1, 0)

    #set up initial proposal
    initial=np.zeros([nwalkers,ndim])
    for j in xrange(ndim) :
        initial[:,j]=np.random.uniform(par_range[j,0],par_range[j,1],nwalkers)

    #get a sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[obs,err,mask] )
            
    # burn it in from initial then reset
    pos,prob,state,blob = sampler.run_mcmc(initial, nburn)
    sampler.reset()

    #sample for real
    sampler.run_mcmc(np.array(pos),nsteps, rstate0=state)

    #diagnostics
    duration = time.time()-start
    print('Sampling done in', duration, 's')
    print("Mean acceptance fraction: {0:.3f}"
        .format(np.mean(sampler.acceptance_fraction)))
    accepted[ix,iy]=np.mean(sampler.acceptance_fraction)
    if (np.mean(sampler.acceptance_fraction) == 0.0) :
        raise ValueError("No Good Models") #debugging
    accepted[ix,iy]=np.mean(sampler.acceptance_fraction)
    max_lnprob[ix,iy]=sampler.flatlnprobability.max()
    
    #output
    nn = np.array(sampler.blobs)
    dustl = nn[:,:,0].reshape(nsteps*nwalkers)
    dustm = nn[:,:,1].reshape(nsteps*nwalkers)
    umin = 10**sampler.flatchain[:,0]
    umax = 10**sampler.flatchain[:,1]
    gamma = sampler.flatchain[:,2]
    qpah = sampler.flatchain[:,3]
    ubar=(1-gamma)*(umin)+gamma*np.log(umax/umin)/(1/umin-1/umax)

    allpars=np.vstack((dustl,dustm,ubar,umin,umax,gamma,qpah))
    sampler.reset() #done with the sampler

    #get the percentiles of the 1d marginalized posterior distribution
    for ipar in xrange(npar):
        par=np.sort(allpars[ipar,:])
        parval[ix,iy,ipar,:]=par[np.int_(np.round(par.shape[0]*percentiles))]

    if ipix % 1 == 0:
        #plot and try to output the full sampler
        plot_all_samples2d('results/x%s_y%s'%(ix,iy))
        #pass
    if ipix == 20:
        raise ValueError("Stop at ipix %s" %ipix)

#write out the parval images
for i in xrange(npar):
    #    header.set('BUNIT',unit[i])
    for j in xrange(3):
        outfile= '%s_p%4.2f.fits' % (par_names[i],percentiles[j]) 
        pyfits.writeto(outfile,parval[:,:,i,j],header=header,clobber=True)


