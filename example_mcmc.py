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
import datacube

import matplotlib.pyplot as pl
#from dustplot import *

############### USER INPUTS ###########
rp = {'outname': 'NGC6822', 'dist': 0.490, 'ngrid': 5e4,
      'wave_min':15e4, 'wave_max': 1e7, #AA, range for determination of L_TIR
      'percentiles':np.array([0.025,0.5,0.975]) #output percentiles
      }

##### Filter info 
#list the filters you want to use (in the same order
#as the images below).  These are based on k_correct

rp['fnamelist'] = ['spitzer_mips_24','herschel_pacs_70','herschel_pacs_100',
             'herschel_pacs_160','herschel_spire_250']

##### Image file names
#images should be convolved to a common resolution, pixel matched,
#and in units of Jy/pixel. Otherwise modify utils.loadImageCube
#or simply add -2.5*log(conversion) to the magnitude arrays

imnamelist = ['mips_24.6arcsec.Jypix','pacs_70.6arcsec.Jypix',
              'pacs_100.6arcsec.Jypix','pacs_160.6arcsec.Jypix',
              'spire_250.6arcsec.Jypix']
errnamelist = ['x','pacs_70.6arcsec.sig','pacs_100.6arcsec.sig',
               'pacs_160.6arcsec.sig','x']
rp['fudge_err'] = [0.1,0.1,0.1,0.1,0.15] #errors to use when true error images don't exist

rp['impath'] = os.getenv('pyxydust')+'/imdata/NGC6822'
rp['imnamelist'] = ['{0}_conv250_{1}.fits'.format(rp['impath'],name) for name in imnamelist]
rp['errnamelist'] = ['{0}_{1}.fits'.format(rp['impath'],name) for name in errnamelist]


##### load the filters and the DL07 model grid
dl07 = dustmodel.DraineLi()
filterlist = observate.load_filters(rp['fnamelist'])


############### END USER INPUT ##########

############### MAIN ####################
#do it this way so the main function can come first
def main(rp):
    print('starting main')
    ##### read the images and errors
    data_mag, data_magerr, header = datacube.loadImageCube(rp['imnamelist'],rp['errnamelist'],rp['fudge_err'])
    dm = 5.0*np.log10(rp['dist'])+25
    data_mag = np.where(data_mag != 0.0, data_mag-dm, 0.0)
    nx, ny = data_mag.shape[0], data_mag.shape[1]

    
    ##### parameter ranges for priors
    rp['par_names'] = ['Umin','Umax','gamma','Qpah', 'Mdust']

    rp['par_range'] = np.array([[0.1,25],
                          dl07.par_range(['UMAX'],inds=dl07.pdr_inds)[0],
                          [0.00,0.5],
                          dl07.par_range(['QPAH'])[0],
                  [0, 1e10]])           

    #par_range[0,:] = np.log10(par_range[0,:])
    rp['par_range'][1,:] = np.log10(rp['par_range'][1,:])

    rp['outpar_names'] = ['Ldust','Ubar']+rp['par_names']
    #par_units = 
    rp['outpar_log']=[0,0,0,1,0,0,0]
    
    ##### set up output
    noutpar = len(rp['outpar_names']) #the ndim + mdust, ldust, ubar
    parval = np.zeros([nx,ny,noutpar,3]) 
    accepted = np.zeros([nx,ny])-99
    max_lnprob = np.zeros([nx,ny])-99
    
    ##### Sampler properties
    rp['ndim'] = len(rp['par_names'])
    rp['nwalkers'] = rp['ndim'] * 10
    rp['nburn'], rp['nsteps'] = 10, 50
    nthreads = 1
    
    ####### Loop over pixels #######
    gg = np.where((data_mag < 0) & np.isfinite(data_mag),1,0)
    goodpix = np.where(gg.sum(axis = 2) == len(rp['imnamelist'])) #restrict to pixels with all detections 

    for ipix in xrange(goodpix[0].shape[0]):
        start = time.time()

        #ix, iy  = goodpix[0][ipix], goodpix[1][ipix]
        ix, iy  = goodpix[0][ipix*10+2815], goodpix[1][ipix*10+2815] #skip around middle of image for testing
        obs, err  = data_mag[ix,iy,:], data_magerr[ix,iy,:]
        obs_maggies = 10**(0-obs/2.5)
        obs_ivar = (obs_maggies*err/1.086)**(-2)
        mask = np.where((obs < 0) & np.isfinite(obs) , 1, 0)
        
        #set up initial proposal
        initial=np.zeros([rp['nwalkers'],rp['ndim']])
        for j in xrange(rp['ndim']) :
            initial[:,j]=np.random.uniform(rp['par_range'][j,0],rp['par_range'][j,1],rp['nwalkers'])

        #get a sampler, burn it in, and reset
        sampler = emcee.EnsembleSampler(rp['nwalkers'], rp['ndim'], lnprob, threads=nthreads,
                                        args = [obs_maggies,obs_ivar,mask, rp] )
        pos,prob,state,blob = sampler.run_mcmc(initial, rp['nburn'])
        sampler.reset()
        #cry havoc
        sampler.run_mcmc(np.array(pos),rp['nsteps'], rstate0=state)

        #diagnostics
        #print(sampler.acor)
        rp['duration'] = time.time()-start
        rp['maf'] = np.mean(sampler.acceptance_fraction)
        print('Sampling done in {duration:0.1f}s with mean acceptance fraction {maf:.3f}'.format(**rp))
        if (rp['maf'] == 0.0) :
            raise ValueError("No Good Models") #debugging
        accepted[ix,iy] = rp['maf']
        max_lnprob[ix,iy] = sampler.flatlnprobability.max()
    
        #output
        allpars = sampler_to_output(sampler, parval, rp)
        #get the percentiles of the 1d marginalized posterior distribution
        for ipar in xrange(len(rp['outpar_names'])):
            par = np.sort(allpars[ipar,:])
            parval[ix,iy,ipar,:]=par[np.int_(np.floor(par.shape[0]*rp['percentiles']))]
        #get the MAP values?

        if ipix % 1 == 0:
            #plot and try to output the full sampler
            plot_all_samples2d('results/x{0}_y{1}'.format(ix,iy), allpars, rp)
        if ipix == 20:
            raise ValueError("Stop at ipix {0}".format(ipix))

        sampler.reset() #done with the sampler

    #write out the parval images
    for i in xrange(noutpar):
        # header.set('BUNIT',unit[i])
        for j in xrange(3):
            outfile= '{0}_p{1:4.2f}.fits' % (outpar_names[i],rp['percentiles'][j]) 
            pyfits.writeto(outfile,parval[:,:,i,j],header=header,clobber=True)



#########
##### FUNCTION TO OBTAIN MODEL SED #####
def model(umin,umax,gamma,qpah,mdust=1):
    alpha = 2.0 #model library does not have other alphas
    
    spec = (dl07.generate_spectrum(umin,umax,gamma,qpah,alpha,mdust))#[:,0]
    sed = observate.getSED(dl07.wavelength,spec,filterlist)

    #Units are L_sun/M_sun, i.e. this is U_bar*P_o.
    lbol = dl07.convert_to_lsun*observate.Lbol(dl07.wavelength,np.squeeze(spec),
                                               wave_min = rp['wave_min'],
                                               wave_max = rp['wave_max'])

    return sed, lbol


#########
##### FUNCTION TO OBTAIN LIKELIHOOD ####
def lnprob(theta, obs_maggies, obs_ivar, mask, rp):

    #prior bounds check
    ptest=[]
    for i,par in enumerate(theta):
        ptest.append(par >= rp['par_range'][i,0])
        ptest.append(par <= rp['par_range'][i,1])
    ptest.append(10**theta[1] >= theta[0]) #require Umax >= Umin

    if False in ptest:
        #set lnp to -infty if parameters out of prior bounds
        lnprob = -np.infty
        lbol = -1

    else:
        #model sed (in AB absolute mag) for these parameters
        sed, lbol = model(theta[0],10**theta[1],theta[2],theta[3], mdust = theta[4])
        sed_maggies = 10**(0-sed/2.5)

        #best scale for these parameters.  this should probably be another parameter
        #mdust = ( (sed_maggies*obs_maggies*obs_ivar).sum() ) / ( ( (sed_maggies**2.0)*obs_ivar ).sum() )
        #lbol = lbol*mdust
        
        #probability
        chi2 = ( (sed_maggies - obs_maggies)**2 )*obs_ivar
        inds = np.where(mask > 0)
        lnprob = -0.5*chi2[inds].sum()
        #delta_mag = ( sed- obs )

    return lnprob, [lbol]

########
####FUNCTION TO CONVERT THE EMCEE SAMPLER OUTPUT TO A PARAMETER ARRAY
def sampler_to_output(sampler, parval, rp):
    
    nn = np.array(sampler.blobs)
    dustl = nn[:,:,0].reshape(rp['nsteps']*rp['nwalkers'])
    dustm = sampler.flatchain[:,4]
    umin = sampler.flatchain[:,0]
    umax = 10**sampler.flatchain[:,1]
    gamma = sampler.flatchain[:,2]
    qpah = sampler.flatchain[:,3]
    ubar=(1-gamma)*(umin)+gamma*np.log(umax/umin)/(1/umin-1/umax)

    return np.vstack((dustl,ubar,umin,umax,gamma,qpah,dustm))
    
####plotting of posterior samples
def plot_all_samples2d(sourcename, allpars, rp):
    pl.figure(1,(12.0,12.0),dpi=600)
    npar = len(rp['outpar_names'])
    for i in range(npar):
        for j in range(i,npar):
            pl.subplot(npar,npar,1+j*npar+i)
                
            if i != j:
                pl.plot( allpars[i,:], allpars[j,:],
                         linestyle='none', marker='o', color='red', mec='red',
                        alpha=.3, label='Posterior', zorder=-99, ms=1.0)
                if rp['outpar_log'][i] == 1 : pl.xscale('log')
                if rp['outpar_log'][j] == 1 : pl.yscale('log')
                if i == 0 :
                    locs,labs = pl.yticks()
                    labs=[]
                    for k in range(locs.shape[0]) : labs.append(str(locs[k]))
                    pl.yticks(locs,labs,fontsize=6)
                    pl.ylabel(rp['outpar_names'][j],fontsize=7)                
                else:
                    pl.yticks([])
                if j == (npar-1) :
                    locs,labs=pl.xticks()
                    labs=[]
                    for k in range(locs.shape[0]) : labs.append(str(locs[k]))
                    pl.xlabel(rp['outpar_names'][i],fontsize=7)
                    pl.xticks(locs,labs,fontsize=6)
                else:
                    pl.xticks([])

            else:
                pl.hist(allpars[i,:],30)
                ax = pl.gca()
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                if rp['outpar_log'][i] == 1 : pl.xscale('log')
                
                locs, labs = pl.xticks()
                labs=[]
                for k in range(locs.shape[0]) : labs.append(str(locs[k]))
                #pl.ylabel('',fontsize=7)
                pl.xticks(locs,labs,fontsize=6)
                pl.yticks([])
                pl.xlabel(rp['outpar_names'][i],fontsize=7)
                
            #tight_layout(0.1)
    pl.savefig('%s_param_dist.png' %sourcename)
    pl.close(1)
    
if __name__ == '__main__' :
      main(rp)
      
